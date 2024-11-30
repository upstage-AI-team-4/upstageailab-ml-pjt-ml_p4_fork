# src/models/classifiers.py
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ExponentialLR
import torchmetrics
from typing import Dict, Any
from .base_model import BaseTextClassifier
from pathlib import Path
import os
import shutil
import torch
# config
# pretrained_model
# num_labels
# optimizer
# lr_scheduler  
class KcBERT(BaseTextClassifier):
    def __init__(
        self,
        pretrained_model: str,
        model_dir: Path,
        model_name: str,
        num_labels: int,
        learning_rate: float,
        optimizer: str = 'AdamW',
        lr_scheduler: str = 'cos',
        num_unfreeze_layers: int = -1  # -1: 모든 레이어 학습, 0: 분류기만 학습
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # 설정값들을 인스턴스 변수로 저장
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer
        self.scheduler_type = lr_scheduler
        self.num_unfreeze_layers = num_unfreeze_layers
        
        self.model_path = model_dir / model_name
        os.makedirs(self.model_path, exist_ok=True)
        self.model = self._load_model(pretrained_model, num_labels)
        self.tokenizer = self._load_tokenizer(pretrained_model)
        
        # 레이어 동결 설정
        self._set_layers_trainable()
        
        self.metrics = torchmetrics.MetricCollection({
            'accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=num_labels),
            'f1': torchmetrics.F1Score(task='multiclass', num_classes=num_labels),
            'precision': torchmetrics.Precision(task='multiclass', num_classes=num_labels),
            'recall': torchmetrics.Recall(task='multiclass', num_classes=num_labels)
        })
        
        self.train_metrics = self.metrics.clone(prefix='train_')
        self.val_metrics = self.metrics.clone(prefix='val_')
    
    def _check_model_files(self) -> bool:
        """모델 파일들이 로컬에 존재하는지 확인"""
        required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json', 'tokenizer_config.json', 'vocab.txt']
        return all((self.model_path / file).exists() for file in required_files)
    
    def _load_model(self, pretrained_model: str, num_labels: int):
        if self._check_model_files():
            print(f"Loading model from local path: {self.model_path}")
            return BertForSequenceClassification.from_pretrained(
                str(self.model_path),
                num_labels=num_labels
            )
        else:
            print(f"Downloading model from HuggingFace: {pretrained_model}")
            model = BertForSequenceClassification.from_pretrained(
                pretrained_model,
                num_labels=num_labels
            )
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(self.model_path))
            return model
    
    def _load_tokenizer(self, pretrained_model: str):
        """로컬 경로에서 토크나이저를 로드하거나 허깅페이스에서 다운로드"""
        try:
            if self._check_model_files():
                print(f"Loading tokenizer from local path: {self.model_path}")
                return BertTokenizer.from_pretrained(str(self.model_path))
            else:
                print(f"Downloading tokenizer from HuggingFace: {pretrained_model}")
                tokenizer = BertTokenizer.from_pretrained(pretrained_model)
                # 토크나이저 저장
                tokenizer.save_pretrained(str(self.model_path))
                return tokenizer
        except Exception as e:
            print(f"Error loading tokenizer: {str(e)}")
            raise
    
    def forward(self, **kwargs):
        return self.model(**kwargs)
    
    def _compute_loss(self, batch):
        data, labels = batch
        outputs = self(input_ids=data, labels=labels)
        return outputs.loss
    
    def _compute_metrics(self, y_true, y_pred):
        metrics = self.train_metrics(y_pred, y_true)
        return {k: v.item() for k, v in metrics.items()}
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        optimizer = self._get_optimizer()
        scheduler = self._get_scheduler(optimizer)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1
            }
        }
    
    def _set_layers_trainable(self, verbose=False):
        """Set which layers to train"""
        # 모든 파라미터 동결
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 분류기 레이어는 항상 학습
        for param in self.model.classifier.parameters():
            param.requires_grad = True
            
        if self.num_unfreeze_layers == -1:
            # 모든 레이어 학습
            for param in self.model.parameters():
                param.requires_grad = True
        elif self.num_unfreeze_layers > 0:
            # ���정된 수의 인코더 레이어만 학습
            # BERT는 일반적으로 12개의 레이어를 가짐
            total_layers = len(self.model.bert.encoder.layer)
            for i in range(total_layers - self.num_unfreeze_layers, total_layers):
                for param in self.model.bert.encoder.layer[i].parameters():
                    param.requires_grad = True
                    
        # 학습 가능한 파라미터 수 출력
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.2f}%)")
        
        if verbose:
            # 레이어별 학습 여부 출력
            print("\nTrainable layers:")
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(f"✓ {name}")
                else:
                    print(f"✗ {name}")
    
    def _get_optimizer(self):
        """Get optimizer with different learning rates for different layers"""
        no_decay = ['bias', 'LayerNorm.weight']
        
        # 파라미터 그룹화
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0
            }
        ]
        
        if self.optimizer_type == 'AdamW':
            return AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        raise ValueError(f"Unsupported optimizer: {self.optimizer_type}")
    
    def _get_scheduler(self, optimizer):
        """Get learning rate scheduler based on configuration"""
        if self.scheduler_type == 'cos':
            return CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
        elif self.scheduler_type == 'exp':
            return ExponentialLR(optimizer, gamma=0.5)
        raise ValueError(f"Unsupported scheduler: {self.scheduler_type}")
    
    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        
        # 메트릭 계산
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        self.train_metrics(predictions, batch['labels'])
        
        # 로그 기록
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        
        # 메트릭 계산
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        self.val_metrics(predictions, batch['labels'])
        
        # 로그 기록
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)
        
        return loss