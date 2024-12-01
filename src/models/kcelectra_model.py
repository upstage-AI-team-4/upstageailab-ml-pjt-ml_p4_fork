import torch
import torch.nn as nn
from transformers import ElectraForSequenceClassification
import pytorch_lightning as pl
from typing import Any, Dict
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
import torchmetrics

class KcELECTRA(pl.LightningModule):
    def __init__(
        self,
        pretrained_model: str = "beomi/KcELECTRA-base",
        num_labels: int = 2,
        num_unfreeze_layers: int = -1,
        **kwargs
    ):
        """
        KcELECTRA 모델
        
        Args:
            pretrained_model: 사전학습 모델 경로
            num_labels: 레이블 수
            num_unfreeze_layers: Fine-tuning할 레이어 수 (-1: 전체)
        """
        super().__init__()
        self.save_hyperparameters()
        
        # 모델 설정
        self.model = ElectraForSequenceClassification.from_pretrained(
            pretrained_model,
            num_labels=num_labels
        )
        
        # Freeze layers if specified
        if num_unfreeze_layers > 0:
            self._freeze_layers(num_unfreeze_layers)
        
        # 학습 파라미터 초기화
        self.lr = 2e-5
        self.optimizer_name = 'AdamW'
        self.scheduler_name = 'cosine'
    
    def _freeze_layers(self, num_unfreeze_layers: int):
        """레이어 동결 설정"""
        # 모든 파라미터 동결
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 분류기 레이어는 항상 학습
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        
        if num_unfreeze_layers == -1:
            # 모든 레이어 학습
            for param in self.model.parameters():
                param.requires_grad = True
        elif num_unfreeze_layers > 0:
            # 지정된 수의 인코더 레이어만 학습
            # ELECTRA는 일반적으로 12개의 레이어를 가짐
            total_layers = len(self.model.electra.encoder.layer)
            for i in range(total_layers - num_unfreeze_layers, total_layers):
                for param in self.model.electra.encoder.layer[i].parameters():
                    param.requires_grad = True
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def training_step(self, batch, batch_idx):
        """학습 단계"""
        outputs = self(**batch)
        loss = outputs.loss
        
        # 메트릭스 계산
        predictions = torch.argmax(outputs.logits, dim=1)
        correct = (predictions == batch['labels']).sum()
        total = len(predictions)
        accuracy = correct.float() / total
        
        # F1, Precision, Recall 계산
        tp = ((predictions == 1) & (batch['labels'] == 1)).sum().float()
        fp = ((predictions == 1) & (batch['labels'] == 0)).sum().float()
        fn = ((predictions == 0) & (batch['labels'] == 1)).sum().float()
        tn = ((predictions == 0) & (batch['labels'] == 0)).sum().float()
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        
        # 로깅
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_accuracy', accuracy, prog_bar=True)
        self.log('train_f1', f1, prog_bar=True)
        self.log('train_precision', precision, prog_bar=True)
        self.log('train_recall', recall, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """검증 단계"""
        outputs = self(**batch)
        loss = outputs.loss
        
        # 메트릭스 계산
        predictions = torch.argmax(outputs.logits, dim=1)
        correct = (predictions == batch['labels']).sum()
        total = len(predictions)
        accuracy = correct.float() / total
        
        # F1, Precision, Recall 계산
        tp = ((predictions == 1) & (batch['labels'] == 1)).sum().float()
        fp = ((predictions == 1) & (batch['labels'] == 0)).sum().float()
        fn = ((predictions == 0) & (batch['labels'] == 1)).sum().float()
        tn = ((predictions == 0) & (batch['labels'] == 0)).sum().float()
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        
        # 로깅
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', accuracy, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)
        self.log('val_precision', precision, prog_bar=True)
        self.log('val_recall', recall, prog_bar=True)
    
    def configure_optimizers(self):
        """옵티마이저와 스케줄러 설정"""
        # Optimizer
        if self.optimizer_name.lower() == 'adamw':
            optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        elif self.optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
        
        # Scheduler
        if self.scheduler_name.lower() == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=0.9)
        elif self.scheduler_name.lower() == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=10)
        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler_name}")
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }
