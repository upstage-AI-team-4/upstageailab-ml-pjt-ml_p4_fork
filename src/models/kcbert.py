import os
import pandas as pd

from pprint import pprint

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR

from pytorch_lightning import LightningModule, Trainer, seed_everything

from transformers import BertForSequenceClassification, BertTokenizer, AdamW

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts
import re
import emoji
from soynlp.normalizer import repeat_normalize

from pathlib import Path

import mlflow
from datetime import datetime
import numpy as np

class Arg:
    def __init__(self):
        self.random_seed: int = 42
        self.base_dir: Path = Path(__file__).parent.parent.parent
        
        # 모델 관련 설정
        self.pretrained_model: str = str(self.base_dir / 'models' / 'pretrained' / 'KcBERT')
        self.pretrained_tokenizer: str = ''
        
        # 데이터 관련 설정
        self.data_dir: Path = self.base_dir / 'data' / 'raw' / 'naver_movie_review'
        self.train_data_path: Path = self.data_dir / 'ratings_train.txt'
        self.val_data_path: Path = self.data_dir / 'ratings_test.txt'
        
        # 학습 관련 설정
        self.batch_size: int = 32
        self.lr: float = 5e-6
        self.epochs: int = 20
        self.max_length: int = 150
        self.report_cycle: int = 100
        self.cpu_workers: int = os.cpu_count()
        self.test_mode: bool = False
        self.optimizer: str = 'AdamW'
        self.lr_scheduler: str = 'exp'
        self.fp16: bool = False

args = Arg()

class KcBERTClassifier(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bert = BertForSequenceClassification.from_pretrained(
            self.args.pretrained_model,
            num_labels=2
        )
        self.tokenizer = BertTokenizer.from_pretrained(
            self.args.pretrained_tokenizer
            if self.args.pretrained_tokenizer
            else self.args.pretrained_model
        )
        
        # MLflow 설정
        self.run_id = None
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("sentiment_classification_kcbert")
        
        # 메트릭을 저장할 리스트
        self.validation_step_outputs = []
        self.training_step_outputs = []
        
        # 데이터 로드 및 전처리
        self.train_data = None
        self.val_data = None
        self.read_data()
        
    def read_data(self):
        """데이터 파일 읽기"""
        print(f"데이터 로드 중... \n- train: {self.args.train_data_path}\n- val: {self.args.val_data_path}")
        self.train_data = pd.read_csv(self.args.train_data_path, sep='\t')
        self.val_data = pd.read_csv(self.args.val_data_path, sep='\t')
        
        # 전처리
        self.train_data = self.preprocess_dataframe(self.train_data)
        self.val_data = self.preprocess_dataframe(self.val_data)
        
        print(f"데이터 로드 완료: train({len(self.train_data)}), val({len(self.val_data)})")

    def preprocess_dataframe(self, df):
        """데이터프레임 전처리"""
        emojis = ''.join(emoji.EMOJI_DATA.keys())
        pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
        url_pattern = re.compile(
            r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

        def clean(x):
            x = pattern.sub(' ', x)
            x = url_pattern.sub('', x)
            x = x.strip()
            x = repeat_normalize(x, num_repeats=2)
            return x

        df['document'] = df['document'].map(lambda x: self.tokenizer.encode(
            clean(str(x)),
            padding='max_length',
            max_length=self.args.max_length,
            truncation=True,
        ))
        return df

    def train_dataloader(self):
        """학습 데이터로더"""
        dataset = TensorDataset(
            torch.tensor(self.train_data['document'].tolist()),
            torch.tensor(self.train_data['label'].tolist())
        )
        
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.cpu_workers
        )

    def val_dataloader(self):
        """검증 데이터로더"""
        dataset = TensorDataset(
            torch.tensor(self.val_data['document'].tolist()),
            torch.tensor(self.val_data['label'].tolist())
        )
        
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.cpu_workers
        )

    def forward(self, **kwargs):
        return self.bert(**kwargs)

    def training_step(self, batch, batch_idx):
        data, labels = batch
        output = self(input_ids=data, labels=labels)
        loss = output.loss
        logits = output.logits
        preds = logits.argmax(dim=-1)

        step_output = {
            'loss': loss,
            'y_true': labels.cpu().numpy(),
            'y_pred': preds.cpu().numpy(),
        }
        
        self.training_step_outputs.append(step_output)
        return step_output

    def on_train_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
        y_true = []
        y_pred = []
        
        for out in self.training_step_outputs:
            y_true.extend(out['y_true'])
            y_pred.extend(out['y_pred'])
            
        metrics = [
            metric(y_true=y_true, y_pred=y_pred)
            for metric in (accuracy_score, precision_score, recall_score, f1_score)
        ]

        mlflow.log_metrics({
            'train_loss': avg_loss.item(),
            'train_acc': metrics[0],
            'train_precision': metrics[1],
            'train_recall': metrics[2],
            'train_f1': metrics[3],
        }, step=self.global_step)

        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        output = self(input_ids=data, labels=labels)
        loss = output.loss
        logits = output.logits
        preds = logits.argmax(dim=-1)

        step_output = {
            'loss': loss,
            'y_true': labels.cpu().numpy(),
            'y_pred': preds.cpu().numpy(),
        }
        
        self.validation_step_outputs.append(step_output)
        return step_output

    def on_validation_epoch_end(self):
        val_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        y_true = []
        y_pred = []

        for out in self.validation_step_outputs:
            y_true.extend(out['y_true'])
            y_pred.extend(out['y_pred'])

        metrics = [
            metric(y_true=y_true, y_pred=y_pred)
            for metric in (accuracy_score, precision_score, recall_score, f1_score)
        ]

        mlflow.log_metrics({
            'val_loss': val_loss.item(),
            'val_acc': metrics[0],
            'val_precision': metrics[1],
            'val_recall': metrics[2],
            'val_f1': metrics[3],
        }, step=self.global_step)
        
        self.validation_step_outputs.clear()
        
        return {'val_loss': val_loss, 'val_metrics': metrics}

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.args.lr)
        
        if self.args.lr_scheduler == 'cos':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
        else:
            scheduler = ExponentialLR(optimizer, gamma=0.5)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

    def save_model(self, save_dir: Path):
        """모델과 토크나이저 저장"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델 저장
        self.bert.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        # MLflow에 모델 아티팩트 로깅
        if self.run_id:
            mlflow.transformers.log_model(
                transformers_model={
                    "model": self.bert,
                    "tokenizer": self.tokenizer
                },
                artifact_path="model",
                task="sentiment-analysis",
                registered_model_name=f"KcBERT_sentiment_classifier_base_epoch_{self.args.epochs}"
            )
        
        print(f"모델이 {save_dir}에 저장되었습니다.")

def main():
    print("Using PyTorch Ver", torch.__version__)
    print("Fix Seed:", args.random_seed)
    seed_everything(args.random_seed)
    model = KcBERTClassifier(args)

    # MLflow 실행 시작
    with mlflow.start_run() as run:
        model.run_id = run.info.run_id
        
        # 하이퍼파라미터 로깅
        mlflow.log_params({
            'learning_rate': args.lr,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'optimizer': args.optimizer,
            'scheduler': args.lr_scheduler,
        })
        
        print(":: Start Training ::")
        trainer = Trainer(
            max_epochs=args.epochs,
            fast_dev_run=args.test_mode,
            num_sanity_val_steps=None if args.test_mode else 0,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            precision=16 if args.fp16 else 32,
        )
        
        # 배치 사이즈가 지정되지 않은 경우 기본값 사용
        if not args.batch_size:
            args.batch_size = 32  # 기본 배치 사이즈
            print(f"배치 사이즈 설정: {args.batch_size}")
        
        trainer.fit(model)
        
        # 모델 저장
        save_dir = Path(args.base_dir) / 'models' / 'finetuned' / 'kcbert_baseline' / f'kcbert_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        model.save_model(save_dir)

if __name__ == "__main__":
    main()