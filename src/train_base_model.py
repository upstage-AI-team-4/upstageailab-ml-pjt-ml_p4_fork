import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

from utils.config import Config
from utils.mlflow_utils import MLflowLogger
from models.model_registry import ModelRegistry
import logging

logger = logging.getLogger(__name__)

config = Config()

class Arg:
    def __init__(self):
        self.random_seed: int = 42
        self.base_dir: Path = Path(__file__).parent.parent
        
        # 모델 관련 설정
        self.pretrained_model: str = str(self.base_dir / 'models' / 'pretrained' / 'KcBERT')
        self.pretrained_tokenizer: str = ''
        
        # 데이터 관련 설정
        self.data_dir: Path = self.base_dir / 'data' / 'raw' / 'naver_movie_review'
        self.train_data_path: Path = self.data_dir / 'ratings_train.txt'
        self.val_data_path: Path = self.data_dir / 'ratings_test.txt'
        print(f'Data directory: {self.data_dir}\nModel directory: {self.pretrained_model}')
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
        
        # MLflow 로거 추가
        self.mlflow_logger = MLflowLogger()
        
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
        
        # 샘플링은 한 번만 수행
        if config.data['sampling_rate'] != 1.0:
            self.train_data = self.train_data.sample(frac=config.data['sampling_rate'], random_state=42)
            self.val_data = self.val_data.sample(frac=config.data['sampling_rate'], random_state=42)
            print(f"샘플링 완료: train({len(self.train_data)}), val({len(self.val_data)})")

        # 전처리
        self.train_data_preped = self.preprocess_dataframe(self.train_data)
        self.val_data_preped = self.preprocess_dataframe(self.val_data)
        
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
            torch.tensor(self.train_data_preped['document'].tolist()),
            torch.tensor(self.train_data_preped['label'].tolist())
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
            torch.tensor(self.val_data_preped['document'].tolist()),
            torch.tensor(self.val_data_preped['label'].tolist())
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

    def save_confusion_matrix(self, y_true, y_pred, normalize=True):
        """혼동 행렬 생성 및 저장"""
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues')
        plt.title('Normalized Confusion Matrix' if normalize else 'Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        confusion_matrix_dir = Path("evaluation")
        confusion_matrix_dir.mkdir(parents=True, exist_ok=True)
        confusion_matrix_path = confusion_matrix_dir / f"confusion_matrix{'_normalized' if normalize else ''}.png"
        
        plt.savefig(confusion_matrix_path)
        plt.close()
        
        return str(confusion_matrix_path)

    def on_validation_epoch_end(self):
        val_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        y_true = []
        y_pred = []

        for out in self.validation_step_outputs:
            y_true.extend(out['y_true'])
            y_pred.extend(out['y_pred'])

        metrics = {
            'val_loss': val_loss.item(),
            'val_acc': accuracy_score(y_true, y_pred),
            'val_precision': precision_score(y_true, y_pred),
            'val_recall': recall_score(y_true, y_pred),
            'val_f1': f1_score(y_true, y_pred)
        }
        
        # 혼동 행렬 생성 및 저장 (일반 + 정규화)
        confusion_matrix_path = self.save_confusion_matrix(y_true, y_pred, normalize=False)
        normalized_cm_path = self.save_confusion_matrix(y_true, y_pred, normalize=True)
        
        # MLflow에 메트릭과 데이터셋 정보 로깅
        self.mlflow_logger.log_evaluate(
            metrics=metrics,
            model=self.bert,
            tokenizer=self.tokenizer,
            train_data=self.train_data,
            val_data=self.val_data,
            model_name='KcBERT',
            dataset_name=config.data['name'],
            sampling_rate=config.data['sampling_rate'],
            confusion_matrix_path=[confusion_matrix_path, normalized_cm_path]
        )
        
        # 모델 레지스트리에 등록 시도
        registry = ModelRegistry()
        registry.add_model(
            model_name='KcBERT',
            run_id=self.run_id,
            metrics=metrics,
            dataset_name=config.data['name'],
            sampling_rate=config.data['sampling_rate'],
            threshold=config.model['register_threshold']
        )
        
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
            mlflow.pytorch.log_model(
                self.bert,
                "model",
                registered_model_name='KcBERT'
            )
            # 토크나이저 저장
            mlflow.log_artifacts(str(save_dir), "tokenizer")
            logger.info(f"모델과 토크나이저가 MLflow에 저장되었습니다.")

def main():
    # MLflow 설정
    file_path = Path(__file__)
    file_name = file_path.stem
    config.mlflow['experiment_name'] = config.mlflow['experiment_name'] + '_' + file_name
    print(f'Experiment name: {config.mlflow["experiment_name"]}')
    
    print("Using PyTorch Ver", torch.__version__)
    print("Fix Seed:", args.random_seed)
    seed_everything(args.random_seed)
    model = KcBERTClassifier(args)
    
    # MLflow 로거 생성
    mlflow_logger = MLflowLogger()
    
    # run_name 생성 - 단순화
    run_name = f"KcBERT_train_{datetime.now().strftime('%Y%m%d')}"
    
    # MLflow 실행 시작
    with mlflow_logger.run_with_logging(
        "training",
        "KcBERT",  # 단순화된 모델 이름
        config.data['name'],
        config.data['sampling_rate'],
        run_name=run_name
    ) as run:
        model.run_id = run.info.run_id
        
        # 하이퍼파라미터 로깅
        mlflow_logger.log_params({
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