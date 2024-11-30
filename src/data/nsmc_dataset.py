import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, Any
from .text_utils import clean_text
import requests
from transformers import PreTrainedTokenizer
from pytorch_lightning import LightningDataModule
from pathlib import Path
import numpy as np
import mlflow

def download_nsmc(data_dir: str = 'data/nsmc'):
    """Download NSMC dataset"""
    base_url = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_{}.txt"
    data_dir = Path(data_dir)
    raw_dir = data_dir / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'test']:
        output_file = raw_dir / f"ratings_{split}.txt"
        if not output_file.exists():
            print(f"Downloading {split} dataset...")
            response = requests.get(base_url.format(split))
            output_file.write_bytes(response.content)
            print(f"Downloaded {split} dataset to {output_file}")

def sample_data(path: str, n_samples: int = 10000, random_state: int = 42):
    """Sample n rows from dataset"""
    df = pd.read_csv(path, sep='\t')
    return df.sample(n=n_samples, random_state=random_state)

class NSMCDataset(Dataset):
    def __init__(
        self,
        documents: np.ndarray,
        labels: np.ndarray,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512
    ):
        """
        NSMC 데이터셋
        
        Args:
            documents: 문서 텍스트 배열
            labels: 레이블 배열
            tokenizer: 토크나이저
            max_length: 최대 시퀀스 길이
        """
        super().__init__()
        self.documents = documents
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = clean_text(str(self.documents[idx]))
        encoding = self.tokenizer(
            text,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# src/data/datamodules.py
class NSMCDataModule(LightningDataModule):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        max_length: int,
        sampling_rate: float,
        config: Any,
        data_dir: Optional[str] = "data",
        train_file: Optional[str] = "ratings_train.txt",
        val_file: Optional[str] = "ratings_test.txt",
    ):
        """
        NSMC 데이터셋을 위한 DataModule
        
        Args:
            tokenizer: 토크나이저
            batch_size: 배치 크기
            max_length: 최대 시퀀스 길이
            sampling_rate: 데이터 샘플링 비율
            config: 설정 객체
            data_dir: 데이터 디렉토리 경로
            train_file: 학습 데이터 파일명
            val_file: 검증 데이터 파일명
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.sampling_rate = sampling_rate
        self.config = config
        
        self.data_dir = Path(data_dir)
        self.train_file = self.data_dir / "raw" / train_file
        self.val_file = self.data_dir / "raw" / val_file
        
        self.train_dataset = None
        self.val_dataset = None
    
    def prepare_data(self):
        """데이터 준비"""
        # 데이터 디렉토리 생성
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        
        # 데이터 파일이 없으면 다운로드
        if not self.train_file.exists() or not self.val_file.exists():
            print("No dataset files found. Downloading NSMC dataset...")
            download_nsmc(str(self.data_dir))

    def setup(self, stage: Optional[str] = None):
        """데이터셋 설정"""
        if stage == "fit" or stage is None:
            # 파일 존재 확인
            if not self.train_file.exists() or not self.val_file.exists():
                raise FileNotFoundError(
                    f"Dataset files not found. Please check if files exist at:\n"
                    f"Train: {self.train_file}\n"
                    f"Val: {self.val_file}"
                )
            
            # 학습 데이터 로드
            train_df = pd.read_csv(self.train_file, sep='\t')
            if self.sampling_rate < 1.0:
                train_df = train_df.sample(frac=self.sampling_rate, random_state=42)
            
            # 컬럼 이름 매핑
            train_df = train_df.rename(columns={
                self.config.data.column_mapping['text']: 'text',
                self.config.data.column_mapping['label']: 'label'
            })
            
            # 데이터셋 생성
            self.train_dataset = NSMCDataset(
                documents=train_df['text'].values,
                labels=train_df['label'].values,
                tokenizer=self.tokenizer,
                max_length=self.max_length
            )
            
            # 검증 데이터도 동일하게 처리
            val_df = pd.read_csv(self.val_file, sep='\t')
            if self.sampling_rate < 1.0:
                val_df = val_df.sample(frac=self.sampling_rate, random_state=42)
            
            val_df = val_df.rename(columns={
                self.config.data.column_mapping['text']: 'text',
                self.config.data.column_mapping['label']: 'label'
            })
            
            self.val_dataset = NSMCDataset(
                documents=val_df['text'].values,
                labels=val_df['label'].values,
                tokenizer=self.tokenizer,
                max_length=self.max_length
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )

def log_data_info(data_module, config):
    """Log dataset information to MLflow and print dataset details."""
    train_path = config.data.train_data_path
    val_path = config.data.val_data_path
    print("=" * 50)
    print(f"Train Path: {train_path}")
    print(f"Val Path: {val_path}")
    # Load raw data
    train_df = pd.read_csv(train_path, sep='\t')
    val_df = pd.read_csv(val_path, sep='\t')
    
    # Print raw data examples and size
    print("\n=== Raw Data Examples ===")
    print(train_df.head())
    print(f"\nTrain Data Size: {len(train_df)}")
    print(f"Validation Data Size: {len(val_df)}")
    
    # Print label distribution in raw data
    print("\n=== Raw Data Label Distribution ===")
    print("Train Labels:")
    print(train_df['label'].value_counts())
    print("Validation Labels:")
    print(val_df['label'].value_counts())
    
    # Log meta data
    for name, df in [("train", train_df), ("val", val_df)]:
        mlflow.log_param(f"{name}_num_rows", len(df))
        mlflow.log_param(f"{name}_num_columns", len(df.columns))
        mlflow.log_param(f"{name}_columns", df.columns.tolist())
        
        # Log statistics
        stats = df.describe(include='all').to_dict()
        mlflow.log_dict(stats, f"{name}_data_statistics.json")
        
        # Log NaNs and zeros
        nans = df.isna().sum().to_dict()
        zeros = (df == 0).sum().to_dict()
        mlflow.log_dict(nans, f"{name}_nans.json")
        mlflow.log_dict(zeros, f"{name}_zeros.json")
    
    # Log tokenized data
    for name, dataset in [("train", data_module.train_dataset), ("val", data_module.val_dataset)]:
        tokenized_samples = [
            {
                'input_ids': sample['input_ids'].tolist(),
                'attention_mask': sample['attention_mask'].tolist(),
                'labels': sample['labels'].item()
            }
            for sample in [dataset[i] for i in range(min(5, len(dataset)))]
        ]
        mlflow.log_dict(tokenized_samples, f"{name}_tokenized_samples.json")
    
    # Print actual dataset sizes
    print(f"\nActual Train Dataset Size: {len(data_module.train_dataset)}")
    print(f"Actual Validation Dataset Size: {len(data_module.val_dataset)}")
    
    # Print label distribution in processed data
    print("\n=== Processed Data Label Distribution ===")
    print("Train Labels:")
    train_labels = [sample['labels'].item() for sample in data_module.train_dataset]
    print(pd.Series(train_labels).value_counts())
    print("Validation Labels:")
    val_labels = [sample['labels'].item() for sample in data_module.val_dataset]
    print(pd.Series(val_labels).value_counts())