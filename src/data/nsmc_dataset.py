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