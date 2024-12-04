import os
from pathlib import Path
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, Any, Tuple
from transformers import PreTrainedTokenizerBase
import torch
import requests
from src.config import Config
from .text_utils import clean_text

def download_nsmc(config):
    """Download NSMC dataset"""
    base_url = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_{}.txt"
    raw_dir = Path(config.raw_data_path)
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
        data: Tuple[np.ndarray, np.ndarray],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int
    ):
        """
        NSMC 데이터셋
        
        Args:
            data: (texts, labels) 튜플 - document 컬럼을 text로 매핑
            tokenizer: 토크나이저
            max_length: 최대 시퀀스 길이
        """
        self.texts, self.labels = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # 텍스트 전처리
        text = clean_text(text)
        
        # 토크나이징
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 배치 차원 제거
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label)
        }

class NSMCDataModule(LightningDataModule):
    def __init__(
        self,
        config: Config,
        tokenizer: PreTrainedTokenizerBase,
        **kwargs
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        
        # 데이터셋 설정
        self.batch_size = self.config.training_config['batch_size']
        self.max_length = self.config.training_config['max_length']
        self.num_workers = kwargs.get('num_workers', 4)
        
        # 데이터 경로
        self.train_path = self.config.paths['raw_data'] / self.config.data['train_data_path']
        self.val_path = self.config.paths['raw_data'] / self.config.data['val_data_path']
        
        # 데이터셋
        self.train_dataset = None
        self.val_dataset = None
        
    def prepare_data(self):
        """데이터 준비"""
        # 데이터 디렉토리 생성
        self.config.paths['raw_data'].mkdir(parents=True, exist_ok=True)
        self.config.paths['processed_data'].mkdir(parents=True, exist_ok=True)
        
        # 학습 데이터 다운로드
        if not self.train_path.exists():
            print("Downloading training data...")
            download_dataset(self.config.data['train_data_path'], self.config.paths['raw_data'])
            
        # 검증 데이터 다운로드
        if not self.val_path.exists():
            print("Downloading validation data...")
            download_dataset(self.config.data['val_data_path'], self.config.paths['raw_data'])
    
    def setup(self, stage: Optional[str] = None):
        """데이터셋 설정"""
        if stage == 'fit' or stage is None:
            # 학습 데이터셋 로드
            train_data = load_dataset(str(self.train_path), self.config.data['column_mapping'])
            if self.config.data['sampling_rate'] < 1.0:
                train_data = sample_dataset(train_data, self.config.data['sampling_rate'])
            self.train_dataset = NSMCDataset(train_data, self.tokenizer, self.max_length)
            
            # 검증 데이터셋 로드
            val_data = load_dataset(str(self.val_path), self.config.data['column_mapping'])
            if self.config.data['sampling_rate'] < 1.0:
                val_data = sample_dataset(val_data, self.config.data['sampling_rate'])
            self.val_dataset = NSMCDataset(val_data, self.tokenizer, self.max_length)
            
            print(f"Train dataset size: {len(self.train_dataset)}")
            print(f"Validation dataset size: {len(self.val_dataset)}")
    
    def train_dataloader(self):
        """학습 데이터로더 반환"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        """검증 데이터로더 반환"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

def log_data_info(data_module: NSMCDataModule):
    """데이터셋 정보 출력"""
    print("\n=== Dataset Information ===")
    
    # 학습 데이터 레이블 분포
    train_labels = [sample['labels'].item() for sample in data_module.train_dataset]
    print("\nTrain Label Distribution:")
    print(pd.Series(train_labels).value_counts())
    
    # 검증 데이터 레이블 분포
    val_labels = [sample['labels'].item() for sample in data_module.val_dataset]
    print("\nValidation Label Distribution:")
    print(pd.Series(val_labels).value_counts())

def download_dataset(filename: str, save_path: Path):
    """NSMC 데이터셋 다운로드"""
    base_url = "https://raw.githubusercontent.com/e9t/nsmc/master"
    
    # 저장 경로 생성
    save_path.mkdir(parents=True, exist_ok=True)
    file_path = save_path / filename
    
    # 파일 다운로드
    url = f"{base_url}/{filename}"
    response = requests.get(url)
    response.raise_for_status()
    
    # 파일 저장
    with open(file_path, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded {filename} to {file_path}")

def load_dataset(file_path: str, column_mapping: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    """데이터셋 로드 및 전처리"""
    df = pd.read_csv(file_path, sep='\t')
    
    # 컬럼 이름 매핑
    text_col = column_mapping['text']
    label_col = column_mapping['label']
    
    return df[text_col].values, df[label_col].values

def sample_dataset(data: Tuple[np.ndarray, np.ndarray], sampling_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """데이터셋 샘플링"""
    if sampling_rate >= 1.0:
        return data
    
    texts, labels = data
    n_samples = int(len(texts) * sampling_rate)
    indices = np.random.choice(len(texts), n_samples, replace=False)
    
    return texts[indices], labels[indices]