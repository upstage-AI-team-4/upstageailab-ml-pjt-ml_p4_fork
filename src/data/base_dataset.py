# src/data/base.py
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import pytorch_lightning as pl
from typing import Dict, Optional, Any
import torch


class BaseDataset(Dataset, ABC):
    @abstractmethod
    def load_data(self) -> None:
        """데이터 로드 로직"""
        pass
    
    @abstractmethod
    def preprocess(self, item: Any) -> Dict[str, torch.Tensor]:
        """개별 아이템 전처리 로직"""
        pass
    @abstractmethod
    def __len__(self) -> int:
        pass
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pass

class BaseDataModule(pl.LightningDataModule, ABC):
    @abstractmethod
    def prepare_data(self) -> None:
        """데이터 다운로드 등 전체 데이터 준비"""
        pass
    
    @abstractmethod
    def setup(self, stage: Optional[str] = None) -> None:
        """train/val/test 데이터셋 설정"""
        pass