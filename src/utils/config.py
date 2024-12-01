from pathlib import Path
import yaml
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union
import os
from pprint import pprint
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import mlflow

@dataclass
class Config:
    def __init__(self, config_path: str = "config/config.yaml"):
        """설정 파일 로드 및 초기화"""
        # 기본 경로 설정
        self.base_path = Path.cwd()
        self.config_path = self.base_path / config_path
        
        # 설정 파일 로드
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 기본 경로 구조 설정
        self.paths = {
            'base': self.base_path,
            'logs': self.base_path / 'logs',
            'mlruns': self.base_path / 'mlruns',
            'models': self.base_path / 'models',
            'checkpoints': self.base_path / 'checkpoints',
            'data': self.base_path / 'data',
            'raw_data': self.base_path / 'data/raw',
            'processed_data': self.base_path / 'data/processed'
        }
        
        # 디렉토리 생성
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
        
        # MLflow 설정에 base_path 적용
        self.mlflow = self.config['mlflow']
        self.mlflow['mlrun_path'] = self.base_path / self.mlflow['mlrun_path']
        self.mlflow['model_info_path'] = self.base_path / self.mlflow['model_info_path']
        
        # 공통 설정
        self.common = self.config['common']
        self.common['trainer']['logger']['save_dir'] = str(self.paths['logs'])
        self.common['trainer']['default_root_dir'] = str(self.paths['logs'])
        
        # 체크포인트 설정에 base_path 적용
        self.checkpoint = self.common['checkpoint']
        self.checkpoint['dirpath'] = self.base_path / self.checkpoint['dirpath']
        
        # 프로젝트 설정
        self.project = self.config['project']
        
        # 데이터 설정
        self.data = self.config['dataset'][self.project['dataset_name']]
        
        # 모이터셋별 경로 설정
        dataset_name = self.data['dataset_name']
        self.paths['dataset'] = self.paths['raw_data'] / dataset_name
        self.paths['processed_dataset'] = self.paths['processed_data'] / dataset_name
        
        # 데이터 파일 경로 설정
        self.data['train_data_path'] = str(self.paths['dataset'] / self.data['train_data_path'])
        self.data['val_data_path'] = str(self.paths['dataset'] / self.data['val_data_path'])
        
        # 모델 설정
        self.model_config = self.config['models'][self.project['model_name']]
        self.training_config = self.model_config['training']
    
    def get_model_kwargs(self):
        """모델 초기화에 필요한 인자들을 반환"""
        return {
            'pretrained_model': self.model_config['pretrained_model'],
            'num_labels': self.training_config['num_labels'],
            'learning_rate': self.training_config['lr'],
            'num_unfreeze_layers': self.training_config['num_unfreeze_layers']
        }
    
    def get_trainer_kwargs(self) -> Dict[str, Any]:
        """Trainer 초기화에 필요한 인자들을 반환"""
        trainer_config = self.common['trainer'].copy()
        
        # Logger 설정
        logger_config = trainer_config.pop('logger', {})
        logger = TensorBoardLogger(
            save_dir=logger_config['save_dir'],
            name=logger_config['name'],
            version=logger_config['version']
        )
        
        # CUDA 사용 가능 여부 확인
        if trainer_config['accelerator'] == 'gpu' and not torch.cuda.is_available():
            print("Warning: GPU requested but CUDA is not available. Falling back to CPU.")
            trainer_config['accelerator'] = 'cpu'
        
        return {
            "max_epochs": self.training_config['epochs'],
            "precision": self.training_config['precision'],
            **trainer_config,
            "logger": logger,
            "callbacks": [
                ModelCheckpoint(**self.checkpoint)
            ]
        }

if __name__ == "__main__":
    config = Config()

