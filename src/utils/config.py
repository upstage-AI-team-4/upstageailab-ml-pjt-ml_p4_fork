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
class MLflowConfig:
    """MLflow 관련 설정"""
    tracking_uri: str
    experiment_name: str
    model_registry_metric_threshold: float
    mlrun_path: str
    backend_store_uri: str
    model_info_path: str
    artifact_location: str
    server_config: Dict[str, Any]

@dataclass
class Config:
    def __init__(self, config_path: str = "config/config.yaml"):
        """설정 초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        self.base_path = Path(os.getcwd())
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
            
        # MLflow 설정
        self.mlflow = MLflowConfig(
            tracking_uri=self._config['mlflow']['tracking_uri'],
            experiment_name=self._config['mlflow']['experiment_name'],
            model_registry_metric_threshold=self._config['mlflow']['model_registry_metric_threshold'],
            mlrun_path=self._config['mlflow']['mlrun_path'],
            backend_store_uri=self._config['mlflow']['backend_store_uri'],
            model_info_path=str(self.base_path / self._config['mlflow']['model_info_path']),
            artifact_location=self._config['mlflow']['artifact_location'],
            server_config=self._config['mlflow']['server_config']
        )
        
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
        
        # 공통 설정
        self.common = self._config['common']
        self.common['trainer']['logger']['save_dir'] = str(self.paths['logs'])
        self.common['trainer']['default_root_dir'] = str(self.paths['logs'])
        
        # 체크포인트 설정에 base_path 적용
        self.checkpoint = self.common['checkpoint']
        self.checkpoint['dirpath'] = self.base_path / self.checkpoint['dirpath']
        
        # 프로젝트 설정
        self.project = self._config['project']
        
        # 데이터 설정
        self.data = self._config['dataset'][self.project['dataset_name']]
        
        # 모이터셋별 경로 설정
        dataset_name = self.data['dataset_name']
        self.paths['dataset'] = self.paths['raw_data'] / dataset_name
        self.paths['processed_dataset'] = self.paths['processed_data'] / dataset_name
        
        # 데이터 파일 경로 설정
        self.data['train_data_path'] = str(self.paths['dataset'] / self.data['train_data_path'])
        self.data['val_data_path'] = str(self.paths['dataset'] / self.data['val_data_path'])
        
        # 모델 설정
        self.model_config = self._config['models'][self.project['model_name']]
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

