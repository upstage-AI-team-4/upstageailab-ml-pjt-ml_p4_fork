from pathlib import Path
import yaml
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union
import os
from pprint import pprint
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

@dataclass
class ProjectConfig:
    random_state: int
    dataset_name: str
    model_name: str

@dataclass
class DataConfig:
    sampling_rate: float
    test_size: float
    train_data_path: str
    val_data_path: str
    dataset_name: str
    column_mapping: Dict[str, str]

@dataclass
class ModelConfig:
    model_name: str
    pretrained_model: str
    model_dir: Path

@dataclass
class TrainerConfig:
    accelerator: str
    devices: int
    deterministic: bool
    gradient_clip_val: float
    accumulate_grad_batches: int
    log_every_n_steps: int
    num_sanity_val_steps: int

@dataclass
class CheckpointConfig:
    dirpath: str
    filename: str
    monitor: str
    mode: str
    save_top_k: int
    save_last: bool

@dataclass
class HPOParamConfig:
    type: str
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    log: Optional[bool] = False

@dataclass
class HPOConfig:
    n_trials: int
    sampler: str
    pruner: str
    direction: str
    metric: str
    params: Dict[str, Dict[str, Any]]

@dataclass
class BaseTrainingConfig:
    num_labels: int
    random_seed: int
    pretrained_model: str
    batch_size: int
    lr: float
    epochs: int
    max_length: int
    report_cycle: int
    optimizer: str
    lr_scheduler: str
    precision: int
    num_unfreeze_layers: int
    trainer: Dict[str, Any]
    checkpoint: Dict[str, Any]
    hpo: Dict[str, Any]

@dataclass
class MLflowConfig:
    tracking_uri: str
    experiment_name: str
    model_registry_metric_threshold: float
    mlrun_path: Union[str, Path]
    backend_store_uri: str
    model_info_path: str

class Config:
    def __init__(self):
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        self._config = self._load_config(config_path)
        
        # Project 설정
        self.project = ProjectConfig(**self._config['project'])
        
        # 데이터 설정
        self.data = DataConfig(**self._config['dataset'][self.project.dataset_name])
        self.data_config = self.data  # alias 추가
        
        # 모델 설정
        self.model = ModelConfig(**self._config['models'][self.project.model_name])
        
        # 학습 설정
        self.base_training = BaseTrainingConfig(**self._config['trainings']['base'])
        
        # 경로 설정
        self.base_path = Path(__file__).parent.parent.parent
        self.data_path = self.base_path / 'data'
        self.raw_data_path = self.data_path / 'raw'
        self.processed_data_path = self.data_path / 'processed'

        self.data.train_data_path = str(self.raw_data_path / self.data.train_data_path)
        self.data.val_data_path = str(self.raw_data_path / self.data.val_data_path)

        # MLflow 설정
        self.mlflow = MLflowConfig(**self._config['mlflow'])
        
        # MLflow 경로 설정
        mlruns_path = self.base_path / self.mlflow.mlrun_path
        
        # tracking_uri와 backend_store_uri를 mlruns 경로로 설정
        self.mlflow.tracking_uri = f"file://{str(mlruns_path)}"
        self.mlflow.backend_store_uri = str(mlruns_path)
        self.mlflow.mlrun_path = mlruns_path
        self.mlflow.model_info_path = self.base_path / 'config' / self.mlflow.model_info_path
        # MLflow 디렉토리 생성
        os.makedirs(self.mlflow.mlrun_path, exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load config from yaml file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def get_trainer_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for Trainer initialization"""
        trainer_config = self.base_training.trainer.copy()  # 복사본 생성
        checkpoint_config = self.base_training.checkpoint
        
        # CUDA 사용 가능 여부 확인
        if trainer_config['accelerator'] == 'gpu' and not torch.cuda.is_available():
            print("Warning: GPU requested but CUDA is not available. Falling back to CPU.")
            trainer_config['accelerator'] = 'cpu'
        
        # Lightning 로그 경로를 절대 경로로 변환
        if 'default_root_dir' in trainer_config:
            trainer_config['default_root_dir'] = str(self.base_path / trainer_config['default_root_dir'])
        
        return {
            "max_epochs": self.base_training.epochs,
            "precision": self.base_training.precision,
            **trainer_config,
            "callbacks": [
                ModelCheckpoint(**checkpoint_config)
            ]
        }

    def get_model_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for model initialization"""
        model_dir = self.base_path / self.model.model_dir
        return {
            "pretrained_model": self.base_training.pretrained_model,
            "model_dir": model_dir,
            "model_name": self.model.model_name,
            "num_labels": self.base_training.num_labels,
            "learning_rate": self.base_training.lr,
            "optimizer": self.base_training.optimizer,
            "lr_scheduler": self.base_training.lr_scheduler,
            "num_unfreeze_layers": self.base_training.num_unfreeze_layers
        }

    def get_data_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for data module initialization"""
        return {
            "train_data_path": self.data.train_data_path,
            "val_data_path": self.data.val_data_path,
            "sampling_rate": self.data.sampling_rate,
            "test_size": self.data.test_size
        }

if __name__ == "__main__":
    config = Config()

