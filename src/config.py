import os
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch

@dataclass
class MLflowConfig:
    """MLflow 관련 설정"""
    tracking_uri: str
    experiment_name: str
    model_registry_metric_threshold: float
    mlrun_path: Path
    backend_store_uri: Path
    model_info_path: Path
    artifact_location: Path
    server_config: Dict[str, Any]

class Config:
    def __init__(self, config_path="config/config.yaml"):
        """설정 초기화"""
        # 프로젝트 루트 디렉토리 찾기
        self.project_root = self._find_project_root()

        #self.base_path = self.project_root.parent
        self.config_path = self.project_root / config_path
        self.config = self._load_config()
        
        # 프로젝트 설정 먼저 로드 (다른 경로에서 사용됨)
        self.project = self.config["project"]
        
        # 기본 경로 설정 (프로젝트 구조에 맞게 체계적으로)
        self.paths = {
            # 데이터 관련 경로
            'data': self.project_root / 'data',
            'raw_data': self.project_root / 'data' / 'raw' / self.project['dataset_name'],
            'processed_data': self.project_root / 'data' / 'processed' / self.project['dataset_name'],
            
            # 모델 관련 경로
            'models_base': self.project_root / 'models',
            'model': self.project_root / 'models' / self.project['model_name'],
            
            # 로그 관련 경로
            'logs_base': self.project_root / 'logs',
            'train_logs': self.project_root / 'logs' / 'training',
            'tensorboard': self.project_root / 'logs' / 'tensorboard',
            
            # 체크포인트 관련 경로
            'checkpoints_base': self.project_root / 'checkpoints',
            'model_checkpoints': self.project_root / 'checkpoints' / self.project['model_name'],
        }
        self.models = self.config["models"]
        
        # MLflow 관련 설정
        self.mlflow = MLflowConfig(
            tracking_uri=self.config["mlflow"]["tracking_uri"],
            experiment_name=self.config["mlflow"]["experiment_name"],
            model_registry_metric_threshold=self.config["mlflow"]["model_registry_metric_threshold"],
            mlrun_path=self.project_root / self.config["mlflow"]["mlrun_path"],
            backend_store_uri=self.project_root / self.config["mlflow"]["backend_store_uri"],
            model_info_path=self.project_root / self.config["mlflow"]["model_info_path"],
            artifact_location=self.project_root / self.config["mlflow"]["artifact_location"],
            server_config=self.config["mlflow"]["server_config"]
        )
        
        # 데이터셋 설정
        self.data = self.config["dataset"][self.project["dataset_name"]]
        
        # 모델 설정
        self.model_config = self.config["models"][self.project["model_name"]]
        self.training_config = self.model_config["training"]
        
        # 체크포인트 설정 (경로 업데이트)
        self.checkpoint = self.config["common"]["checkpoint"]
        self.checkpoint["dirpath"] = str(self.paths["model_checkpoints"])
        
        # 공통 설정 (로거 경로 업데이트)
        self.common = self.config["common"]
        self.common["trainer"]["default_root_dir"] = str(self.paths["train_logs"])
        self.common["trainer"]["logger"]["save_dir"] = str(self.paths["tensorboard"])
        
        # HPO 설정
        self.hpo = self.config["hpo"]
        
        # 필요한 디렉토리 생성
        self._create_directories()
    
    def _find_project_root(self) -> Path:
        """프로젝트 루트 디렉토리 찾기
        
        'src' 디렉토리나 'config/config.yaml' 파일이 있는 위치를 프로젝트 루트로 간주
        """
        current_dir = Path(__file__).resolve().parent
        
        # 상위 디렉토리로 이동하면서 프로젝트 루트 찾기
        while current_dir.name:
            if (current_dir / 'src').exists() or (current_dir / 'config' / 'config.yaml').exists():
                return current_dir
            current_dir = current_dir.parent
            
        # 프로젝트 루트를 찾지 못한 경우
        raise RuntimeError(
            "Project root directory not found. "
            "Please make sure you have 'src' directory or 'config/config.yaml' in your project root."
        )
    
    def _create_directories(self):
        """필요한 디렉토리 생성"""
        # 모든 기본 경로 생성
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
            
        # MLflow 경로 생성
        self.mlflow.mlrun_path.mkdir(parents=True, exist_ok=True)
        self.mlflow.backend_store_uri.mkdir(parents=True, exist_ok=True)
        self.mlflow.artifact_location.mkdir(parents=True, exist_ok=True)
        Path(self.mlflow.model_info_path).parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nDebug: Directories initialized:")
        print(f"\nDebug: Project root: {self.project_root}")
        print(f"\nDebug: Data paths:")
        print(f"  - Raw data: {self.paths['raw_data']}")
        print(f"  - Processed data: {self.paths['processed_data']}")
        print(f"\nDebug: Model paths:")
        print(f"  - Models base: {self.paths['models_base']}")
        print(f"  - Current model: {self.paths['model']}")
        print(f"\nDebug: Log paths:")
        print(f"  - Training logs: {self.paths['train_logs']}")
        print(f"  - Tensorboard logs: {self.paths['tensorboard']}")
        print(f"\nDebug: Checkpoint paths:")
        print(f"  - Model checkpoints: {self.paths['model_checkpoints']}")
        print(f"\nDebug: MLflow paths:")
        print(f"  - MLrun path: {self.mlflow.mlrun_path}")
        print(f"  - Backend store: {self.mlflow.backend_store_uri}")
        print(f"  - Artifact location: {self.mlflow.artifact_location}")
        print(f"  - Model info: {self.mlflow.model_info_path}")

    def _load_config(self) -> Dict[str, Any]:
        """YAML 설정 파일 로드"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get_model_kwargs(self) -> Dict[str, Any]:
        """모델 초기화를 위한 인자 반환"""
        return {
            "pretrained_model": self.model_config["pretrained_model"],
            "num_labels": self.training_config["num_labels"],
            "learning_rate": self.training_config["lr"],
            "num_train_epochs": self.training_config["epochs"],
            "per_device_train_batch_size": self.training_config["batch_size"],
            "per_device_eval_batch_size": self.training_config["batch_size"],
            "max_length": self.training_config["max_length"],
            "report_cycle": self.training_config["report_cycle"],
            "optimizer": self.training_config["optimizer"],
            "lr_scheduler": self.training_config["lr_scheduler"],
            "precision": self.training_config["precision"],
            "num_unfreeze_layers": self.training_config["num_unfreeze_layers"],
            "accumulate_grad_batches": self.training_config["accumulate_grad_batches"]
        }
    
    def get_data_paths(self) -> Dict[str, Path]:
        """데이터 파일 경로 반환"""
        return {
            "train": self.project_root / self.paths["raw_data"] / self.data["train_data_path"],
            "val": self.project_root / self.paths["raw_data"] / self.data["val_data_path"]
        }
    
    def get_column_mapping(self) -> Dict[str, str]:
        """데이터셋 컬럼 매핑 반환"""
        return self.data["column_mapping"]
    
    def get_sampling_config(self) -> Dict[str, float]:
        """샘플링 설정 반환"""
        return {
            "sampling_rate": self.data["sampling_rate"],
            "test_size": self.data["test_size"]
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
    
    def get_hpo_config(self) -> Dict[str, Any]:
        """HPO 설정 반환"""
        return self.hpo