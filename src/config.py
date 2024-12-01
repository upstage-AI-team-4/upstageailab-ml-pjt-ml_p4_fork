import os
import yaml
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any

class Config:
    def __init__(self, config_path="config/config.yaml"):
        """설정 초기화"""
        self.config_path = config_path
        self.config = self._load_config()
        self.base_path = Path.cwd()
        
        # 프로젝트 설정 먼저 로드 (다른 경로에서 사용됨)
        self.project = self.config["project"]
        
        # 기본 경로 설정 (프로젝트 구조에 맞게 체계적으로)
        self.paths = {
            # 데이터 관련 경로
            'data': self.base_path / 'data',
            'raw_data': self.base_path / 'data' / 'raw' / self.project['dataset_name'],
            'processed_data': self.base_path / 'data' / 'processed' / self.project['dataset_name'],
            
            # 모델 관련 경로
            'models_base': self.base_path / 'models',
            'model': self.base_path / 'models' / self.project['model_name'],
            
            # 로그 관련 경로
            'logs_base': self.base_path / 'logs',
            'train_logs': self.base_path / 'logs' / 'training',
            'tensorboard': self.base_path / 'logs' / 'tensorboard',
            
            # 체크포인트 관련 경로
            'checkpoints_base': self.base_path / 'checkpoints',
            'model_checkpoints': self.base_path / 'checkpoints' / self.project['model_name'],
        }
        
        # MLflow 관련 경로 설정
        self.mlflow = SimpleNamespace(
            tracking_uri=self.config["mlflow"]["tracking_uri"],
            experiment_name=self.config["mlflow"]["experiment_name"],
            model_registry_metric_threshold=self.config["mlflow"]["model_registry_metric_threshold"],
            mlrun_path=self.base_path / self.config["mlflow"]["mlrun_path"],
            backend_store_uri=self.base_path / self.config["mlflow"]["backend_store_uri"],
            model_info_path=self.base_path / self.config["mlflow"]["model_info_path"],
            artifact_location=self.base_path / self.config["mlflow"]["artifact_location"],
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
        
    def _create_directories(self):
        """필요한 디렉토리 생성"""
        # 모든 기본 경로 생성
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
            
        # MLflow 경로 생성
        self.mlflow.mlrun_path.mkdir(parents=True, exist_ok=True)
        self.mlflow.backend_store_uri.mkdir(parents=True, exist_ok=True)
        self.mlflow.artifact_location.mkdir(parents=True, exist_ok=True)
        
        print(f"\nDebug: Directories initialized:")
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
            "train": self.paths["raw_data"] / self.data["train_data_path"],
            "val": self.paths["raw_data"] / self.data["val_data_path"]
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
    
    def get_trainer_config(self) -> Dict[str, Any]:
        """Trainer 설정 반환"""
        return self.common["trainer"]
    
    def get_hpo_config(self) -> Dict[str, Any]:
        """HPO 설정 반환"""
        return self.hpo