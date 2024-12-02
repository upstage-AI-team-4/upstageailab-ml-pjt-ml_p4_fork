import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import pandas as pd
import numpy as np
import torch
import mlflow
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from src.config import Config
import yaml

class ModelStage(Enum):
    """모델 스테이지 정의"""
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"

def cleanup_old_runs(config, days_to_keep=7):
    """오래된 MLflow 실행 정리"""
    try:
        client = MlflowClient()
        experiment = mlflow.get_experiment_by_name(config.mlflow.experiment_name)
        
        if experiment is None:
            print(f"No experiment found with name: {config.mlflow.experiment_name}")
            return
            
        # 실험의 모든 실행 가져오기
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )
        
        # 기준 시간 계산
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # 오래된 실행 삭제
        for run in runs:
            run_date = datetime.fromtimestamp(run.info.start_time / 1000.0)
            if run_date < cutoff_date:
                client.delete_run(run.info.run_id)
                print(f"Deleted run: {run.info.run_id} from {run_date}")
                
    except Exception as e:
        print(f"Error cleaning up old runs: {str(e)}")

def cleanup_artifacts(config, metrics: Dict[str, float], run_id: str):
    """MLflow 아티팩트 정리
    
    Args:
        config: 설정 객체
        metrics: 평가 지표
        run_id: MLflow 실행 ID
    """
    try:
        # 성능이 좋지 않은 실행의 아티팩트 삭제
        if metrics.get('val_f1', 0) < config.mlflow.model_registry_metric_threshold:
            print(f"\nRemoving artifacts for run {run_id} due to low performance...")
            artifact_path = Path(config.mlflow.artifact_location) / run_id
            if artifact_path.exists():
                shutil.rmtree(str(artifact_path))
                print(f"Removed artifacts at: {artifact_path}")
    except Exception as e:
        print(f"Error cleaning up artifacts: {str(e)}")

def setup_mlflow_server(config: Config):
    """MLflow 서버 설정
    
    Args:
        config: 설정 객체
    """
    # 서버 환경 변수 설정
    os.environ['MLFLOW_WORKERS'] = str(config.mlflow.server_config.get('workers', 4))
    os.environ['MLFLOW_HTTP_REQUEST_HEADER_SIZE'] = str(config.mlflow.server_config.get('request_header_size', 65536))
    os.environ['MLFLOW_HTTP_REQUEST_TIMEOUT'] = '1800'  # 30분
    os.environ['MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE'] = '5242880'  # 5MB
    
    print(f"Debug: Setting up MLflow server with tracking URI: {config.mlflow.tracking_uri}")
    print(f"Debug: Workers: {os.environ['MLFLOW_WORKERS']}")
    print(f"Debug: Request header size: {os.environ['MLFLOW_HTTP_REQUEST_HEADER_SIZE']}")
    
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)

def initialize_mlflow(config: Config) -> str:
    """MLflow 초기화 및 설정"""
    try:
        print("\nInitializing MLflow...")
        
        # MLflow 기본 디렉토리 설정
        mlruns_dir = Path(config.project_root) / "mlruns"
        
        # MLflow 디렉토리 초기화
        if mlruns_dir.exists():
            import shutil
            shutil.rmtree(mlruns_dir)
        mlruns_dir.mkdir(parents=True, exist_ok=True)
        
        # trash 디렉토리 생성
        trash_dir = mlruns_dir / ".trash"
        trash_dir.mkdir(parents=True, exist_ok=True)
        
        # tracking URI 설정
        tracking_uri = f"file://{str(mlruns_dir.absolute())}"
        mlflow.set_tracking_uri(tracking_uri)
        print(f"Debug: MLflow tracking URI: {tracking_uri}")
        
        # 기본 실험(ID: 0) 생성
        default_experiment_dir = mlruns_dir / "0"
        default_experiment_dir.mkdir(parents=True, exist_ok=True)
        default_meta_content = {
            "artifact_location": str(default_experiment_dir / "artifacts"),
            "experiment_id": "0",
            "lifecycle_stage": "active",
            "name": "Default",
            "tags": {},
            "creation_time": int(datetime.now().timestamp() * 1000)
        }
        with open(default_experiment_dir / "meta.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(default_meta_content, f, default_flow_style=False)
        
        # 사용자 실험 생성
        experiment_id = mlflow.create_experiment(
            name=config.mlflow.experiment_name,
            artifact_location=str(mlruns_dir / config.mlflow.experiment_name)
        )
        print(f"Debug: Created new experiment with ID: {experiment_id}")
        
        # meta.yaml 파일 생성
        meta_dir = mlruns_dir / str(experiment_id)
        meta_dir.mkdir(parents=True, exist_ok=True)
        
        meta_content = {
            "artifact_location": str(mlruns_dir / config.mlflow.experiment_name),
            "experiment_id": experiment_id,
            "lifecycle_stage": "active",
            "name": config.mlflow.experiment_name,
            "tags": {},
            "creation_time": int(datetime.now().timestamp() * 1000)
        }
        
        meta_path = meta_dir / "meta.yaml"
        with open(meta_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(meta_content, f, default_flow_style=False)
        
        print(f"Debug: Created meta.yaml at: {meta_path}")
        
        # 실험 정보 확인
        experiment = mlflow.get_experiment(experiment_id)
        print(f"Debug: Experiment info:")
        print(f"  - Name: {experiment.name}")
        print(f"  - ID: {experiment.experiment_id}")
        print(f"  - Artifact Location: {experiment.artifact_location}")
        
        # 현재 실험 설정
        mlflow.set_experiment(experiment.name)
        print(f"Debug: Set active experiment to: {experiment.name}")
        
        return experiment_id
        
    except Exception as e:
        print(f"Error initializing MLflow: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

class MLflowModelManager:
    def __init__(self, config):
        self.config = config
        self.model_registry_path = Path(config.mlflow.model_info_path)
        
        # 모델 레지스트리 디렉토리가 없으면 생성
        self.model_registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        # MLflow 초기화
        self.experiment_id = initialize_mlflow(config)
        
        # MLflow 클라이언트 초기화
        mlruns_dir = str(config.project_root / "mlruns")
        self.tracking_uri = f"file://{mlruns_dir}"
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri)
        
        # 현재 실험 재설정 (안전을 위해)
        experiment = self.client.get_experiment(self.experiment_id)
        mlflow.set_experiment(experiment.name)
        
        print(f"Debug: MLflow tracking URI: {self.tracking_uri}")
        print(f"Debug: Model registry path: {self.model_registry_path}")
        print(f"Debug: Experiment ID: {self.experiment_id}")
        print(f"Debug: Active experiment: {experiment.name}")
    
    def log_training_params(self, run_id: str) -> None:
        """학습 파라미터를 MLflow에 기록"""
        try:
            print("\nLogging training parameters...")
            
            # MLflow 클라이언트를 통해 현재 로깅된 파라미터 확인
            client = mlflow.tracking.MlflowClient()
            current_run = client.get_run(run_id)
            logged_params = current_run.data.params
            
            # 학습 파라미터 로깅
            with mlflow.start_run(run_id=run_id):
                # 모델 설정
                mlflow.log_param("model_name", self.config.model.model_name)
                mlflow.log_param("num_labels", self.config.model.num_labels)
                
                # 데이터 설정
                mlflow.log_param("train_size", self.config.data.train_size)
                mlflow.log_param("test_size", self.config.data.test_size)
                mlflow.log_param("max_length", self.config.model.max_length)
                
                # 학습 설정
                mlflow.log_param("batch_size", self.config.training.batch_size)
                mlflow.log_param("learning_rate", self.config.training.learning_rate)
                mlflow.log_param("num_epochs", self.config.training.num_epochs)
                
                print("Debug: Logged training parameters successfully")
                
        except Exception as e:
            print(f"Error logging training parameters: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    # ... (나머지 클래스 메서드들은 그대로 유지)

def log_dataset_info(data_module) -> Dict[str, Any]:
    """데이터셋 정보 로깅"""
    try:
        print("\nLogging dataset info...")
        
        # 데이터셋 크기 계산
        train_size = len(data_module.train_dataset)
        val_size = len(data_module.val_dataset)
        total_size = train_size + val_size
        train_ratio = train_size / total_size
        test_ratio = val_size / total_size
        
        # 설정에서 정보 가져오기
        try:
            model_name = data_module.config.project["model_name"]
            model_config = data_module.config.models[model_name]["training"]
            num_labels = model_config["num_labels"]
            max_length = model_config["max_length"]
            print(f"Debug: Found model config for {model_name}")
            print(f"Debug: num_labels = {num_labels}, max_length = {max_length}")
        except Exception as e:
            print(f"Warning: Error getting model config: {str(e)}")
            print("Using default values...")
            num_labels = 2  # NSMC의 경우 이진 분류
            max_length = 150
        
        dataset_info = {
            'train_size': train_ratio,
            'test_size': test_ratio,
            'num_samples': {
                'train': train_size,
                'val': val_size,
                'total': total_size
            },
            'num_labels': num_labels,
            'max_length': max_length
        }
        
        print(f"\nDataset statistics:")
        print(f"  - Total samples: {total_size:,}")
        print(f"  - Train samples: {train_size:,} ({train_ratio:.1%})")
        print(f"  - Val samples: {val_size:,} ({test_ratio:.1%})")
        print(f"  - Number of labels: {num_labels}")
        print(f"  - Max sequence length: {max_length}")
        
        # 레이블 분포 계산
        try:
            train_label_dist = pd.Series(data_module.train_dataset.labels).value_counts().to_dict()
            val_label_dist = pd.Series(data_module.val_dataset.labels).value_counts().to_dict()
            
            dataset_info.update({
                'train_label_distribution': train_label_dist,
                'val_label_distribution': val_label_dist
            })
            
            print("\nLabel distribution:")
            print("Train:", train_label_dist)
            print("Val:", val_label_dist)
        except Exception as e:
            print(f"Warning: Error calculating label distribution: {str(e)}")
        
        # MLflow에 로깅
        try:
            mlflow.log_params({
                'train_size': train_ratio,
                'test_size': test_ratio,
                'num_train_samples': train_size,
                'num_val_samples': val_size,
                'total_samples': total_size,
                'num_labels': num_labels,
                'max_length': max_length
            })
            
            # 레이블 분포를 JSON으로 저장
            temp_dir = Path("temp_artifacts")
            temp_dir.mkdir(exist_ok=True)
            
            json_path = temp_dir / "label_distribution.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'train': train_label_dist,
                    'val': val_label_dist
                }, f, indent=2, ensure_ascii=False)
            
            # MLflow에 아티팩트 로깅
            mlflow.log_artifact(str(json_path), "dataset_info")
            
            # 임시 파일 정리
            json_path.unlink()
            temp_dir.rmdir()
            
            print("\nDataset info logged successfully")
            
        except Exception as e:
            print(f"Warning: Failed to log to MLflow: {str(e)}")
        
        return dataset_info
        
    except Exception as e:
        print(f"Error logging dataset info: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # 기본값 반환
        default_info = {
            'train_size': 0.8,
            'test_size': 0.2,
            'num_labels': 2,
            'max_length': 150,
            'error': str(e)
        }
        print(f"Returning default values: {default_info}")
        return default_info

def log_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> None:
    """혼동 행렬 생성 및 로깅"""
    try:
        # 혼동 행렬 계산
        cm = pd.DataFrame(
            confusion_matrix(y_true, y_pred),
            index=labels,
            columns=labels
        )
        
        # 시각화
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 이미지 저장
        plt.savefig('confusion_matrix.png', bbox_inches='tight')
        plt.close()
        
        # MLflow에 로깅
        mlflow.log_artifact('confusion_matrix.png', 'evaluation')
        os.remove('confusion_matrix.png')
        
    except Exception as e:
        print(f"Error logging confusion matrix: {str(e)}")

def convert_to_serializable(obj):
    """numpy/torch 타입을 JSON 직렬화 가능한 형태로 변환"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray, torch.Tensor)):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    return obj

def log_model(self, model, run_id: str, model_name: str):
    """MLflow에 모델 로깅"""
    try:
        print("\nSaving and registering model...")
        
        # MLflow 클라이언트를 통해 현재 로깅된 파라미터 확인
        client = mlflow.tracking.MlflowClient()
        current_run = client.get_run(run_id)
        logged_params = current_run.data.params
        
        # 모델 설정 준비
        if hasattr(self.config, "model_config") and hasattr(self.config, "training_config"):
            model_config = {
                "model_type": self.config.project["model_name"],
                "pretrained_model": getattr(self.config.model_config, "pretrained_model", ""),
                "max_length": getattr(self.config.training_config, "max_length", 150),
                "num_labels": getattr(self.config.training_config, "num_labels", 2)
            }
            print(f"Debug: Model config prepared: {model_config}")
        else:
            print("Warning: Using default model config")
            model_config = {
                "model_type": self.config.project["model_name"],
                "pretrained_model": "beomi/kcbert-base",
                "max_length": 150,
                "num_labels": 2
            }
        
        # 이미 로깅된 파라미터는 제외하고 새로운 파라미터만 로깅
        new_params = {}
        for key, value in model_config.items():
            if key not in logged_params:
                new_params[key] = value
            else:
                print(f"Debug: Parameter '{key}' already logged with value: {logged_params[key]}")
        
        with mlflow.start_run(run_id=run_id, nested=True):
            # 새로운 파라미터만 로깅
            if new_params:
                print("Logging new model parameters:")
                for key, value in new_params.items():
                    print(f"  - {key}: {value}")
                mlflow.log_params(new_params)
            
            # 임시 디렉토리에 모델 저장
            temp_dir = Path("temp_artifacts") / run_id / "model"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # 모델 상태 저장
                model_path = temp_dir / "model.pt"
                torch.save(model.state_dict(), model_path)
                
                # 모델 설정 저장
                config_path = temp_dir / "config.json"
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(model_config, f, indent=2, ensure_ascii=False)
                
                # MLflow에 아티팩트 로깅
                artifact_uri = self._get_artifact_uri(run_id)
                artifact_repo = mlflow.store.artifact.artifact_repository_registry.get_artifact_repository(artifact_uri)
                
                # 아티팩트 저장
                artifact_repo.log_artifact(str(model_path))
                artifact_repo.log_artifact(str(config_path))
                
                print(f"Model saved successfully:")
                print(f"  - Model path: {model_path}")
                print(f"  - Config path: {config_path}")
                print(f"  - Artifact URI: {artifact_uri}")
                
            finally:
                # 임시 파일 정리
                if temp_dir.exists():
                    shutil.rmtree(temp_dir.parent)
                    print("Temporary files cleaned up")
            
            # 모델을 MLflow 모델 레지스트리에 등록하고 Staging으로 승격
            model_version = self.promote_to_staging(model_name, run_id)
            print(f"Model registered and promoted to Staging (version: {model_version.version})")
            
    except Exception as e:
        print(f"Error logging model: {str(e)}")
        print(f"Debug: Error during model saving: {str(e)}")
        print(f"Debug: Error type: {type(e)}")
        print("Debug: Full traceback:")
        import traceback
        traceback.print_exc()
        raise

def get_model_path(self, run_id: str) -> str:
    """특정 run의 모델 경로 반환"""
    return os.path.join(
        str(self.config.project_root),
        "mlruns",
        self.experiment_id,
        run_id,
        "artifacts/model"
    )

def get_model_info(self, run_id: str) -> Dict[str, Any]:
    """특정 run의 모델 정보 반환"""
    try:
        run = self.client.get_run(run_id)
        return {
            "run_id": run_id,
            "experiment_id": self.experiment_id,
            "model_path": self.get_model_path(run_id),
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": run.data.tags,
            "artifacts": [
                artifact.path 
                for artifact in self.client.list_artifacts(run_id, "model")
            ]
        }
    except Exception as e:
        print(f"Error getting model info: {str(e)}")
        return None

def load_specific_model(self, run_id: str) -> Any:
    """특정 run의 모델 로드"""
    try:
        model_path = self.get_model_path(run_id)
        print(f"Loading model from: {model_path}")
        return mlflow.pytorch.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def list_all_models(self) -> List[Dict[str, Any]]:
    """모든 모델 정보 리스트 반환"""
    try:
        # 실험의 모든 run 가져오기
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string="",
            order_by=["start_time DESC"]
        )
        
        models_info = []
        for run in runs:
            model_path = self.get_model_path(run.info.run_id)
            if os.path.exists(model_path):
                models_info.append({
                    "run_id": run.info.run_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "metrics": run.data.metrics,
                    "model_path": model_path
                })
        
        return models_info
    except Exception as e:
        print(f"Error listing models: {str(e)}")
        return []

def get_best_model(self, metric_name: str = "val_accuracy", mode: str = "max") -> Optional[Dict[str, Any]]:
    """최고 성능의 모델 정보 반환"""
    try:
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=f"metrics.{metric_name} != ''",
            order_by=[f"metrics.{metric_name} {'DESC' if mode == 'max' else 'ASC'}"]
        )
        
        if not runs:
            return None
        
        best_run = runs[0]
        return {
            "run_id": best_run.info.run_id,
            "model_path": self.get_model_path(best_run.info.run_id),
            "metrics": best_run.data.metrics,
            "params": best_run.data.params
        }
    except Exception as e:
        print(f"Error getting best model: {str(e)}")
        return None

def print_model_structure(self, run_id: str):
    """모델 아티팩트 구조 출력"""
    try:
        model_path = self.get_model_path(run_id)
        print(f"\nModel Structure for Run ID: {run_id}")
        print(f"Base Path: {model_path}")
        print("\nArtifacts:")
        
        def print_artifacts(path, prefix=""):
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    print(f"{prefix}└── {item}/")
                    print_artifacts(item_path, prefix + "    ")
                else:
                    print(f"{prefix}└── {item}")
        
        print_artifacts(model_path)
    except Exception as e:
        print(f"Error printing model structure: {str(e)}")

def log_training_params(self, run_id: str):
    """학습 파라미터 로깅"""
    try:
        print("\nLogging training parameters...")
        
        # MLflow 클라이언트를 통해 현재 로깅된 파라미터 확인
        client = mlflow.tracking.MlflowClient()
        current_run = client.get_run(run_id)
        logged_params = current_run.data.params
        experiment_id = current_run.info.experiment_id
        
        # MLflow 기본 설정
        mlruns_dir = Path(self.config.project_root) / "mlruns"
        tracking_uri = f"file://{str(mlruns_dir.absolute())}"
        mlflow.set_tracking_uri(tracking_uri)
        print(f"Debug: MLflow tracking URI: {tracking_uri}")
        
        # 프로젝트 기본 정보
        params_to_log = {
            "model_name": self.config.project["model_name"],
            "dataset_name": self.config.project["dataset_name"],
            "random_state": self.config.project["random_state"]
        }
        
        # 모델 설정
        model_name = self.config.project["model_name"]
        if hasattr(self.config, "model_config"):
            training_params = {
                "pretrained_model": getattr(self.config.model_config, "pretrained_model", ""),
                "model_dir": getattr(self.config.model_config, "model_dir", "models"),
                "max_length": getattr(self.config.training_config, "max_length", 150),
                "batch_size": getattr(self.config.training_config, "batch_size", 32),
                "learning_rate": getattr(self.config.training_config, "lr", 5e-5),
                "num_epochs": getattr(self.config.training_config, "epochs", 1),
                "optimizer": getattr(self.config.training_config, "optimizer", "AdamW"),
                "lr_scheduler": getattr(self.config.training_config, "lr_scheduler", "linear"),
                "num_labels": getattr(self.config.training_config, "num_labels", 2),
                "precision": getattr(self.config.training_config, "precision", 16),
                "accumulate_grad_batches": getattr(self.config.training_config, "accumulate_grad_batches", 1)
            }
            print(f"Debug: Found model config for {model_name}")
        else:
            print(f"Warning: Model config not found for {model_name}")
            training_params = {}
        
        # 데이터셋 설정
        dataset_name = self.config.project["dataset_name"]
        if hasattr(self.config, "data"):
            dataset_params = {
                "sampling_rate": getattr(self.config.data, "sampling_rate", 1.0),
                "test_size": getattr(self.config.data, "test_size", 0.2),
                "train_data_path": getattr(self.config.data, "train_data_path", ""),
                "val_data_path": getattr(self.config.data, "val_data_path", "")
            }
            print(f"Debug: Found dataset config for {dataset_name}")
        else:
            print(f"Warning: Dataset config not found for {dataset_name}")
            dataset_params = {}
        
        # 이미 로깅된 파라미터는 제외하고 새로운 파라미터만 로깅
        new_params = {}
        for param_dict in [params_to_log, training_params, dataset_params]:
            for key, value in param_dict.items():
                if key not in logged_params:
                    new_params[key] = value
                else:
                    print(f"Debug: Parameter '{key}' already logged with value: {logged_params[key]}")
        
        if new_params:
            print("Logging new parameters:")
            for key, value in new_params.items():
                print(f"  - {key}: {value}")
            
            with mlflow.start_run(run_id=run_id, nested=True):
                mlflow.log_params(new_params)
        else:
            print("No new parameters to log")
        
        # 전체 설정을 JSON으로 저장
        try:
            config_dict = {
                "project": dict(self.config.project),
                "data": dict(self.config.data) if hasattr(self.config, "data") else {},
                "model_config": dict(self.config.model_config) if hasattr(self.config, "model_config") else {},
                "training_config": dict(self.config.training_config) if hasattr(self.config, "training_config") else {},
                "common": dict(self.config.common) if hasattr(self.config, "common") else {}
            }
            
            # 임시 디렉토리 생성 및 이전 디렉토리 정리
            temp_dir = Path("temp_artifacts")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            temp_dir.mkdir(exist_ok=True)
            
            # 설정 파일 저장
            config_path = temp_dir / "config_snapshot.json"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            # MLflow에 아티팩트 로깅
            artifact_uri = f"file://{str(mlruns_dir.absolute())}/{experiment_id}/{run_id}/artifacts"
            print(f"Debug: Artifact URI: {artifact_uri}")
            
            artifact_repo = mlflow.store.artifact.artifact_repository_registry.get_artifact_repository(artifact_uri)
            artifact_repo.log_artifact(str(config_path))
            
            # 임시 파일 정리
            try:
                shutil.rmtree(temp_dir)
                print("Temporary directory cleaned up successfully")
            except Exception as cleanup_error:
                print(f"Warning: Failed to clean up temporary directory: {cleanup_error}")
            
            print("Configuration snapshot saved successfully")
            
        except Exception as e:
            print(f"Warning: Failed to save config snapshot: {str(e)}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"Error logging training parameters: {str(e)}")
        import traceback
        traceback.print_exc()

def log_evaluation_results(self, run_id: str, metrics: Dict[str, float], y_true: List, y_pred: List):
    """평가 결과 로깅"""
    try:
        with mlflow.start_run(run_id=run_id, nested=True):
            # 메트릭 로깅 (numpy 타입 변환)
            mlflow_metrics = convert_to_serializable(metrics)
            mlflow.log_metrics(mlflow_metrics)
            
            # 혼동 행렬 로깅
            log_confusion_matrix(y_true, y_pred, labels=['Negative', 'Positive'])
            
            # 상세 평��� 결과를 JSON으로 저장
            eval_path = Path("evaluation_results.json")
            with open(eval_path, "w", encoding="utf-8") as f:
                json_data = {
                    "metrics": mlflow_metrics,
                    "predictions": {
                        "true": convert_to_serializable(y_true),
                        "pred": convert_to_serializable(y_pred)
                    }
                }
                json.dump(json_data, f, indent=2)
            mlflow.log_artifact(str(eval_path), "evaluation")
            eval_path.unlink()
    except Exception as e:
        print(f"Error logging evaluation results: {str(e)}")
        raise

def register_model_info(self, run_id: str, run_name: str, metrics: Dict[str, float]) -> Dict:
    """모델 정보 등록
    
    Args:
        run_id: MLflow 실행 ID
        run_name: 실행 이름
        metrics: 모델 평가 지표
        
    Returns:
        등록된 모델 정보
    """
    try:
        print("\nRegistering model info...")
        
        # MLflow에서 실험 정보 가져오기
        run = mlflow.get_run(run_id)
        experiment_id = run.info.experiment_id
        experiment = mlflow.get_experiment(experiment_id)
        
        # 모델 버전 정보 가져오기
        model_name = self.config.project["model_name"]
        model_versions = self.client.search_model_versions(f"run_id='{run_id}'")
        version = model_versions[0].version if model_versions else "1"
        
        # 모델 정보 구성
        model_info = {
            "experiment_name": experiment.name,
            "experiment_id": experiment_id,
            "run_id": run_id,
            "run_name": run_name,
            "model_name": model_name,
            "version": version,
            "metrics": metrics,
            "stage": "Staging",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        print("Model info prepared:")
        for key, value in model_info.items():
            if key != "metrics":
                print(f"  - {key}: {value}")
            else:
                print(f"  - metrics:")
                for metric_name, metric_value in value.items():
                    print(f"      {metric_name}: {metric_value:.4f}")
        
        # 기존 정보 로드 및 업데이트
        model_infos = self.load_model_info()
        model_infos.append(model_info)
        
        # 정보 저장
        self.save_model_info(model_infos)
        
        print("Model info registered successfully")
        return model_info
        
    except Exception as e:
        print(f"Error registering model info: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

class ModelInference:
    def __init__(self, config):
        self.config = config
        self.model_manager = MLflowModelManager(config)
        self.model = None
        self.tokenizer = None
    
    def load_production_model(self):
        """프로덕션 모델 로드"""
        model_name = self.config.project['model_name']
        
        # Production 모델 체크
        if not self.model_manager.check_production_model_exists(model_name):
            print(f"No production model found for {model_name}")
            print("Attempting to load latest model and promote to production...")
            
            # 최신 모델 정보 가져오기
            latest_model = self.model_manager.get_latest_model_info()
            if latest_model:
                print(f"Debug: Latest model info: {latest_model}")
                # Staging을 거쳐 Production으로 승격
                model_version = self.model_manager.promote_to_staging(
                    model_name, 
                    latest_model['run_id']
                )
                self.model_manager.promote_to_production(
                    model_name, 
                    model_version.version
                )
                print(f"Model {model_name} promoted to production.")
                
                # MLflow 업데이트를 위한 짧은 대기
                import time
                time.sleep(1)
                
                # 모델 로드 재시도
                return self.load_production_model()
            else:
                print("No models found in registry.")
                return None
        
        # Production 모델 로드
        model = self.model_manager.load_production_model(model_name)
        if model is None:
            print("Failed to load production model.")
            return None
        
        self.model = model
        return model
    
    def predict(self, texts: List[str]) -> List[int]:
        """텍스트 감성 분석"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model and tokenizer must be loaded first")
        
        import torch
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.training_config['max_length'],
            return_tensors="pt"
        )
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        return predictions.tolist()

# 사용 예시
if __name__ == "__main__":
    config = Config()
    
    # MLflow 모 관리 초기화
    model_manager = MLflowModelManager(config)
    
    # 현재 등록된 모델 표시
    print("\nRegistered Models:")
    model_manager.display_models()
    
    # 추론 예시
    print("\nTesting inference:")
    inference = ModelInference(config)
    if inference.load_production_model():
        texts = [
            "정말 재미있는 영화였어요!",
            "시간 낭비했네요...",
            "그저 그랬어요"
        ]
        predictions = inference.predict(texts)
        for text, pred in zip(texts, predictions):
            print(f"Text: {text}")
            print(f"Sentiment: {'긍정' if pred == 1 else '부정'}\n")
    else:
        print("Failed to load model for inference.")