import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import pandas as pd
import torch
import mlflow
import shutil
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from src.config import Config

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
    """MLflow 초기화 및 설정
    
    Args:
        config: 설정 객체
        
    Returns:
        str: experiment_id
    """
    # MLflow 실험 설정
    experiment = mlflow.get_experiment_by_name(config.mlflow.experiment_name)
    if experiment is None:
        
        experiment_id = mlflow.create_experiment(
            name=config.mlflow.experiment_name,
            artifact_location=str(config.mlflow.artifact_location)
        )
    else:
        experiment_id = experiment.experiment_id
    
    mlflow.set_experiment(config.mlflow.experiment_name)
    
    print(f"Debug: MLflow initialized:")
    print(f"Debug: Experiment name: {config.mlflow.experiment_name}")
    print(f"Debug: Experiment ID: {experiment_id}")
    print(f"Debug: Artifact location: {config.mlflow.artifact_location}")
    
    return experiment_id

class MLflowModelManager:
    def __init__(self, config: Config):
        """MLflow 모델 관리자 초기화
        
        Args:
            config: 설정 객체
        """
        self.config = config
        self.model_info_path = Path(config.mlflow.model_info_path)
        
        # MLflow 클라이언트 설정
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        self.client = mlflow.tracking.MlflowClient()
        
        print(f"Debug: MLflow Model Manager initialized")
        print(f"Debug: Model info path: {self.model_info_path}")
        print(f"Debug: Tracking URI: {config.mlflow.tracking_uri}")
        
    def register_model(self, model_name: str, run_id: str, model_uri: str = 'model') -> ModelVersion:
        """MLflow에 모델을 등록하고 버전 정보를 반환"""
        # MLflow에 모델 등록
        model_uri = f"runs:/{run_id}/{model_uri}"
        try:
            model_version = mlflow.register_model(model_uri, model_name)
            print(f"Registered model '{model_name}' version {model_version.version}")
            
            # 모델 버전 정보 반환
            return model_version
            
        except Exception as e:
            print(f"Error registering model: {str(e)}")
            raise
    
    def promote_to_staging(self, model_name: str, run_id: str, model_uri: str = 'model') -> ModelVersion:
        """모델을 Staging 단계로 승격"""
        try:
            model_version = self.register_model(model_name, run_id, model_uri)
            
            # 모델을 Staging으로 변경
            self.client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Staging"
            )
            
            # model_registry.json 업데이트
            model_infos = self.load_model_info()
            for model_info in model_infos:
                if model_info.get('run_id') == run_id:
                    model_info['stage'] = "Staging"
                    
            # 변경된 정보 저장
            with open(self.model_info_path, 'w', encoding='utf-8') as f:
                json.dump(model_infos, f, indent=2, ensure_ascii=False)
            
            print(f"Model: {model_name}, version: {model_version.version} promoted to Staging")
            return model_version
            
        except Exception as e:
            print(f"Error promoting model to staging: {str(e)}")
            raise
    
    def promote_to_production(self, model_name: str, version: str) -> None:
        """모델을 Production 단계로 승격"""
        try:
            print(f"\nPromoting model {model_name} version {version} to Production")
            
            # MLflow에서 모델 상태 변경
            client = self.client
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production"
            )
            print(f"MLflow model stage updated to Production")
            
            # model_registry.json 업데이트
            model_infos = self.load_model_info()
            print(f"Current model infos: {json.dumps(model_infos, indent=2)}")
            
            # 기존 Production 모델들을 Archived로 변경
            for model_info in model_infos:
                if model_info['stage'] == "Production" and model_info['version'] != version:
                    model_info['stage'] = "Archived"
                    print(f"Archived previous production model: version {model_info['version']}")
            
            # 선택된 모델을 Production으로 변경
            for model_info in model_infos:
                if model_info['version'] == version:
                    model_info['stage'] = "Production"
                    print(f"Updated selected model to Production: version {version}")
                    
            # 변경된 정보 저장
            with open(self.model_info_path, 'w', encoding='utf-8') as f:
                json.dump(model_infos, f, indent=2, ensure_ascii=False)
            print(f"Updated model_registry.json saved")
            
            print(f"Model {model_name} version {version} successfully promoted to Production")
            
        except Exception as e:
            print(f"Error promoting model to production: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def archive_model(self, model_name: str, version: str) -> None:
        """모델을 Archive 단계로 이동"""
        try:
            # MLflow에서 모델 상태 변경
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Archived"
            )
            
            # model_registry.json 업데이트
            model_infos = self.load_model_info()
            for model_info in model_infos:
                if model_info.get('version') == version:
                    model_info['stage'] = "Archived"
                    
            # 변경된 정보 저장
            with open(self.model_info_path, 'w', encoding='utf-8') as f:
                json.dump(model_infos, f, indent=2, ensure_ascii=False)
            
            print(f"Model: {model_name}, version: {version} Archived")
            
        except Exception as e:
            print(f"Error archiving model: {str(e)}")
            raise
    
    def get_latest_versions(self, model_name: str, stages: Optional[List[str]] = None) -> List[ModelVersion]:
        """특정 스테이지의 최신 모델 버전들을 조회"""
        try:
            return self.client.get_latest_versions(model_name, stages)
        except Exception as e:
            print(f"Error getting latest versions: {str(e)}")
            return []
    
    def save_model_info(self, run_id: str, metrics: Dict[str, float], params: Dict[str, Any], version: str) -> None:
        """모델 정보를 JSON 파일로 저장"""
        try:
            # Path 객체를 문자열로 변환
            serializable_params = {k: str(v) if isinstance(v, Path) else v for k, v in params.items()}
            
            # experiment_id 가져오기
            run = mlflow.get_run(run_id)
            experiment_id = run.info.experiment_id
            experiment_name = mlflow.get_experiment(experiment_id).name
            
            model_info = {
                "experiment_name": experiment_name,
                "experiment_id": experiment_id,
                "run_id": run_id,
                "run_name": f"{self.config.project['model_name']}_{self.config.project['dataset_name']}",
                "metrics": metrics,
                "params": serializable_params,
                "stage": "Staging",
                "version": version,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 기존 정보 로드 및 업데이트
            model_infos = self.load_model_info()
            model_infos.append(model_info)
            
            # 파일 저장
            self.model_info_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.model_info_path, 'w', encoding='utf-8') as f:
                json.dump(model_infos, f, indent=2, ensure_ascii=False)
                
            print(f"Model info saved successfully to {self.model_info_path}")
            
        except Exception as e:
            print(f"Error saving model info: {str(e)}")
            raise
    
    def load_model_info(self) -> List[Dict]:
        """저장된 모델 정보를 JSON 파일에서 로드"""
        try:
            if self.model_info_path.exists():
                with open(self.model_info_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"Error loading model info: {str(e)}")
            return []
    
    def display_models(self) -> None:
        """저장된 모델 정보를 테이블 형태로 출력"""
        model_infos = self.load_model_info()
        if not model_infos:
            print("No models found in registry.")
            return
        
        df = pd.DataFrame(model_infos)
        display_columns = [
            "experiment_name",
            "run_name",
            "run_id",
            "metrics",
            "stage",
            "timestamp"
        ]
        df = df[display_columns]
        
        # metrics 컬럼을 보기 좋게 포맷팅
        df['metrics'] = df['metrics'].apply(lambda x: {k: f"{v:.4f}" for k, v in x.items()})
        
        # 인덱스 이름 설정 및 1부터 시작하도록 변경
        df.index = range(1, len(df) + 1)
        df.index.name = 'model_index'
        
        print("\nRegistered Models:")
        print(df.to_string())
    
    def manage_model(self, model_name: str) -> None:
        """대화형으로 모델 스테이지를 관리"""
        self.display_models()
        
        while True:
            print("\n=== Model Management ===")
            print("1. Promote model to Production")
            print("2. Archive model")
            print("3. Display models")
            print("4. View model versions")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ")
            
            if choice == '1':
                version = input("Enter model version to promote: ")
                try:
                    self.promote_to_production(model_name, version)
                    self.display_models()
                except Exception as e:
                    print(f"Error: {e}")
            
            elif choice == '2':
                version = input("Enter model version to archive: ")
                try:
                    self.archive_model(model_name, version)
                    self.display_models()
                except Exception as e:
                    print(f"Error: {e}")
            
            elif choice == '3':
                self.display_models()
            
            elif choice == '4':
                try:
                    versions = self.client.search_model_versions(f"name='{model_name}'")
                    print("\nAll model versions:")
                    for v in versions:
                        print(f"\nVersion: {v.version}")
                        print(f"Stage: {v.current_stage}")
                        print(f"Run ID: {v.run_id}")
                        print(f"Status: {v.status}")
                        print(f"Creation Time: {datetime.fromtimestamp(v.creation_timestamp/1000.0)}")
                except Exception as e:
                    print(f"Error viewing versions: {e}")
            
            elif choice == '5':
                break
            
            else:
                print("Invalid choice. Please try again.")
    
    def get_production_model_path(self, model_name: str = 'default') -> Optional[str]:
        """프로덕션 모델의 저장 경로 반환"""
        try:
            print("\nDebug: Finding production model path...")
            
            # 프로덕션 모델 정보 가져오기
            production_models = self.get_production_models()
            if not production_models:
                print("Debug: No production models found.")
                return None
                
            print(f"Debug: Found {len(production_models)} production models")
            
            # 가장 최근의 프로덕션 모델 선택
            latest_model = sorted(
                production_models,
                key=lambda x: x.get('timestamp', ''),
                reverse=True
            )[0]
            
            print(f"Debug: Selected latest model: {latest_model['run_name']}")
            print(f"Debug: Run ID: {latest_model['run_id']}")
            print(f"Debug: Experiment ID: {latest_model['experiment_id']}")
            
            # MLflow에서 모델 경로 가져오기
            experiment_id = latest_model['experiment_id']
            run_id = latest_model['run_id']
            
            # 두 가지 가능한 경로 확인
            mlruns_path = self.config.project_root / 'mlruns' / experiment_id / run_id / 'artifacts' / 'model'
           # mlartifacts_path = self.config.base_path / 'mlartifacts' / experiment_id / run_id / 'artifacts' / 'model'
            
            print(f"Debug: Checking mlruns path: {mlruns_path}")
            print(f"Debug: mlruns path exists: {os.path.exists(mlruns_path)}")
            #
            # 존재하는 경로 반환
            if os.path.exists(mlruns_path):
                return mlruns_path
            # elif os.path.exists(mlartifacts_path):
            #     return mlartifacts_path
            else:
                print("Model path not found in either mlruns or mlartifacts directories")
                return None
                
        except Exception as e:
            import traceback
            print(f"Error getting production model path: {str(e)}")
            traceback.print_exc()
            return None
    
    def load_production_model(self, model_name: str):
        """프로덕션 모델 로드"""
        try:
            # 프로덕션 모델 정보 가져오기
            production_models = self.get_production_models()
            if not production_models:
                print("No production models found.")
                return None
                
            # 가장 최근의 프로덕션 모델 선택
            latest_model = sorted(
                production_models,
                key=lambda x: x.get('timestamp', ''),
                reverse=True
            )[0]
            
            # 모델 파일 경로 확인
            model_path = self.get_production_model_path(model_name)
            if not model_path:
                print(f"Model path not found for: {model_name}")
                return None
                
            print(f"\nLoading model from: {model_path}")
            
            # config.json 로드
            config_path = Path(model_path) / "config.json"
            if not config_path.exists():
                print(f"Config file not found at: {config_path}")
                return None
                
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            print(f"Loaded config: {config}")
            
            # 기본값 설정
            default_config = {
                'num_unfreeze_layers': -1,  # 기본값 설정
                'learning_rate': 2e-5 if 'ELECTRA' in config['model_type'] else 5e-6,
                'optimizer': 'AdamW',
                'lr_scheduler': 'cosine' if 'ELECTRA' in config['model_type'] else 'exp',
                'precision': 16,
                'batch_size': 32,
                'accumulate_grad_batches': 2
            }
            
            # config에 없는 키는 기본값으로 설정
            for key, value in default_config.items():
                if key not in config:
                    print(f"Warning: {key} not found in config.json, using default value: {value}")
                    config[key] = value
            
            # 모델 타입에 따라 적절한 클래스 초기화
            if config['model_type'] == 'KcBERT':
                from src.models.kcbert_model import KcBERT
                model = KcBERT(
                    pretrained_model=config['pretrained_model'],
                    num_labels=config['num_labels'],
                    num_unfreeze_layers=config['num_unfreeze_layers']
                )
            elif config['model_type'] == 'KcELECTRA':
                from src.models.kcelectra_model import KcELECTRA
                model = KcELECTRA(
                    pretrained_model=config['pretrained_model'],
                    num_labels=config['num_labels'],
                    num_unfreeze_layers=config['num_unfreeze_layers']
                )
            else:
                raise ValueError(f"Unknown model type: {config['model_type']}")
            
            # 모델 가중치 로드
            model_pt_path = Path(model_path) / "model.pt"
            if not model_pt_path.exists():
                print(f"Model weights file not found at: {model_pt_path}")
                print(f"Checking directory contents:")
                print(f"Directory exists: {model_path.exists()}")
                if model_path.exists():
                    print(f"Files in directory:")
                    for file in model_path.iterdir():
                        print(f"  - {file}")
                return None
                
            # CPU 환경에서도 동작하도록 map_location 추가
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            state_dict = torch.load(model_pt_path, map_location=torch.device(device))
            model.load_state_dict(state_dict)
            model.eval()
            
            return model
                
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_best_model_info(self, metric: str = "val_f1") -> Optional[Dict]:
        """최고 성능의 모델 정보 반환"""
        try:
            model_infos = self.load_model_info()
            if not model_infos:
                return None
            
            sorted_models = sorted(
                model_infos,
                key=lambda x: x['metrics'].get(metric, 0),
                reverse=True
            )
            return sorted_models[0]
            
        except Exception as e:
            print(f"Error getting best model info: {str(e)}")
            return None
    
    def check_production_model_exists(self, model_name: str) -> bool:
        """Production 단계의 모델이 존재하는지 확인"""
        try:
            versions = self.client.get_latest_versions(model_name, stages=["Production"])
            return len(versions) > 0
        except Exception as e:
            print(f"Error checking production model: {str(e)}")
            return False
    
    def get_latest_model_info(self) -> Optional[Dict]:
        """가장 최근의 모델 정보 반환"""
        try:
            model_infos = self.load_model_info()
            return model_infos[-1] if model_infos else None
        except Exception as e:
            print(f"Error getting latest model info: {str(e)}")
            return None
    
    def get_production_models(self) -> List[Dict]:
        """Production 단계의 모든 모델 정보 반환"""
        try:
            model_infos = self.load_model_info()
            production_models = [
                info for info in model_infos 
                if info.get('stage') == "Production"
            ]
            return production_models
        except Exception as e:
            print(f"Error getting production models: {str(e)}")
            return []
    
    def select_production_model(self) -> Optional[Dict]:
        """Production 모델 중 하나를 선택"""
        production_models = self.get_production_models()
        
        if not production_models:
            print("No production models found.")
            return None
        
        if len(production_models) == 1:
            return production_models[0]
        
        print("\n=== Production Models ===")
        df = pd.DataFrame(production_models)
        df.index = range(1, len(df) + 1)
        df.index.name = 'model_index'
        print(df.to_string())
        
        while True:
            try:
                choice = input("\nSelect model index (or 'q' to quit): ")
                if choice.lower() == 'q':
                    return None
                
                idx = int(choice) - 1
                if 0 <= idx < len(production_models):
                    return production_models[idx]
                else:
                    print(f"Invalid index. Please enter a number between 1 and {len(production_models)}")
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    def load_production_model_info(self) -> Optional[Dict]:
        """Production 모델 정보 로드 (UI 용)"""
        try:
            production_models = self.get_production_models()
            
            if not production_models:
                return None
            
            if len(production_models) == 1:
                return production_models[0]
            
            # 가장 최근의 production 모델 반환
            return sorted(
                production_models,
                key=lambda x: x.get('timestamp', ''),
                reverse=True
            )[0]
        except Exception as e:
            print(f"Error loading production model info: {str(e)}")
            return None

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
    
    # MLflow 모델 관리 초기화
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