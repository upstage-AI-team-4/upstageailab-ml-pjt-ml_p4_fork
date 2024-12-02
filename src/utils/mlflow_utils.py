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
    """MLflow 초기화 및 설정"""
    try:
        print("\nInitializing MLflow...")
        
        # 환경 변수 설정
        os.environ["MLFLOW_TRACKING_URI"] = config.mlflow.tracking_uri
        os.environ["MLFLOW_ALLOW_FILE_URI_AS_MODEL_VERSION_SOURCE"] = "true"
        os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
        
        # MLflow 기본 디렉토리 설정
        mlruns_dir = Path(config.project_root) / config.mlflow.mlrun_path
        mlruns_dir.mkdir(parents=True, exist_ok=True)
        
        # 아티팩트 저장 디렉토리 설정
        artifacts_dir = Path(config.project_root) / config.mlflow.artifact_location
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # MLflow 설정
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        print(f"Debug: MLflow tracking URI: {config.mlflow.tracking_uri}")
        
        # 실험 생성 또는 가져오기
        experiment = mlflow.get_experiment_by_name(config.mlflow.experiment_name)
        
        if experiment is None:
            # 새 실험 생성
            experiment_id = mlflow.create_experiment(
                name=config.mlflow.experiment_name,
                artifact_location=str(artifacts_dir / config.mlflow.experiment_name)
            )
            print(f"Debug: Created new experiment with ID: {experiment_id}")
        else:
            experiment_id = experiment.experiment_id
            print(f"Debug: Using existing experiment with ID: {experiment_id}")
        
        # 현재 실험 설정
        mlflow.set_experiment(config.mlflow.experiment_name)
        print(f"Debug: Set active experiment to: {config.mlflow.experiment_name}")
        
        return experiment_id
        
    except Exception as e:
        print(f"Error initializing MLflow: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

class MLflowModelManager:
    def __init__(self, config):
        self.config = config
        self.model_info_path = Path(config.mlflow.model_info_path)
        
        # 모델 레지스트리 디렉토리가 없으면 생성
        self.model_info_path.parent.mkdir(parents=True, exist_ok=True)
        
        # MLflow 초기화
        self.experiment_id = initialize_mlflow(config)
        
        # MLflow 클라이언트 초기화
        self.tracking_uri = config.mlflow.tracking_uri
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri)
        
        print(f"Debug: MLflow tracking URI: {self.tracking_uri}")
        print(f"Debug: Model info path: {self.model_info_path}")
        print(f"Debug: Experiment ID: {self.experiment_id}")
    
    def load_model_info(self) -> Dict:
        """모델 정보 JSON 파일 로드"""
        try:
            print(f"\nLoading model info from: {self.model_info_path}")
            if not self.model_info_path.exists():
                print("Model info file does not exist")
                return {}
            
            with open(self.model_info_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                print(f"Debug: Raw model info data type: {type(data)}")
                print(f"Debug: Raw model info content: {json.dumps(data, indent=2)}")
                
                if not isinstance(data, dict):
                    print(f"Warning: Model info is not a dictionary. Got {type(data)}")
                    return {}
                
                return data
            
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {str(e)}")
            return {}
        except Exception as e:
            print(f"Error loading model info: {str(e)}")
            return {}
    
    def save_model_info(
        self,
        run_id: str,
        metrics: Dict[str, float],
        params: Dict[str, Any] = None,
        version: str = None,
        model_name: str = None,
        run_name: str = None
    ) -> None:
        """모델 정보 저장"""
        try:
            print(f"\nSaving model info to {self.model_info_path}...")
            
            # 기본값 설정
            model_name = model_name or self.config.project['model_name']
            run_name = run_name or f"{model_name}_{self.config.project['dataset_name']}"
            
            # Path 객체를 문자열로 변환
            if params:
                serializable_params = {k: str(v) if isinstance(v, Path) else v for k, v in params.items()}
            else:
                serializable_params = {}
            
            # 기존 정보 로드
            model_info = self.load_model_info()
            
            # 성능 지표 확인
            main_metric = next(iter(metrics.values()))  # 첫 번째 지표 사용
            initial_stage = "None"
            
            # 새로운 모델 정
            new_model_info = {
                "model_name": model_name,
                "version": version,
                "metrics": metrics,
                "params": serializable_params,
                "run_id": run_id,
                "run_name": run_name,
                "timestamp": datetime.now().isoformat(),
                "experiment_id": self.experiment_id,
                "stage": initial_stage
            }
            
            # 모델별 정보 업데이트
            if model_name not in model_info:
                model_info[model_name] = []
            model_info[model_name].append(new_model_info)
            
            # 정보 저장
            self.model_info_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Debug: Saving model info to: {self.model_info_path}")
            print(f"Debug: Model info content: {new_model_info}")
            
            with open(self.model_info_path, "w", encoding="utf-8") as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)
            
            print(f"Debug: Successfully saved model info")
            
            # 성능 지표가 threshold를 넘으면 자동으로 Staging으로 승격
            if main_metric >= self.config.mlflow.model_registry_metric_threshold:
                print(f"Model performance ({main_metric:.4f}) exceeds threshold ({self.config.mlflow.model_registry_metric_threshold})")
                self.promote_to_staging(model_name, version)
            
        except Exception as e:
            print(f"Error saving model info: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def register_model(self, model_name: str, run_id: str) -> ModelVersion:
        """모델을 MLflow 모델 레지스트리에 등록"""
        try:
            print(f"\nRegistering model {model_name} from run {run_id}...")
            
            # 환경 변수 재설정 (안전을 위해)
            os.environ["MLFLOW_ALLOW_FILE_URI_AS_MODEL_VERSION_SOURCE"] = "true"
            
            # 실제 모델 경로 구성
            mlruns_dir = Path(self.config.project_root) / self.config.mlflow.mlrun_path
            model_path = mlruns_dir / str(self.experiment_id) / run_id / "artifacts/model"
            
            print(f"Debug: Looking for model at: {model_path}")
            
            # 모델이 존재하는지 확인
            if not model_path.exists():
                print(f"Warning: Model not found at primary location: {model_path}")
                # 대체 경로 확인 (artifacts 디렉토리)
                artifacts_dir = Path(self.config.project_root) / self.config.mlflow.artifact_location
                alt_model_path = artifacts_dir / str(self.experiment_id) / run_id / "artifacts/model"
                
                if alt_model_path.exists():
                    model_path = alt_model_path
                    print(f"Debug: Found model at alternate location: {model_path}")
                else:
                    raise FileNotFoundError(f"Model not found at either location: {model_path} or {alt_model_path}")
            
            # 모델 등록 시도
            try:
                # 먼저 runs:/ 형식으로 시도
                model_uri = f"runs:/{run_id}/model"
                print(f"Debug: Attempting to register model using runs URI: {model_uri}")
                model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
            except Exception as e1:
                print(f"Warning: Failed to register model using runs URI: {e1}")
                try:
                    # file:// 형식으로 시도
                    model_uri = f"file://{str(model_path)}"
                    print(f"Debug: Attempting to register model using file URI: {model_uri}")
                    model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
                except Exception as e2:
                    print(f"Warning: Failed to register model using file URI: {e2}")
                    # 마지막으로 직접 경로 시도
                    print(f"Debug: Attempting to register model using direct path: {model_path}")
                    model_version = mlflow.register_model(model_uri=str(model_path), name=model_name)
            
            print(f"Debug: Successfully registered model version: {model_version.version}")
            return model_version
            
        except Exception as e:
            print(f"Error registering model: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def log_config_params(self, run_id: str):
        """Log important configuration parameters"""
        with mlflow.start_run(run_id=run_id, nested=True):
            # Project common parameters
            mlflow.log_params({
                "project_name": self.config.project['name'],
                "model_name": self.config.project['model_name'],
                "base_path": str(self.config.project_root),
                "experiment_name": self.experiment_name
            })
            
            # Training parameters
            training_params = {
                "model_type": self.config.model_config['type'],
                "pretrained_model_name": self.config.model_config['pretrained_model'],
                "max_length": self.config.model_config['max_length'],
                "batch_size": self.config.training_config['batch_size'],
                "learning_rate": self.config.training_config['learning_rate'],
                "num_epochs": self.config.training_config['num_epochs'],
                "early_stopping_patience": self.config.training_config['early_stopping_patience'],
                "warmup_steps": self.config.training_config['warmup_steps'],
                "weight_decay": self.config.training_config['weight_decay'],
                "gradient_clip_val": self.config.training_config['gradient_clip_val']
            }
            mlflow.log_params(training_params)
            
            # Save complete config as JSON artifact
            config_dict = self.config.to_dict()
            config_path = Path(self.config.project_root) / "artifacts" / "config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            mlflow.log_artifact(str(config_path), "config")
    
    def get_production_model_path(self, model_name: str) -> Optional[str]:
        """현재 Production 단계에 있는 모델의 경로를 반환"""
        try:
            print("\nLooking for production model...")
            
            # 모델 정보 로드
            model_info = self.load_model_info()
            print(f"Debug: Model info type after load: {type(model_info)}")
            
            if not isinstance(model_info, dict):
                print(f"Error: model_info is not a dictionary: {type(model_info)}")
                return None
            
            if model_name not in model_info:
                print(f"No model info found for {model_name}")
                return None
            
            versions = model_info[model_name]
            print(f"Debug: Versions type: {type(versions)}")
            print(f"Debug: Found {len(versions)} versions")
            
            if not isinstance(versions, list):
                print(f"Error: versions is not a list: {type(versions)}")
                return None
            
            # Production 모델 찾기
            production_versions = []
            for version in versions:
                if not isinstance(version, dict):
                    print(f"Warning: Invalid version format: {version}")
                    continue
                    
                stage = version.get('stage', '')
                print(f"Debug: Checking version {version.get('version', 'unknown')} (stage: {stage})")
                
                if isinstance(stage, str) and stage.upper() == "PRODUCTION":
                    production_versions.append(version)
                    print(f"Debug: Found production version {version.get('version', 'unknown')}")
                    
            if not production_versions:
                print("No production version found")
                return None
            
            # 최신 버전 선택
            latest_version = production_versions[0]
            print(f"Debug: Selected version: {latest_version.get('version', 'unknown')}")
            
            # 필수 정보 확인
            run_id = str(latest_version.get('run_id', ''))
            experiment_id = str(latest_version.get('experiment_id', ''))
            
            if not run_id or not experiment_id:
                print(f"Missing required info - run_id: {run_id}, experiment_id: {experiment_id}")
                return None
            
            # 모델 파일 경로 구성
            mlruns_path = os.path.join(
                self.config.project_root,
                self.config.mlflow.mlrun_path,
                experiment_id,
                run_id,
                "artifacts/model"
            )
            
            artifacts_path = os.path.join(
                self.config.project_root,
                self.config.mlflow.artifact_location,
                experiment_id,
                run_id,
                "artifacts/model"
            )
            
            print(f"Checking model paths:")
            print(f"- MLruns path: {mlruns_path}")
            print(f"- Artifacts path: {artifacts_path}")
            
            # 경로 존재 확인
            if os.path.exists(mlruns_path):
                print(f"Found model at: {mlruns_path}")
                return mlruns_path
            elif os.path.exists(artifacts_path):
                print(f"Found model at: {artifacts_path}")
                return artifacts_path
                
            print("Model files not found at any expected location")
            return None
            
        except Exception as e:
            print(f"Error getting production model path: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_production_model(self, model_name: str):
        """프로덕션 모델 로드"""
        try:
            print("\nLoading production model...")
            
            # 프로덕션 모델 경로 가져오기
            model_path = self.get_production_model_path(model_name)
            if not model_path:
                print("No production model path found")
                return None
            
            print(f"Loading model from: {model_path}")
            
            # 모델 파일 경로
            model_pt_path = Path(model_path) / "model.pt"
            config_path = Path(model_path) / "config.json"
            
            # 파일 존재 확인
            if not model_pt_path.exists() or not config_path.exists():
                missing_files = []
                if not model_pt_path.exists():
                    missing_files.append(f"model.pt (path: {model_pt_path})")
                if not config_path.exists():
                    missing_files.append(f"config.json (path: {config_path})")
                print(f"Missing required files: {', '.join(missing_files)}")
                return None
            
            # 모델 설정 로드
            print("Loading model config...")
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                print(f"Debug: Loaded config: {json.dumps(config, indent=2)}")
            
            # 모델 초기화
            print(f"Initializing {model_name} model...")
            if model_name == "KcBERT":
                from src.models.kcbert_model import KcBERT
                model = KcBERT(**config)
            elif model_name == "KcELECTRA":
                from src.models.kcelectra_model import KcELECTRA
                model = KcELECTRA(**config)
            else:
                print(f"Unsupported model type: {model_name}")
                return None
            
            # 모델 가중치 로드
            print("Loading model weights...")
            state_dict = torch.load(model_pt_path)
            model.load_state_dict(state_dict)
            model.eval()
            
            print(f"Successfully loaded {model_name} model")
            return model
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def promote_to_staging(self, model_name: str, version: str) -> None:
        """모델을 Staging 단계로 승격"""
        try:
            # MLflow에서 스테이지 변경
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Staging"
            )
            
            # 로컬 정보 업데이트
            model_info = self.load_model_info()
            if model_name in model_info:
                for version_info in model_info[model_name]:
                    if str(version_info['version']) == version:
                        version_info['stage'] = "Staging"
                        break
                
                # 변경사항 저장
                with open(self.model_info_path, "w", encoding="utf-8") as f:
                    json.dump(model_info, f, indent=2, ensure_ascii=False)
            
            print(f"Model {model_name} version {version} promoted to Staging")
            
        except Exception as e:
            print(f"Error promoting model to staging: {str(e)}")
            raise
    
    def promote_to_production(self, model_name: str, version: str) -> None:
        """모델을 Production 단계로 승격"""
        try:
            # 현재 Production 모델이 있다면 Archived로 변경
            current_versions = self.client.search_model_versions(f"name='{model_name}'")
            for v in current_versions:
                if v.current_stage == "Production":
                    self.client.transition_model_version_stage(
                        name=model_name,
                        version=v.version,
                        stage="Archived"
                    )
                    print(f"Archived previous production model: version {v.version}")
                    
                    # model_registry.json에서도 해당 모델의 스테이지를 Archived로 변경
                    model_infos = self.load_model_info()
                    for model_info in model_infos:
                        if (model_info.get('version') == v.version and 
                            model_info.get('stage') == "Production"):
                            model_info['stage'] = "Archived"
            
            # 새 모델을 Production으로 승격
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production"
            )
            print(f"Model {model_name} version {version} promoted to Production")
            
            # model_registry.json에서도 해당 모델의 스테이지를 Production으로 변경
            model_infos = self.load_model_info()
            for model_info in model_infos:
                if model_info.get('version') == version:
                    model_info['stage'] = "Production"
                    
            # 변경된 정보 저장
            with open(self.model_info_path, 'w', encoding='utf-8') as f:
                json.dump(model_infos, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            print(f"Error promoting model to production: {str(e)}")
            raise
    
    def archive_model(self, model_name: str, version: str) -> None:
        """모델을 Archive 단계로 이동"""
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Archived"
            )
            print(f"Model: {model_name}, version: {version} Archived...")
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
    
    def display_models(self) -> None:
        """저장된 모델 정보를 테이블 형태로 출력"""
        try:
            model_info = self.load_model_info()
            if not model_info:
                print("No models found in registry.")
                return

            # 모든 모델의 정보를 하나의 리스트로 변환
            all_models = []
            for model_name, versions in model_info.items():
                for version_info in versions:
                    # 메트릭스를 문자로 변환
                    metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in version_info.get("metrics", {}).items()])
                    
                    model_data = {
                        "model_name": model_name,
                        "version": version_info.get("version", "N/A"),
                        "run_name": version_info.get("run_name", "N/A"),
                        "metrics": metrics_str,
                        "stage": version_info.get("stage", "None"),
                        "timestamp": version_info.get("timestamp", "N/A")
                    }
                    all_models.append(model_data)

            if not all_models:
                print("No model versions found.")
                return

            # DataFrame 생성 및 표시
            df = pd.DataFrame(all_models)
            display_columns = ["model_name", "version", "run_name", "metrics", "stage", "timestamp"]
            df = df[display_columns]
            
            print("\nRegistered Models:")
            print(df.to_string(index=False))

        except Exception as e:
            print(f"Error displaying models: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def manage_model(self, model_name: str) -> None:
        """대델 버전 관리"""
        try:
            # 모델 정보 로드
            model_info = self.load_model_info()
            if model_name not in model_info:
                print(f"No versions found for model {model_name}")
                return
            
            # 모델 버전 표시
            versions = model_info[model_name]
            print(f"\nVersions for model {model_name}:")
            for version in versions:
                print(f"Version: {version['version']}, Stage: {version['stage']}")
            
            # 사용자 입력 받기
            version_to_promote = input("\nEnter model version to promote: ")
            
            # MLflow 클라이언트에서 모델 버전 정보 가져오기
            try:
                model_version = self.client.get_model_version(
                    name=model_name,
                    version=version_to_promote
                )
            except Exception as e:
                print(f"Error getting model version from MLflow: {e}")
                return
            
            # 선택된 버전 찾기
            selected_version = None
            for version in versions:
                if str(version['version']) == version_to_promote:
                    selected_version = version
                    break
            
            if not selected_version:
                print(f"Version {version_to_promote} not found")
                return
            
            # 현재 스테이지 확인
            current_stage = selected_version['stage']
            
            if current_stage == "None":
                print("Cannot promote model in 'None' stage. Model must be in 'Staging' stage first.")
            elif current_stage == "Staging":
                # MLflow에서 Production으로 승격
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=version_to_promote,
                    stage="Production",
                    archive_existing_versions=True  # 기존 Production 모델을 자동으로 Archive
                )
                
                # 로컬 정보 업데이트
                selected_version['stage'] = "Production"
                print(f"Model {model_name} version {version_to_promote} promoted to Production")
                
                # 기존 Production 모델이 있다면 Archived로 변경
                for other_version in versions:
                    if other_version != selected_version and other_version['stage'] == "Production":
                        other_version['stage'] = "Archived"
                        print(f"Previous production model (version {other_version['version']}) archived")
            
            elif current_stage == "Production":
                # Archive 옵션
                action = input("Model is in Production. Enter 'archive' to archive it: ")
                if action.lower() == 'archive':
                    # MLflow에서 Archived로 변경
                    self.client.transition_model_version_stage(
                        name=model_name,
                        version=version_to_promote,
                        stage="Archived"
                    )
                    
                    # 로컬 정보 업데이트
                    selected_version['stage'] = "Archived"
                    print(f"Model {model_name} version {version_to_promote} archived")
            
            # 변경사항 저장
            with open(self.model_info_path, "w", encoding="utf-8") as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)
            
            # MLflow 모델 버전 상태 확인
            updated_version = self.client.get_model_version(
                name=model_name,
                version=version_to_promote
            )
            print(f"\nMLflow model version status:")
            print(f"Stage: {updated_version.current_stage}")
            print(f"Status: {updated_version.status}")
            
        except Exception as e:
            print(f"Error managing model: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def get_best_model_info(self, metric: str = "val_f1") -> Optional[Dict]:
        """최고 성능의 모델 정보 반환"""
        try:
            model_info = self.load_model_info()
            if not model_info:
                return None
            
            best_model = None
            best_metric = float('-inf')
            
            # 모든 모델 버전을 확인
            for model_name, versions in model_info.items():
                for version in versions:
                    metrics = version.get('metrics', {})
                    if metric in metrics:
                        current_metric = float(metrics[metric])
                        if current_metric > best_metric:
                            best_metric = current_metric
                            best_model = version
            
            return best_model
            
        except Exception as e:
            print(f"Error getting best model: {str(e)}")
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
    
    def log_model(self, model: Any, run_id: str, model_name: str):
        """모델을 MLflow에 저장"""
        print(f"\nSaving model artifacts...")
        print(f"Debug: Run ID: {run_id}")
        print(f"Debug: Experiment ID: {self.experiment_id}")
        
        try:
            with mlflow.start_run(run_id=run_id, nested=True) as run:
                # 모델 저장 경로 설정
                artifact_path = "model"
                save_path = os.path.join(self.artifact_location, self.experiment_id, run_id, "artifacts", artifact_path)
                print(f"Debug: Full save path: {save_path}")
                
                # 저장 디렉토리 생성
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # 모델 저장
                mlflow.pytorch.log_model(
                    model,
                    artifact_path,
                    registered_model_name=model_name
                )
                print(f"Debug: Model saved successfully")
                
                # 저장된 파일 확인
                if os.path.exists(save_path):
                    print(f"Debug: Verified model files at: {save_path}")
                    print(f"Debug: Directory contents: {os.listdir(save_path)}")
                else:
                    print(f"Warning: Model directory not found at: {save_path}")
                
        except Exception as e:
            print(f"Error saving model: {str(e)}")
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