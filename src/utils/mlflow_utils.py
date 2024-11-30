from typing import Optional, Dict, Any, List, Tuple
import mlflow
from mlflow.tracking import MlflowClient
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from tabulate import tabulate
import pandas as pd
from datetime import datetime

def cleanup_old_runs(config, days_to_keep=7):
    """오래된 실험 결과 정리"""
    from datetime import datetime, timedelta
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    
    # MLflow 실험 정보 조회
    client = mlflow.tracking.MlflowClient()
    for experiment in client.search_experiments():
        for run in client.search_runs(experiment.experiment_id):
            run_date = datetime.fromtimestamp(run.info.start_time / 1000.0)
            if run_date < cutoff_date:
                # threshold를 넘지 못한 오래된 실험은 삭제
                if run.data.metrics.get("val_accuracy", 0) <= config.mlflow.model_registry_metric_threshold:
                    client.delete_run(run.info.run_id) 
class ModelStage(Enum):
    """MLflow 모델 스테이지 정의"""
    STAGING = 'staging'
    PRODUCTION = 'production'
    ARCHIVED = 'archived'

@dataclass
class ModelVersion:
    """MLflow 모델 버전 정보"""
    name: str
    version: str
    stage: ModelStage

class MLflowModelManager:
    def __init__(self, config):
        self.config = config
        self.client = MlflowClient()
        self.model_info_path = config.mlflow.model_info_path
    def register_model(self, 
                      model_name: str, 
                      run_id: str, 
                      model_uri: str = 'model') -> ModelVersion:
        """MLflow에 모델을 등록하고 버전 정보를 반환"""
        model_uri = f"runs:/{run_id}/{model_uri}"
        model_version = mlflow.register_model(model_uri, model_name)
        
        return ModelVersion(
            name=model_name,
            version=model_version.version,
            stage=ModelStage.STAGING
        )
    
    def promote_to_staging(self, 
                         model_name: str, 
                         run_id: str, 
                         model_uri: str = 'model') -> ModelVersion:
        """모델을 Staging 단계로 승격"""
        model_version = self.register_model(model_name, run_id, model_uri)
        
        self.client.set_model_version_tag(
            name=model_name,
            version=model_version.version,
            key='stage',
            value=ModelStage.STAGING.value
        )
        
        print(f"Model: {model_name}, version: {model_version.version} promoted to Staging...")
        return model_version
    
    def promote_to_production(self, model_name: str, version: str) -> None:
        """모델을 Production 단계로 승격"""
        self.client.set_model_version_tag(
            name=model_name,
            version=version,
            key='stage',
            value=ModelStage.PRODUCTION.value
        )
        print(f"Model: {model_name}, version: {version} promoted to Production...")
    
    def archive_model(self, model_name: str, version: str) -> None:
        """모델을 Archive 단계로 이동"""
        self.client.set_model_version_tag(
            name=model_name,
            version=version,
            key='stage',
            value=ModelStage.ARCHIVED.value
        )
        print(f"Model: {model_name}, version: {version} Archived...")
    
    def get_latest_versions(self, model_name: str, stages: Optional[list] = None) -> list:
        """특정 스테이지의 최신 모델 버전들을 조회"""
        return self.client.get_latest_versions(model_name, stages)
    
    def save_model_info(self, run_id: str, metrics: Dict[str, float],
                       params: Dict[str, Any]) -> None:
        """모델 정보를 JSON 파일로 저장"""
        # experiment_id 가져오기
        run = self.client.get_run(run_id)
        experiment_id = run.info.experiment_id

        model_info = {
            "experiment_name": self.config.mlflow.experiment_name,
            "experiment_id": experiment_id,
            "run_id": run_id,
            "run_name": f"{self.config.project.model_name}_{self.config.project.dataset_name}",
            "metrics": metrics,
            "params": params,
            "stage": ModelStage.STAGING.value,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        model_infos = self.load_model_info()
        model_infos.append(model_info)
        
        with open(self.model_info_path, 'w') as f:
            json.dump(model_infos, f, indent=2)
    
    def load_model_info(self) -> List[Dict]:
        """저장된 모델 정보를 JSON 파일에서 로드"""
        try:
            if self.model_info_path.exists():
                with open(self.model_info_path, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"Error loading model info: {str(e)}")
            return []
    
    def display_models(self) -> None:
        """저장된 모델 정보를 테이블 형태로 표시"""
        model_infos = self.load_model_info()
        if not model_infos:
            print("No models found.")
            return
        
        df = pd.DataFrame(model_infos)
        # 표시할 컬럼 선택
        display_columns = ['experiment_name', 'run_name', 'metrics', 'stage', 'timestamp']
        df = df[display_columns]
        df['metrics'] = df['metrics'].apply(lambda x: f"acc: {x.get('val_accuracy', 0):.4f}")
        df.index.name = 'model_index'
        
        print("\n=== Model Registry ===")
        print(tabulate(df, headers='keys', tablefmt='pretty'))
    
    def manage_model(self, model_name: str) -> None:
        """대화형으로 모델 스테이지를 관리"""
        self.display_models()
        
        while True:
            print("\n=== Model Management ===")
            print("1. Promote model to Production")
            print("2. Archive model")
            print("3. Display models")
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ")
            
            if choice == '1':
                identifier = input("Enter run_id or model_index: ")
                try:
                    model_infos = self.load_model_info()
                    if identifier.isdigit():
                        run_id = model_infos[int(identifier)]['run_id']
                    else:
                        run_id = identifier
                    
                    # 먼저 staging으로 승격
                    model_version = self.promote_to_staging(model_name, run_id)
                    # 그 다음 production으로 승격
                    self.promote_to_production(model_name, model_version.version)
                    
                    # model_info.json 업데이트
                    for info in model_infos:
                        if info['run_id'] == run_id:
                            info['stage'] = ModelStage.PRODUCTION.value
                    
                    with open(self.model_info_path, 'w') as f:
                        json.dump(model_infos, f, indent=2)
                        
                except (IndexError, ValueError) as e:
                    print(f"Error: {e}")
            
            elif choice == '2':
                identifier = input("Enter run_id or model_index: ")
                try:
                    model_infos = self.load_model_info()
                    if identifier.isdigit():
                        info = model_infos[int(identifier)]
                    else:
                        info = next((info for info in model_infos if info['run_id'] == identifier), None)
                    
                    if info is None:
                        raise ValueError(f"Model with identifier {identifier} not found")
                    
                    # 모델 버전 찾기
                    versions = self.get_latest_versions(model_name, stages=['Production', 'Staging'])
                    version = next((v for v in versions if v.run_id == info['run_id']), None)
                    
                    if version:
                        # 모델 아카이브
                        self.archive_model(model_name, version.version)
                        info['stage'] = ModelStage.ARCHIVED.value
                        
                        with open(self.model_info_path, 'w') as f:
                            json.dump(model_infos, f, indent=2)
                    else:
                        print(f"No registered model version found for run_id: {info['run_id']}")
                        
                except (IndexError, ValueError) as e:
                    print(f"Error: {e}")
            
            elif choice == '3':
                self.display_models()
            
            elif choice == '4':
                break
            
            else:
                print("Invalid choice. Please try again.")
    
    # 새로 추가된 메서드들
    def get_production_model_path(self, model_name: str) -> Optional[str]:
        """프로덕션 단계의 모델 경로 반환"""
        try:
            print("\nDebug: Finding production model path...")
            versions = self.client.get_latest_versions(model_name, stages=["Production"])
            print(f"Debug: Found versions: {versions}")
            
            if not versions:
                print(f"No production model found for {model_name}")
                return None
            
            production_version = versions[0]
            run_id = production_version.run_id
            print(f"Debug: Production version info:")
            print(f"  - Version: {production_version.version}")
            print(f"  - Run ID: {run_id}")
            print(f"  - Source: {production_version.source}")
            
            # experiment ID 가져오기
            run = self.client.get_run(run_id)
            experiment_id = run.info.experiment_id
            print(f"  - Experiment ID: {experiment_id}")
            
            # 실제 아티팩트 경로 구성 (experiment ID 포함)
            artifact_path = self.config.mlflow.mlrun_path / str(experiment_id) / run_id / "artifacts/model"
            print(f"Debug: Checking artifact path: {artifact_path}")
            print(f"Debug: Path exists: {artifact_path.exists()}")
            
            if not artifact_path.exists():
                print(f"Model artifacts not found at: {artifact_path}")
                # 대체 경로 시도
                alt_path = Path(production_version.source)
                print(f"Debug: Trying alternative path: {alt_path}")
                print(f"Debug: Alt path exists: {alt_path.exists()}")
                if alt_path.exists():
                    return str(alt_path)
                return None
            
            print(f"Loading production model from: {artifact_path}")
            return str(artifact_path)
            
        except Exception as e:
            print(f"Error getting production model: {str(e)}")
            print(f"Debug: Full error details:", exc_info=True)
            return None
    
    def load_production_model(self, model_name: str):
        """프로덕션 모델 로드"""
        model_path = self.get_production_model_path(model_name)
        print(f"\nDebug: Attempting to load model from: {model_path}")
        
        if model_path:
            try:
                model_pt_path = Path(model_path) / "model.pt"
                config_path = Path(model_path) / "config.json"
                print(f"Debug: Checking paths:")
                print(f"  - model.pt: {model_pt_path.exists()}")
                print(f"  - config.json: {config_path.exists()}")
                
                # 모델 상태 딕셔너리 로드
                state_dict = torch.load(model_pt_path)
                print("Debug: Successfully loaded state_dict")
                
                # 설정 파일 로드
                with open(config_path, 'r') as f:
                    config = json.load(f)
                print(f"Debug: Loaded config: {config}")
                
                # 모델 초기화 및 가중치 로드
                from src.models.kcbert_model import KcBERT
                model = KcBERT(**self.config.get_model_kwargs())
                model.load_state_dict(state_dict)
                model.eval()
                print("Debug: Successfully initialized and loaded model")
                
                return model
                
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                import traceback
                print("Debug: Full traceback:")
                print(traceback.format_exc())
                return None
        return None
    
    def get_best_model_info(self, metric: str = "val_accuracy") -> Optional[Dict]:
        """최고 성의 모델 정보 반환"""
        try:
            with open(self.model_info_path, 'r') as f:
                model_infos = json.load(f)
            
            if not model_infos:
                return None
            
            sorted_models = sorted(
                model_infos, 
                key=lambda x: x['metrics'].get(metric, 0), 
                reverse=True
            )
            return sorted_models[0]
            
        except FileNotFoundError:
            return None
    
    def check_production_model_exists(self, model_name: str) -> bool:
        """Production 단계의 모델이 존재하는지 확인"""
        try:
            versions = self.client.get_latest_versions(model_name, stages=["Production"])
            return len(versions) > 0
        except Exception:
            return False
    
    def get_latest_model_info(self) -> Optional[Dict]:
        """가장 최근의 모델 정보 반환"""
        model_infos = self.load_model_info()
        return model_infos[-1] if model_infos else None

# 모델 추론을 위한 유틸리티 클래스
class ModelInference:
    def __init__(self, config):
        self.config = config
        self.model_manager = MLflowModelManager(config)
        self.model = None
        self.tokenizer = None
    
    def load_production_model(self):
        """프로덕션 모델 로드"""
        model_name = self.config.project.model_name
        
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
                time.sleep(2)  # 2초 대기
                
                # 모델 경로 출력
                print("\nDebug: Checking model locations:")
                print(f"Run ID: {latest_model['run_id']}")
                print(f"Experiment ID: {latest_model['experiment_id']}")
                expected_path = self.config.mlflow.mlrun_path / str(latest_model['experiment_id']) / latest_model['run_id'] / "artifacts/model"
                print(f"Expected model path: {expected_path}")
                print(f"Path exists: {expected_path.exists()}")
                
                # MLflow 등록 정보 확인
                versions = self.model_manager.client.get_latest_versions(model_name, stages=["Production"])
                print(f"Debug: Production versions after promotion: {versions}")
            else:
                print("No models found in registry.")
                return False
        
        self.model = self.model_manager.load_production_model(model_name)
        
        if self.model:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_training.pretrained_model
            )
            return True
        return False
    
    def predict(self, texts: List[str]) -> List[int]:
        """텍스트 감성 분석"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model and tokenizer must be loaded first")
        
        import torch
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.base_training.max_length,
            return_tensors="pt"
        )
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        return predictions.tolist()

# 사용 예시
if __name__ == "__main__":
    from config import Config
    
    config = Config()
    
    # MLflow 모델 관리자 초기화
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