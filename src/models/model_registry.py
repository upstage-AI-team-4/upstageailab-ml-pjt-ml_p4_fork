from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import logging
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient


from utils.mlflow_utils import MLflowLogger

logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self, registry_file: Optional[Path] = None):
        if registry_file is None:
            registry_file = Path(__file__).parent.parent.parent / 'config' / 'production_models.json'
        self.registry_file = registry_file
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.client = MlflowClient()
        self.mlflow_logger = MLflowLogger()
        self._load_registry()

    def _load_registry(self):
        """레지스트리 파일 로드"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r', encoding='utf-8') as f:
                self.registry = json.load(f)
        else:
            self.registry = {}
            self._save_registry()

    def _save_registry(self):
        """레지스트리 파일 저장"""
        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)

    def generate_model_name(self, 
                          model_name: str, 
                          task: str = "sentiment",
                          dataset_name: str = None,
                          version: str = None,
                          sampling_rate: float = None) -> str:
        """모델 레지스트리에 등록할 이름 생성
        
        Args:
            model_name: 기본 모델 이름 (예: KcBERT)
            
        Returns:
            str: 생성된 모델 이름 (예: "KcBERT")
        """
        # 모델 이름만 사용하여 가장 단순하게 유지
        return model_name

    def add_model(self, model_name: str, run_id: str, metrics: Dict, 
                 dataset_name: str, sampling_rate: float, threshold: float = 0.8):
        """새로운 모델을 Staging 단계로 등록"""
        logger.info(f"\n=== 모델 등록 프로세스 시작 ===")
        logger.info(f"모델명: {model_name}")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"데이터셋: {dataset_name}")
        logger.info(f"샘플링 비율: {sampling_rate}")
        logger.info(f"성능 지표: {metrics}")
        
        if model_name not in self.registry:
            logger.info(f"새로운 모델 '{model_name}' 레지스트리 생성")
            self.registry[model_name] = []

        # 성능 체크
        performance_metric = metrics.get('accuracy', 0) or metrics.get('f1_score', 0)
        logger.info(f"성능 체크 - 현재: {performance_metric:.4f}, 임계값: {threshold}")
        
        if performance_metric < threshold:
            logger.info(f"모델 성능({performance_metric:.4f})이 threshold({threshold})에 미달하여 등록하지 않습니다.")
            return None
        
        # MLflow 모델 레지스트리에 등록
        model_uri = f"runs:/{run_id}/model"
        logger.info(f"\n모델 등록 시도:")
        logger.info(f"- 모델 URI: {model_uri}")
        logger.info(f"- 등록할 모델명: {model_name}")
        
        try:
            # 현재 등록된 버전 확인
            existing_versions = self.client.search_model_versions(f"name='{model_name}'")
            logger.info(f"\n현재 등록된 버전 수: {len(existing_versions)}")
            for version in existing_versions:
                logger.info(f"- 버전 {version.version}: {version.current_stage} (Run ID: {version.run_id})")
            
            # 새 버전 등록
            model_version = mlflow.register_model(model_uri, model_name)
            logger.info(f"\n새 버전 등록 완료:")
            logger.info(f"- 버전 번호: {model_version.version}")
            logger.info(f"- 상태: {model_version.status}")
            
            # 'staging' 별칭 설정
            try:
                self.client.set_registered_model_alias(
                    name=model_name,
                    alias="staging",
                    version=model_version.version
                )
                logger.info(f"- 별칭 설정: staging")
            except Exception as e:
                logger.warning(f"모델 별칭 설정 중 오류 발생: {str(e)}")

            # Confusion Matrix 아티팩트 로깅
            try:
                # MLflow 실행 정보 가져오기
                run = mlflow.get_run(run_id)
                if not run:
                    logger.warning(f"Run ID {run_id}에 대한 실행 정보를 찾을 수 없습니다.")
                    return None
                
                # MLflow 클라이언트 생성
                client = mlflow.tracking.MlflowClient()
                
                # 현재 실행의 아티팩트 경로 가져오기
                artifacts_path = run.info.artifact_uri
                
                # 로컬 파일 시스템 경로로 변환
                if artifacts_path.startswith('file://'):
                    artifacts_path = artifacts_path[7:]
                
                # Confusion Matrix 파일 경로
                confusion_matrix_path = Path(artifacts_path) / "confusion_matrices/confusion_matrix.png"
                normalized_cm_path = Path(artifacts_path) / "confusion_matrices/confusion_matrix_normalized.png"
                
                logger.info(f"Confusion Matrix 파일 확인:")
                logger.info(f"- Run ID: {run_id}")
                logger.info(f"- Artifacts 경로: {artifacts_path}")
                logger.info(f"- 기본 CM 경로: {confusion_matrix_path}")
                logger.info(f"- 정규화 CM 경로: {normalized_cm_path}")
                
                if confusion_matrix_path.exists():
                    mlflow.log_artifact(str(confusion_matrix_path), "confusion_matrices")
                    logger.info("기본 Confusion Matrix가 로깅되었습니다.")
                else:
                    logger.warning(f"기본 Confusion Matrix 파일을 찾을 수 없습니다: {confusion_matrix_path}")
                    
                if normalized_cm_path.exists():
                    mlflow.log_artifact(str(normalized_cm_path), "confusion_matrices")
                    logger.info("정규화된 Confusion Matrix가 로깅되었습니다.")
                else:
                    logger.warning(f"정규화된 Confusion Matrix 파일을 찾을 수 없습니다: {normalized_cm_path}")
                    
            except Exception as e:
                logger.warning(f"Confusion Matrix 로깅 중 오류 발생: {str(e)}")
                logger.warning(f"오류 상세: {type(e).__name__}")
                confusion_matrix_path = None
                normalized_cm_path = None

            model_info = {
                'run_id': run_id,
                'metrics': {k: float(v) if hasattr(v, 'item') else v for k, v in metrics.items()},
                'dataset_name': dataset_name,
                'sampling_rate': sampling_rate,
                'stage': 'staging',
                'version': str(model_version.version),
                'registered_model_name': model_name,
                'registration_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'confusion_matrix_path': str(confusion_matrix_path) if 'confusion_matrix_path' in locals() else None
            }

            self.registry[model_name].append(model_info)
            self._save_registry()
            logger.info(f"\n레지스트리 정보 업데이트 완료")
            logger.info(f"=== 모델 등록 프로세스 완료 ===\n")
            
        except Exception as e:
            logger.error(f"\n모델 등록 중 오류 발생:")
            logger.error(f"- 오류 유형: {type(e).__name__}")
            logger.error(f"- 오류 메시지: {str(e)}")
            logger.error(f"=== 모델 등록 프로세스 실패 ===\n")
            raise

    def promote_to_production(self, model_name: str, run_id: str):
        """모델을 Production으로 승격"""
        if model_name not in self.registry:
            raise ValueError(f"모델 {model_name}이 레지스트리에 없습니다.")

        logger.info(f"=== 모델 버전 탐색 시작 ===")
        logger.info(f"찾는 모델명: {model_name}")
        logger.info(f"찾는 Run ID: {run_id}")

        # MLflow에서 모델 버전 찾기
        versions = self.client.search_model_versions(f"name='{model_name}'")
        logger.info(f"검색된 모델 버전 수: {len(versions)}")
        
        target_version = None
        for version in versions:
            if version.run_id == run_id:
                target_version = version
                logger.info(f"\n대상 버전 찾음:")
                logger.info(f"- 버전 번호: {version.version}")
                logger.info(f"- Run ID: {version.run_id}")
                break

        if target_version is None:
            logger.error(f"Run ID {run_id}에 해당하는 모델 버전을 찾을 수 없습니다.")
            logger.info("=== 모델 버전 탐색 실패 ===")
            raise ValueError(f"Run ID {run_id}에 해당하는 모델 버전을 찾을 수 없습니다.")

        try:
            # 1. 기존 staging alias 제거
            try:
                self.client.delete_registered_model_alias(
                    name=model_name,
                    alias="staging"
                )
                logger.info("기존 staging alias 제거됨")
            except Exception as e:
                logger.warning(f"staging alias 제거 중 오류: {str(e)}")

            # 2. production alias 설정
            self.client.set_registered_model_alias(
                name=model_name,
                alias="production",
                version=target_version.version
            )
            logger.info("production alias 설정됨")

            # 3. 태그 업데이트
            try:
                # 기존 stage 태그 제거
                self.client.delete_model_version_tag(
                    name=model_name,
                    version=target_version.version,
                    key="stage"
                )
                # 새로운 stage 태그 설정
                self.client.set_model_version_tag(
                    name=model_name,
                    version=target_version.version,
                    key="stage",
                    value="production"
                )
                logger.info("모델 태그가 production으로 업데이트됨")
            except Exception as e:
                logger.warning(f"태그 업데이트 중 오류: {str(e)}")
            
            # 4. 레지스트리 업데이트
            for model in self.registry[model_name]:
                if model['run_id'] == run_id:
                    model['stage'] = 'production'
                    model['promotion_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            self._save_registry()
            logger.info(f"모델 {model_name} (run_id: {run_id})가 Production으로 승격되었습니다.")
            
        except Exception as e:
            logger.error(f"모델 승격 중 오류 발생: {str(e)}")
            logger.error(f"상세 오류: {type(e).__name__}")
            raise

    def archive_model(self, model_name: str, run_id: str):
        """모델을 Archive 단계로 변경"""
        if model_name not in self.registry:
            raise ValueError(f"모델 {model_name}이 레지스트리에 없습니다.")

        target_model = None
        for model in self.registry[model_name]:
            if model['run_id'] == run_id:
                model['stage'] = 'ARCHIVED'
                model['archive_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                target_model = model
                break

        if target_model:
            try:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=target_model['version'],
                    stage="ARCHIVED"
                )
                self._save_registry()
                logger.info(f"모델 {model_name} (run_id: {run_id})가 ARCHIVED 되었습니다.")
            except Exception as e:
                logger.error(f"모델 아카이브 중 오류 발생: {str(e)}")
                raise
        else:
            raise ValueError(f"Run ID {run_id}를 찾을 수 없습니다.")

    def get_staging_models(self, model_name: str) -> List[Dict]:
        """Staging 단계의 모델 목록 조회"""
        if model_name not in self.registry:
            return []
        return [model for model in self.registry[model_name] if model['stage'].lower() == 'staging']

    def get_production_model(self, model_name: str) -> Optional[Dict]:
        """Production 단계의 모델 정보 조회"""
        if model_name not in self.registry:
            return None
        
        for model in self.registry[model_name]:
            # 대소문자 구분 없이 비교하도록 수정
            if model['stage'].lower() == 'production':
                return model
        return None

    def list_models(self) -> Dict[str, List[Dict]]:
        """등록된 모든 모델 정보 조회"""
        return self.registry

    def get_model_versions(self, model_name: str) -> List[Dict]:
        """특정 모델의 모든 버전 조회"""
        return self.registry.get(model_name, [])