import logging
from models.model_registry import ModelRegistry
import mlflow
import os

logger = logging.getLogger(__name__)

def list_staging_models(model_name: str):
    """Staging 단계의 모델 목록 출력"""
    registry = ModelRegistry()
    staging_models = registry.get_staging_models(model_name)
    
    print(f"\n=== Staging Models for {model_name} ===")
    for model in staging_models:
        print(f"\nRun ID: {model['run_id']}")
        print(f"Version: {model['version']}")
        print(f"Dataset: {model['dataset_name']}")
        print(f"Sampling Rate: {model['sampling_rate']}")
        print(f"Metrics: {model['metrics']}")
        print(f"Registration Time: {model['registration_time']}")

def promote_model_to_production(model_name: str, run_id: str):
    """선택한 모델을 Production으로 승격"""
    registry = ModelRegistry()
    client = mlflow.tracking.MlflowClient()
    
    # run_id로 모델 찾기
    target_model = None
    for model in registry.registry[model_name]:
        if model['run_id'] == run_id:
            target_model = model
            break
    
    if target_model is None:
        logger.error(f"Run ID {run_id}를 찾을 수 없습니다.")
        return
    
    try:
        # 1. 모델 URI 생성
        model_uri = f"runs:/{run_id}/model"
        
        # 2. 모델 버전 검색 (수정된 부분)
        versions = client.search_model_versions(f"name='{model_name}'")
        target_version = None
        for version in versions:
            if version.run_id == run_id:
                target_version = version
                break
                
        if not target_version:
            # 2-1. 모델 버전을 찾지 못한 경우, 새로 등록
            logger.info(f"모델 버전을 새로 등록합니다: {model_uri}")
            model_details = mlflow.register_model(model_uri, model_name)
            target_version = client.get_model_version(
                name=model_details.name,
                version=model_details.version
            )
        
        # 3. production 별칭 설정 (새로운 방식)
        try:
            client.set_registered_model_alias(
                name=model_name,
                alias="production",
                version=target_version.version
            )
            logger.info(f"Production 별칭이 설정되었습니다. (버전: {target_version.version})")
        except Exception as e:
            logger.warning(f"별칭 설정 중 오류 발생: {str(e)}")
            # 기존 방식으로 시도
            client.transition_model_version_stage(
                name=model_name,
                version=target_version.version,
                stage="production"
            )
        
        # 4. 레지스트리 업데이트
        registry.promote_to_production(model_name, run_id)
        logger.info(f"모델이 성공적으로 Production으로 승격되었습니다.")
        
    except Exception as e:
        logger.error(f"MLflow 작업 중 오류 발생: {str(e)}")
        logger.error(f"상세 오류: {type(e).__name__}")

def archive_specific_model(model_name: str, run_id: str):
    """선택한 모델을 Archive로 변경"""
    registry = ModelRegistry()
    registry.archive_model(model_name, run_id)

def show_model_status(model_name: str):
    """모든 모델의 현재 상태 출력"""
    registry = ModelRegistry()
    models = registry.get_model_versions(model_name)
    
    print(f"\n=== Model Status for {model_name} ===")
    print("\n[Production Models]")
    production_model = registry.get_production_model(model_name)
    if production_model:
        print_model_info(production_model)
    else:
        print("No production model available")
    
    print("\n[Staging Models]")
    staging_models = registry.get_staging_models(model_name)
    if staging_models:
        for model in staging_models:
            print_model_info(model)
    else:
        print("No staging models available")

def print_model_info(model: dict):
    """모델 정보 출력 헬퍼 함수"""
    print(f"\nRun ID: {model['run_id']}")
    print(f"Version: {model['version']}")
    print(f"Dataset: {model['dataset_name']}")
    print(f"Sampling Rate: {model['sampling_rate']}")
    print(f"Metrics: {model['metrics']}")
    print(f"Stage: {model['stage']}")
    print(f"Registration Time: {model['registration_time']}")

if __name__ == "__main__":
    # 현재 모델 상태 출력
    show_model_status("KcBERT")
    
    # 작업 선택
    print("\n수행할 작업을 선택하세요:")
    print("1. Staging 모델을 Production으로 승격")
    print("2. 모델을 Archive로 이동")
    print("3. 종료")
    
    choice = input("선택 (1-3): ")
    
    if choice == "1":
        # Staging 모델 목록 출력
        staging_models = ModelRegistry().get_staging_models("KcBERT")
        if not staging_models:
            print("Staging 상태의 모델이 없습니다.")
        else:
            print("\n=== Staging 모델 목록 ===")
            for model in staging_models:
                print(f"\nRun ID: {model['run_id']}")
                print(f"등록된 모델 이름: {model['registered_model_name']}")
                print(f"성능 메트릭: {model['metrics']}")
        
        run_id = input("\n승격할 모델의 Run ID를 입력하세요: ")
        promote_model_to_production("KcBERT", run_id)
    elif choice == "2":
        run_id = input("Archive할 모델의 Run ID를 입력하세요: ")
        archive_specific_model("KcBERT", run_id) 