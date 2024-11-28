from utils.mlflow_utils import MLflowLogger, MLflowConfig
from pathlib import Path
import pandas as pd
import torch
from models.model_factory import ModelFactory
from models.model_registry import ModelRegistry
from typing import Dict, Tuple, List, Optional, Any
import logging
import mlflow
from transformers import pipeline
from utils.config import Config
from datetime import datetime

logger = logging.getLogger(__name__)

config = Config()

# 모델 설정 사용
model_names = config.model['models']

def get_inference_model(model_name: str):
    """추론에 사용할 모델 선택"""
    registry = ModelRegistry()
    
    # Production 모델 먼저 확인
    production_model = registry.get_production_model(model_name)
    if production_model:
        logger.info(f"Production 모델을 사용합니다: {production_model}")
        return load_model(model_name, production_model)
    
    # Production 모델이 없으면 Staging 모델 목록 출력
    staging_models = registry.get_staging_models(model_name)
    if not staging_models:
        raise ValueError(f"사용 가능한 모델이 없습니다: {model_name}")
    
    print("\n=== 사용 가능한 Staging 모델 목록 ===")
    for idx, model in enumerate(staging_models, 1):
        print(f"\n[{idx}] Run ID: {model['run_id']}")
        print(f"Dataset: {model['dataset_name']}")
        print(f"Sampling Rate: {model['sampling_rate']}")
        print(f"Metrics: {model['metrics']}")
        print(f"Registration Time: {model['registration_time']}")
    
    while True:
        try:
            choice = int(input("\n사용할 모델 번호를 선택하세요: ")) - 1
            if 0 <= choice < len(staging_models):
                selected_model = staging_models[choice]
                logger.info(f"선택된 Staging 모델: {selected_model}")
                return load_model(model_name, selected_model)
            print("잘못된 선택입니다. 다시 선택해주세요.")
        except ValueError:
            print("숫자를 입력해주세요.")

def load_model(model_name: str, model_info: dict):
    """선택된 모델 로드"""
    try:
        # 먼저 run_id로 직접 로드 시도
        model_uri = f"runs:/{model_info['run_id']}/model"
        logger.info(f"모델 로드 시도 (URI: {model_uri})")
        
        # transformers 모델로 로드
        model = mlflow.transformers.load_model(
            model_uri=model_uri,
            return_type="pipeline"
        )
        return model
    except Exception as e:
        logger.error(f"모델 로드 중 오류 발생: {str(e)}")
        # 모델 아티팩트 경로 직접 확인
        artifact_uri = mlflow.get_artifact_uri(model_info['run_id'])
        logger.info(f"아티팩트 URI: {artifact_uri}")
        raise

class SentimentPredictor:
    """감성 분석 예측을 위한 클래스"""
    
    def __init__(self, mlflow_logger: MLflowLogger):
        self.mlflow_logger = mlflow_logger
        self.registry = ModelRegistry()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def load_model(self, run_id: str, model_name: str) -> Any:
        """MLflow에서 모델 로드"""
        logger.info(f"Loading model from run: {run_id}")
        return mlflow.transformers.load_model(
            f"runs:/{run_id}/model",
            return_type="pipeline"
        )
    
    def predict_sentiment(self, 
                         text: str, 
                         classifier: Any) -> Tuple[str, float]:
        """텍스트 감성 예측"""
        result = classifier(text)[0]
        label = result['label']
        confidence = result['score']
        
        sentiment = "긍정" if label == "LABEL_1" else "부정"
        
        # MLflow에 예측 결과 로깅
        with self.mlflow_logger.start_run(nested=True):
            self.mlflow_logger.log_metrics({
                'prediction_confidence': confidence
            })
            self.mlflow_logger.log_params({
                'input_text': text,
                'predicted_sentiment': sentiment
            })
        
        return sentiment, confidence
    
    def run_predictions(self, 
                       texts: List[str], 
                       model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """여러 텍스트에 대한 예측 수행"""
        if model_name:
            prod_model = self.registry.get_best_production_model(model_name)
        else:
            prod_model = self.registry.get_best_production_model()
            
        if not prod_model:
            raise ValueError("Production 모델을 찾을 수 없습니다.")
            
        classifier = self.load_model(prod_model['run_id'], prod_model['model_name'])
        
        results = []
        for text in texts:
            sentiment, confidence = self.predict_sentiment(text, classifier)
            results.append({
                'text': text,
                'sentiment': sentiment,
                'confidence': confidence
            })
            
        return results

def main():
    # MLflow 설정
    file_path = Path(__file__)
    file_name = file_path.stem
    config.mlflow['experiment_name'] = config.mlflow['experiment_name'] + '_' + file_name
    print(f'Experiment name: {config.mlflow["experiment_name"]}')
    
    mlflow_logger = MLflowLogger()
    predictor = SentimentPredictor(mlflow_logger)
    model_name = config.model['model_name']
    # MLflow 실행 이름 생성 - 단순화
    run_name = f"{model_name}_infer_{datetime.now().strftime('%Y%m%d')}"
    
    # 예시 텍스트로 추론
    test_texts = [
        "이 영화 정말 재미있었어요. 다음에 또 보고 싶네요!",
        "시간 낭비였습니다. 정말 별로에요.",
        "그럭저럭 볼만했어요. 나쁘지 않네요."
    ]
    
    with mlflow_logger.start_run(run_name=run_name):
        results = predictor.run_predictions(test_texts)
        
        logger.info("\n=== 감성 분석 결과 ===")
        for result in results:
            logger.info(f"\n입력 텍스트: {result['text']}")
            logger.info(f"예측 감성: {result['sentiment']} (확률: {result['confidence']:.2%})")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 