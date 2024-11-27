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

logger = logging.getLogger(__name__)

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
    mlflow_config = MLflowConfig(experiment_name="model_inference")
    mlflow_logger = MLflowLogger(mlflow_config)
    
    predictor = SentimentPredictor(mlflow_logger)
    
    # 모든 Production 모델 정보 출력
    all_models = predictor.registry.get_production_models()
    logger.info("=== Production 모델 목록 ===")
    for model_name, versions in all_models.items():
        logger.info(f"\n{model_name}:")
        for version in versions:
            logger.info(f"  - Version {version['version']} (F1: {version['metrics'].get('eval_f1', 'N/A'):.4f})")
    
    # 예시 텍스트로 추론
    test_texts = [
        "이 영화 정말 재미있었어요. 다음에 또 보고 싶네요!",
        "시간 낭비였습니다. 정말 별로에요.",
        "그럭저럭 볼만했어요. 나쁘지 않네요."
    ]
    
    with mlflow_logger.start_run(
        run_name=mlflow_logger.generate_run_name("sentiment_inference")
    ):
        results = predictor.run_predictions(test_texts)
        
        logger.info("\n=== 감성 분석 결과 ===")
        for result in results:
            logger.info(f"\n입력 텍스트: {result['text']}")
            logger.info(f"예측 감성: {result['sentiment']} (확률: {result['confidence']:.2%})")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 