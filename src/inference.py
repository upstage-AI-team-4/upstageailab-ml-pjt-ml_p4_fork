import mlflow
from pathlib import Path
import pandas as pd
import torch
from models.model_factory import ModelFactory
from model_registry import ModelRegistry

def load_model_from_mlflow(run_id: str, model_name: str) -> tuple:
    """
    MLflow에서 특정 run_id의 모델을 로드
    """
    print(f"\n=== MLflow에서 모델 로드 중... ===")
    print(f"Run ID: {run_id}")
    
    # MLflow 설정
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    # 모델 로드
    loaded_model = mlflow.transformers.load_model(f"runs:/{run_id}/model")
    
    return loaded_model["model"], loaded_model["tokenizer"]

def predict_sentiment(text: str, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    텍스트의 감성 예측
    """
    # 토큰화
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    
    # 디바이스 이동
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model = model.to(device)
    model.eval()
    
    # 예측
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=1)
        confidence = predictions[0][predicted_class].item()
    
    # 결과 반환
    sentiment = "긍정" if predicted_class.item() == 1 else "부정"
    return sentiment, confidence

def main():
    registry = ModelRegistry()
    
    # 모든 Production 모델 정보 출력
    all_models = registry.get_production_models()
    print("\n=== Production 모델 목록 ===")
    for model_name, versions in all_models.items():
        print(f"\n{model_name}:")
        for version in versions:
            print(f"  - Version {version['version']} (F1: {version['metrics'].get('eval_f1', 'N/A'):.4f})")
    
    # 특정 모델 선택 또는 최고 성능 모델 사용
    model_name = input("\n사용할 모델 이름을 입력하세요 (Enter for best model): ").strip()
    if model_name:
        prod_model = registry.get_best_production_model(model_name)
    else:
        prod_model = registry.get_best_production_model()
    
    if not prod_model:
        print("Production 모델을 찾을 수 없습니다.")
        return
        
    print(f"\n선택된 모델: {prod_model['model_name']} (Version {prod_model['version']})")
    print(f"F1 Score: {prod_model['metrics'].get('eval_f1', 'N/A'):.4f}")
    
    # 모델 로드
    model, tokenizer = load_model_from_mlflow(prod_model['run_id'], prod_model['model_name'])
    
    # 예시 텍스트로 추론
    test_texts = [
        "이 영화 정말 재미있었어요. 다음에 또 보고 싶네요!",
        "시간 낭비였습니다. 정말 별로에요.",
        "그럭저럭 볼만했어요. 나쁘지 않네요."
    ]
    
    print("\n=== 감성 분석 결과 ===")
    for text in test_texts:
        sentiment, confidence = predict_sentiment(text, model, tokenizer)
        print(f"\n입력 텍스트: {text}")
        print(f"예측 감성: {sentiment} (확률: {confidence:.2%})")

if __name__ == "__main__":
    main() 