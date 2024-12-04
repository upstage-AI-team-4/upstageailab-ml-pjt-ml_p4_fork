import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.utils.inferencer import ModelInferencer
from src.config import Config
from src.utils.mlflow_utils import MLflowModelManager
from transformers import AutoTokenizer

def load_model_and_tokenizer(config):
    """Load production model and tokenizer"""
    model_manager = MLflowModelManager(config)
    
    # Get production model info
    model_info = model_manager.load_production_model_info()
    if model_info is None:
        raise RuntimeError("No production model found. Please train and promote a model first.")
    
    print("\n=== Loading Production Model ===")
    print(f"Model: {model_info['run_name']}")
    print(f"Metrics: {model_info['metrics']}")
    print(f"Stage: {model_info['stage']}")
    print(f"Timestamp: {model_info['timestamp']}")
    
    # Load model
    model = model_manager.load_production_model(config.project['model_name'])
    if model is None:
        raise RuntimeError("Failed to load the model. Please check if the model files exist.")
    
    # Load tokenizer based on model info
    tokenizer = AutoTokenizer.from_pretrained(model_info['params']['pretrained_model'])
    
    return model, tokenizer

def main():
    config = Config()
    
    try:
        model, tokenizer = load_model_and_tokenizer(config)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    inferencer = ModelInferencer(model, tokenizer)
    
    print("\n=== Sample Predictions ===")
    texts = [
        "정말 재미있는 영화였어요!",
        "시간 낭비했네요...",
        "그저 그랬어요. 기대만큼은 아니었네요.",
        "최고의 영화! 또 보고 싶습니다.",
        "스토리가 너무 뻔해서 재미없었어요."
    ]
    
    results = inferencer.predict(texts)
    
    for text, result in zip(texts, results):
        print("\nText:", text)
        print(f"Prediction: {'긍정' if result['prediction'] == 1 else '부정'}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("-" * 80)
    
    print("\n=== Interactive Mode ===")
    print("Enter text to analyze (or 'q' to quit):")
    
    while True:
        text = input("\nText: ").strip()
        if text.lower() == 'q':
            break
            
        if not text:
            continue
            
        result = inferencer.predict(text)[0]
        print(f"Prediction: {'긍정' if result['prediction'] == 1 else '부정'}")
        print(f"Confidence: {result['confidence']:.4f}")

if __name__ == '__main__':
    main() 