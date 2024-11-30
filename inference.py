from src.utils.inferencer import ModelInferencer
from src.utils.config import Config
# ... 필요한 import들

def main():
    config = Config()
    
    # 모델과 토크나이저 로드
    model = load_model(config)
    tokenizer = load_tokenizer(config)
    
    # 추론기 초기화
    inferencer = ModelInferencer(model, tokenizer)
    
    # 예시 텍스트로 추론
    texts = [
        "정말 재미있는 영화였어요!",
        "시간 낭비했네요...",
        "그럭저럭 볼만했습니다."
    ]
    
    results = inferencer.predict(texts)
    
    # 결과 출력
    print_inference_results(results)

if __name__ == '__main__':
    main() 