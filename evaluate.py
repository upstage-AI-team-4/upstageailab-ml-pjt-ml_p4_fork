from src.utils.evaluator import ModelEvaluator
from src.utils.config import Config
# ... 필요한 import들

def main():
    config = Config()
    
    # 모델과 데이터 로드
    model = load_model(config)
    tokenizer = load_tokenizer(config)
    data_module = load_data_module(config)
    
    # 평가 수행
    evaluator = ModelEvaluator(model, tokenizer)
    metrics = evaluator.evaluate_dataset(data_module)
    
    # 결과 출력
    print_evaluation_results(metrics)
    
    # MLflow에 결과 로깅
    log_results_to_mlflow(metrics)

if __name__ == '__main__':
    main() 