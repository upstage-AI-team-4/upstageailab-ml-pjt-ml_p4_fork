import mlflow
from pathlib import Path
from models.model_factory import ModelFactory
from utils.data_preprocessor import DataPreprocessor
from models.model_registry import ModelRegistry
from datetime import datetime
from pytorch_lightning import Trainer
import torch

def evaluate_model(model_name: str, pretrained_model_name: str, data_file: Path, 
                  experiment_name: str = "model_evaluation"):
    """
    기존 모델을 로드하여 평가하고 결과를 MLflow에 로깅
    """
    print(f"\n=== 모델 평가 시작: {model_name} ===")
    
    # MLflow 설정
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name)
    
    # 모델 디렉토리 설정
    models_dir = Path(__file__).parent.parent / 'models'
    pretrained_model_dir = models_dir / 'pretrained' / model_name
    
    # 모델 생성
    factory = ModelFactory()
    model = factory.get_model(
        model_name=model_name,
        data_file=data_file,
        model_dir=pretrained_model_dir,
        pretrained_model_name=pretrained_model_name,
        pretrained_model_dir=pretrained_model_dir
    )
    
    # Trainer 설정
    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # MLflow 실행 시작
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with mlflow.start_run(run_name=f"{model_name}_evaluation_{timestamp}") as run:
        # 모델 평가
        metrics = trainer.validate(model)[0]  # validate() 메서드 사용
        
        # 메트릭 로깅
        mlflow.log_metrics({
            'eval_loss': metrics['val_loss'],
            'eval_accuracy': metrics['val_acc'],
            'eval_precision': metrics['val_precision'],
            'eval_recall': metrics['val_recall'],
            'eval_f1': metrics['val_f1']
        })
        
        # 모델 정보 로깅
        mlflow.log_params({
            'model_name': model_name,
            'pretrained_model': pretrained_model_name,
            'dataset': data_file.stem
        })
        
        print("\n=== 평가 결과 ===")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

         # 데이터셋 정보 로깅
        train_df = model.train_dataset.to_pandas()
        val_df = model.val_dataset.to_pandas()
        log_dataset(train_df, "train")
        log_dataset(val_df, "val")
        
        mlflow.log_input(
            dataset=mlflow.data.from_pandas(train_df),
            context="training",
            tags={"format": "pandas", "dataset_name": data_file.stem}
        )
        mlflow.log_input(
            dataset=mlflow.data.from_pandas(val_df),
            context="validation",
            tags={"format": "pandas", "dataset_name": data_file.stem}
        )
        
        # 모델 아티팩트 로깅
        mlflow.transformers.log_model(
            transformers_model={
                "model": model.model,
                "tokenizer": model.tokenizer
            },
            artifact_path="model",
            task="sentiment-analysis",
            registered_model_name=f"{model_name}_sentiment_classifier",
            metadata={
                "source_model": pretrained_model_name,
                "task": "sentiment-analysis",
                "language": "ko",
                "dataset": data_file.stem,
                "framework": "pytorch"
            }
        )
        
        # 모델 레지스트리에 등록 (필요한 경우)
        registry = ModelRegistry(metric_threshold={'eval_f1': 0.65})
        registry.evaluate_and_register(run.info.run_id, model_name)
        
        
        return run.info.run_id, metrics
def log_dataset(df, artifact_path):
    # 데이터셋 메타데이터 생성
    metadata = {
        "columns": list(df.columns),
        "null_values": df.isnull().sum().to_dict(),
        "description": "Naver Movie Review Dataset. Train",
        "row_count": len(df),
    }

    # MLflow에 메타데이터 로깅
    mlflow.log_dict(metadata, f"{artifact_path}_metadata.json")
    mlflow.log_artifact(f"{artifact_path}.csv", artifact_path=f"{artifact_path}_datasets")
    
def main():
    # 데이터 준비
    data_dir = Path(__file__).parent.parent / 'data'
    data_prep = DataPreprocessor(data_dir=data_dir)
    data_prep.prep_naver_data(sampling_rate=0.001)
    preped_file_path = data_prep.preprocess()
    
    # 평가할 모델 설정
    models_config = {
        'KcELECTRA': 'beomi/KcELECTRA-base',
        'KcBERT': 'beomi/kcbert-base'
    }
    
    # 각 모델 평가
    results = {}
    experiment_name = "model_evaluation_sentiment_classification"
    for model_name, pretrained_model_name in models_config.items():
        run_id, metrics = evaluate_model(
            model_name=model_name,
            pretrained_model_name=pretrained_model_name,
            data_file=preped_file_path,
            experiment_name=experiment_name
        )
        results[model_name] = {'run_id': run_id, 'metrics': metrics}
    
    # 결과 요약 출력
    print("\n=== 전체 평가 결과 요약 ===")
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"Run ID: {result['run_id']}")
        print(f"F1 Score: {result['metrics']['eval_f1']:.4f}")

if __name__ == "__main__":
    main() 