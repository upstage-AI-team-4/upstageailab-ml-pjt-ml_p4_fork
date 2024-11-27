from pathlib import Path
from models.model_factory import ModelFactory
from utils.data_preprocessor import DataPreprocessor
from models.model_registry import ModelRegistry
from datetime import datetime
<<<<<<< HEAD
from transformers import TrainingArguments, Trainer
import pandas as pd
from typing import Dict, Tuple, Optional
import logging
from utils.mlflow_utils import MLflowLogger, MLflowConfig
=======
from pytorch_lightning import Trainer
import torch
>>>>>>> c65065e06dc6f892a3bf2f47965983ee8213ee05

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """모델 평가를 위한 클래스"""
    
    def __init__(self, mlflow_logger: MLflowLogger):
        self.mlflow_logger = mlflow_logger
        self.registry = ModelRegistry(metric_threshold={'eval_f1': 0.3})
    
<<<<<<< HEAD
    def evaluate_model(self, 
                      model_name: str, 
                      pretrained_model_name: str, 
                      dataset_name: str, 
                      data_file: Path,
                      sampling_rate: float) -> Tuple[str, Dict[str, float]]:
        """
        기존 모델을 로드하여 평가하고 결과를 MLflow에 로깅
        """
        logger.info(f"\n=== 모델 평가 시작: {model_name} ===")
        
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
=======
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
>>>>>>> c65065e06dc6f892a3bf2f47965983ee8213ee05
        )
        
        # 데이터 로드 및 준비
        model.load_data()
        model.prepare_data()
        
        # Trainer 초기화
        training_args = TrainingArguments(
            output_dir=str(model.model_dir / 'eval_results'),
            per_device_eval_batch_size=16,
            evaluation_strategy="epoch"
        )
        
<<<<<<< HEAD
        model.trainer = Trainer(
            model=model.model,
            args=training_args,
            eval_dataset=model.val_dataset,
            compute_metrics=model.compute_metrics
        )
        
        # MLflow 실행 시작
        run_name = self.mlflow_logger.generate_run_name(model_name, "evaluation")
        
        with self.mlflow_logger.start_run(run_name=run_name) as run:
            # 모델 평가
            metrics = model.evaluate()
            
            # Confusion Matrix 생성
            model.save_confusion_matrix(normalize=False)  # 일반 confusion matrix
            model.save_confusion_matrix(normalize=True)   # 정규화된 confusion matrix
            
            # Confusion Matrix MLflow에 로깅
            self.mlflow_logger.log_artifacts(
                str(model.model_dir / "confusion_matrix.png"), 
                "confusion_matrices"
            )
            self.mlflow_logger.log_artifacts(
                str(model.model_dir / "confusion_matrix_normalized.png"), 
                "confusion_matrices"
            )
            
            # 메트릭 로깅
            self.mlflow_logger.log_metrics({
                'eval_loss': metrics['eval_loss'],
                'eval_accuracy': metrics['eval_accuracy'],
                'eval_precision': metrics['eval_precision'],
                'eval_recall': metrics['eval_recall'],
                'eval_f1': metrics['eval_f1']
            })
            
            # 파라미터 로깅
            self.mlflow_logger.log_params({
                'model_name': model_name,
                'pretrained_model_name': pretrained_model_name,
                'evaluation_timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
            })
            
            # 데이터셋 정보 로깅
            train_df = model.train_dataset.to_pandas()
            val_df = model.val_dataset.to_pandas()
            
            self.mlflow_logger.log_dataset(
                train_df,
                context="training",
                dataset_name=dataset_name,
                tags={
                    "format": "pandas",
                    "dataset_name": dataset_name,
                    "dataset_file_name": data_file.stem,
                    "dataset_type": "training",
                    "sampling_rate": str(sampling_rate)
                }
            )
            self.mlflow_logger.log_dataset(
                val_df,
                context="validation",
                dataset_name=dataset_name,
                tags={
                    "format": "pandas",
                    "dataset_name": dataset_name,
                    "dataset_file_name": data_file.stem,
                    "dataset_type": "validation",
                    "sampling_rate": str(sampling_rate)
                }
            )
            
            # 모델 아티팩트 로깅
            self.mlflow_logger.log_model_artifacts(
                model=model.model,
                tokenizer=model.tokenizer,
                model_name=model_name,
                dataset_name=dataset_name
            )
            
            # 모델 레지스트리에 등록
            self.registry.evaluate_and_register(run.info.run_id, model_name)
            
            logger.info("\n=== 평가 결과 ===")
            for metric_name, value in metrics.items():
                logger.info(f"{metric_name}: {value:.4f}")
            
            return run.info.run_id, metrics
=======
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
>>>>>>> c65065e06dc6f892a3bf2f47965983ee8213ee05

    # MLflow에 메타데이터 로깅
    mlflow.log_dict(metadata, f"{artifact_path}_metadata.json")
    mlflow.log_artifact(f"{artifact_path}.csv", artifact_path=f"{artifact_path}_datasets")
    
def main():
    # MLflow 설정
    mlflow_config = MLflowConfig(experiment_name="model_evaluation")
    mlflow_logger = MLflowLogger(mlflow_config)
    evaluator = ModelEvaluator(mlflow_logger)
    
    # 데이터 준비
    dataset_name = "naver_movie_review"
    sampling_rate = 0.001
    data_dir = Path(__file__).parent.parent / 'data'
    data_prep = DataPreprocessor(data_dir=data_dir)
    
    if dataset_name == "naver_movie_review":
        data_prep.prep_naver_data(sampling_rate=sampling_rate)
    else:
        logger.error('error')
        return
        
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
        run_id, metrics = evaluator.evaluate_model(
            model_name=model_name,
            pretrained_model_name=pretrained_model_name,
<<<<<<< HEAD
            dataset_name=dataset_name,
            data_file=preped_file_path,
            sampling_rate=sampling_rate
=======
            data_file=preped_file_path,
            experiment_name=experiment_name
>>>>>>> c65065e06dc6f892a3bf2f47965983ee8213ee05
        )
        results[model_name] = {'run_id': run_id, 'metrics': metrics}
    
    # 결과 요약 출력
    logger.info("\n=== 전체 평가 결과 요약 ===")
    for model_name, result in results.items():
        logger.info(f"\n{model_name}:")
        logger.info(f"Run ID: {result['run_id']}")
        logger.info(f"F1 Score: {result['metrics']['eval_f1']:.4f}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 