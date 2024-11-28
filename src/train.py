# main.py

from utils.data_preprocessor import DataPreprocessor
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from models.model_factory import ModelFactory
from datetime import datetime
import logging
from utils.mlflow_utils import MLflowLogger
from models.model_registry import ModelRegistry
from utils.config import Config
import mlflow

logger = logging.getLogger(__name__)
config = Config()

# 설정값 사용
data_name = config.data['name']
sampling_rate = config.data['sampling_rate']
model_register_threshold = config.model['register_threshold']
num_unfrozen_layers = config.model['num_unfrozen_layers']

class ModelTrainer:
    """모델 학습을 위한 클래스"""
    
    def __init__(self, mlflow_logger: MLflowLogger):
        self.mlflow_logger = mlflow_logger
        
    def train_model(self, 
                   model_name: str,
                   pretrained_model_name: str,
                   dataset_name: str,
                   data_file: Path,
                   base_model_dir: Path,
                   pretrained_model_dir: Path,
                   train_args: Dict,
                   num_unfrozen_layers: int = 2) -> Tuple[str, Dict[str, float]]:
        """단일 모델 학습 및 평가 수행"""
        logger.info(f"\n=== 모델 학습 시작: {model_name} ===")
        
        # MLflow 실행 이름 생성 - 단순화
        run_name = f"{model_name}_train_{datetime.now().strftime('%Y%m%d')}"
        
        with self.mlflow_logger.run_with_logging(
            "training", 
            model_name,
            dataset_name, 
            train_args.get('sampling_rate', 1.0),
            run_name=run_name
        ) as run:
            run_id = run.info.run_id
            model_dir = base_model_dir / run_id
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # 모델 생성 및 학습
            factory = ModelFactory()
            model = factory.get_model(
                model_name=model_name,
                data_file=data_file,
                model_dir=model_dir,
                pretrained_model_name=pretrained_model_name,
                pretrained_model_dir=pretrained_model_dir
            )
            
            model.set_mlflow_logger(self.mlflow_logger)
            model.train(train_args=train_args, num_unfrozen_layers=num_unfrozen_layers)
            metrics = model.evaluate()
            
            # 예측 수행
            predictions = model.predict(model.val_dataset.to_pandas())
            
            # MLflow 로깅
            self.mlflow_logger.log_evaluate(
                metrics=metrics,
                model=model.model,
                tokenizer=model.tokenizer,
                train_data=model.train_dataset.to_pandas(),
                val_data=model.val_dataset.to_pandas(),
                model_name=model_name,
                dataset_name=dataset_name,
                sampling_rate=train_args.get('sampling_rate', 1.0)
            )
            
            # 모델 레지스트리에 등록 시도
            registry = ModelRegistry()
            registry.add_model(
                model_name=model_name,
                run_id=run_id,
                metrics=metrics,
                dataset_name=dataset_name,
                sampling_rate=train_args.get('sampling_rate', 1.0),
                threshold=config.model['register_threshold']
            )
            
            return run_id, metrics

def main():
    # MLflow 설정
    file_path = Path(__file__)
    file_name = file_path.stem
    config.mlflow['experiment_name'] = config.mlflow['experiment_name'] + '_' + file_name + '_1' 
    print(f'Experiment name: {config.mlflow["experiment_name"]}')
    
    mlflow_logger = MLflowLogger()
    trainer = ModelTrainer(mlflow_logger)
    
    # 데이터 준비
    base_dir = Path(__file__).parent
    data_dir = base_dir.parent / 'data'
    
    data_prep = DataPreprocessor(data_dir=data_dir)
    
    # config에서 sampling_rate 가져오기
    sampling_rate = config.data['sampling_rate']
    
    # 데이터 준비 과정 로깅 추가
    logger.info("데이터 전처리 시작")
    df = data_prep.prep_naver_data(sampling_rate=sampling_rate)
    
    # 컬럼 이름 확인 및 변경
    if 'text' not in df.columns and 'document' in df.columns:
        logger.info("'document' 컬럼을 'text' 컬럼으로 변경합니다.")
        df = df.rename(columns={'document': 'text'})
    elif 'text' not in df.columns:
        raise ValueError("'text' 또는 'document' 컬럼이 없습니다.")
    
    # 데이터 크기 확인
    logger.info(f"전처리된 데이터 크기: {len(df)}")
    if len(df) == 0:
        raise ValueError("전처리된 데이터가 비어있습니다.")
    
    # 전처리된 데이터를 임시 파일로 저장
    preped_file_path = data_dir / 'processed' / 'preprocessed_naver_review.csv'
    df.to_csv(preped_file_path, index=False)
    logger.info(f"전처리된 데이터 저장 완료: {preped_file_path}")
    
    # 학습 파라미터 설정
    train_args = {
        'learning_rate': config.train['learning_rate'],
        'num_train_epochs': config.train['num_train_epochs'],
        'batch_size': config.train['batch_size'],
        'sampling_rate': 1.0  # 이미 샘플링된 데이터이므로 1.0으로 설정
    }
    
   
    
    # 모델 디렉토리 설정
    models_dir = base_dir.parent / 'models'
    pretrained_models_dir = models_dir / 'pretrained'
    fine_tuned_models_dir = models_dir / 'finetuned'
    
    # 각 모델 학습
    results = {}

    #model_name = config.model['models']
    models_config = {
        'KcELECTRA': 'beomi/KcELECTRA-base',
        'KcBERT': 'beomi/kcbert-base'
    }
    
    #pretrained_model_name = models_config[model_name]
    for model_name, pretrained_model_name in models_config.items():
        #config.model['models'] = model_name
        if config.model['models'] == {model_name: pretrained_model_name}:
            print(f'{model_name} 모델 학습 시작')
            base_model_dir = fine_tuned_models_dir / model_name
            pretrained_model_dir = pretrained_models_dir / model_name
            
            # 모델 디렉토리 생성
            base_model_dir.mkdir(parents=True, exist_ok=True)
            pretrained_model_dir.mkdir(parents=True, exist_ok=True)

            
            run_id, metrics = trainer.train_model(
                model_name=model_name,
                pretrained_model_name=pretrained_model_name,
                dataset_name=config.data['name'],
                data_file=preped_file_path,
                base_model_dir=base_model_dir,
                pretrained_model_dir=pretrained_model_dir,
                train_args=train_args,
                num_unfrozen_layers=num_unfrozen_layers
            )
            
            results[model_name] = {'run_id': run_id, 'metrics': metrics}
        else:
            print(f'{model_name} 모델 학습 pass..')
    # 결과 요약 출력
    logger.info("\n=== 전체 학습 결과 요약 ===")
    for model_name, result in results.items():
        logger.info(f"\n{model_name}:")
        logger.info(f"Run ID: {result['run_id']}")
        logger.info(f"F1 Score: {result['metrics']['f1']:.4f}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
