# main.py
from utils.twitter_collector import TwitterCollector
from utils.data_preprocessor import DataPreprocessor
from pathlib import Path
from typing import List
import pandas as pd
from models.model_factory import ModelFactory
import mlflow
import optuna
from typing import Dict
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # GUI 백엔드 비활성화
from models.model_registry import ModelRegistry

def objective(trial: optuna.Trial, model_name: str, data_file: Path, model_dir: Path, 
             pretrained_model_name: str, pretrained_model_dir: Path) -> float:
    # Define hyperparameters to optimize
    train_args = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'num_train_epochs': trial.suggest_int('num_train_epochs', 2, 5),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32])
    }
    num_unfrozen_layers = trial.suggest_int('num_unfrozen_layers', 1, 4)
    
    # Create and train model
    factory = ModelFactory()
    model = factory.get_model(model_name, 
                            data_file=data_file, 
                            model_dir=model_dir,
                            pretrained_model_name=pretrained_model_name,
                            pretrained_model_dir=pretrained_model_dir)
    
    model.train(train_args=train_args, num_unfrozen_layers=num_unfrozen_layers)
    metrics = model.evaluate()
    
    return metrics['eval_f1']

def main():
    # 1. 데이터 수집
    # collector = TwitterCollector(api_key='YOUR_API_KEY',
    #                              api_secret='YOUR_API_SECRET',
    #                              access_token='YOUR_ACCESS_TOKEN',
    #                              access_secret='YOUR_ACCESS_SECRET')
    # collector.collect_tweets(query='감정 OR 기분 OR 행복 OR 슬픔', max_tweets=1000,
    #                          output_file='data/raw/tweets.csv')

    exp_name = 'test_241126_1'

    data_name ='naver_movie'
    models = ['KcELECTRA', 'KcBERT']

    mlflow_uri = 'http://127.0.0.1:5000'
    
    base_dir = Path(__file__).parent
    data_dir = base_dir.parent / 'data' 
    
    
    if data_name == 'naver_movie':
        data_prep= DataPreprocessor(data_dir=data_dir)
        data_prep.prep_naver_data(sampling_rate=0.001)
        preped_file_path = data_prep.preprocess()
    else:
        print('error. no data name')
    # 3. 모델 학습 및 평가
    #  # 모델 이름에 따라 동적으로 모델 생성

    # model_dir = Path(__file__).parent / 'models' #Path('e:/models')
    # #model_kcbert = model_dir / 'KcBERT'
    # model_dir = model_dir / model_name
    # 평가할 모델 리스트
    
    # MLflow 설정
    print(f'\n<<< MLflow Experiment: URI: {mlflow_uri}, Exp: {exp_name}')
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(exp_name)
    
    # pretrained 모델을 저장할 공통 디렉토리 설정
    models_dir = Path(__file__).parent.parent / 'models'
    pretrained_models_dir = models_dir / 'pretrained'
    fine_tuned_models_dir = models_dir / 'finetuned'
    models_dir.mkdir(parents=True, exist_ok=True)
    pretrained_models_dir.mkdir(parents=True, exist_ok=True)
    
    total_models = len(models)
    for idx, model_name in enumerate(models, 1):
        print(f"\n=== Training {model_name} ({idx}/{total_models}) ===")
        base_model_dir = fine_tuned_models_dir / model_name
        base_model_dir.mkdir(parents=True, exist_ok=True)
        pretrained_model_dir = pretrained_models_dir / model_name
        pretrained_model_dir.mkdir(parents=True, exist_ok=True)
        
        # pretrained model 설정
        pretrained_model_name = {
            'KcELECTRA': 'beomi/KcELECTRA-base',
            'KcBERT': 'beomi/kcbert-base'
        }[model_name]
        
        # pretrained model 경로 설정
        # Optuna study 생성
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(
                trial, 
                model_name, 
                preped_file_path, 
                base_model_dir, 
                pretrained_model_name,
                pretrained_model_dir
            ), 
            n_trials=2
        )
        
        # 최적의 하이퍼파라미터로 최종 모델 학습
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset_name = preped_file_path.stem
        
        # 기존 run이 있다면 종료
        mlflow.end_run()
        
        # 새로운 run 시작
        with mlflow.start_run(run_name=f"{model_name}_{dataset_name}_{timestamp}") as run:
            run_id = run.info.run_id
            model_dir = base_model_dir / run_id  # fine-tuned 모델 저장 경로
            model_dir.mkdir(parents=True, exist_ok=True)
            
            factory = ModelFactory()
            model = factory.get_model(model_name, 
                                    data_file=preped_file_path, 
                                    model_dir=model_dir,
                                    pretrained_model_name=pretrained_model_name,
                                    pretrained_model_dir=pretrained_model_dir)  # pretrained 모델 경로 전달
            
            # TrainingArguments에서 output_dir 설정 제거 (랜덤 이름 방지)
            best_params = {
                'train_args': {
                    'learning_rate': study.best_params['learning_rate'],
                    'num_train_epochs': study.best_params['num_train_epochs'],
                    'batch_size': study.best_params['batch_size']
                },
                'num_unfrozen_layers': study.best_params['num_unfrozen_layers']
            }
            
            model.train(**best_params)
            metrics = model.evaluate()
            
            # MLflow에 로깅
            mlflow.log_params({
                **study.best_params,
                'model_name': model_name,
                'dataset_name': dataset_name,
                'run_id': run_id,
                'timestamp': timestamp
            })
            mlflow.log_metrics({
                'validation_loss': metrics['eval_loss'],
                'validation_accuracy': metrics['eval_accuracy'],
                'validation_precision': metrics['eval_precision'],
                'validation_recall': metrics['eval_recall'],
                'validation_f1': metrics['eval_f1']
            })
            
            # 모델 저장
            save_path = model_dir / 'best_model'
            model.save_model(save_path)
            
            # MLflow에 모델 등록 (모델 카드 정보 포함) info from huggingface
            mlflow.transformers.log_model(
                transformers_model={
                    "model": model.model,
                    "tokenizer": model.tokenizer
                },
                artifact_path="model",
                task="sentiment-analysis",
                registered_model_name=f"{model_name}_{dataset_name}",
                metadata={
                    "source_model": pretrained_model_name,
                    "task": "sentiment-analysis",
                    "language": "ko",
                    "dataset": dataset_name,
                    "framework": "pytorch"
                }
            )
            
            # 데이터셋 정보 로깅
            train_df = model.train_dataset.to_pandas()
            val_df = model.val_dataset.to_pandas()
            
            mlflow.log_input(
                dataset=mlflow.data.from_pandas(train_df),
                context="training",
                tags={"format": "pandas", "dataset_name": dataset_name}
            )
            mlflow.log_input(
                dataset=mlflow.data.from_pandas(val_df),
                context="validation",
                tags={"format": "pandas", "dataset_name": dataset_name}
            )
            
            print(f"\n=== 실험 정보 ===")
            print(f"Run ID: {run_id}")
            print(f"Model path: {model_dir}")
            print(f"Best parameters: {study.best_params}")
            print(f"Validation F1: {metrics['eval_f1']:.4f}")
            
            # 모델 평가 및 등록
            registry = ModelRegistry(metric_threshold={'eval_f1': 0.85}, )
            registry.evaluate_and_register(run.info.run_id, model_name)

    # 마지막 run 종료
    mlflow.end_run()

if __name__ == '__main__':
    main()
