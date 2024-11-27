# main.py
from utils.twitter_collector import TwitterCollector
from utils.data_preprocessor import DataPreprocessor
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from models.model_factory import ModelFactory
import optuna
from datetime import datetime
import logging
from utils.mlflow_utils import MLflowLogger, MLflowConfig
from models.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

exp_name = "model_train_optuna"  # 실험 이름
# dataset
data_name = 'naver_movie_review'
sampling_rate = 0.001
model_register_threshold = 0.6
#Optuna의 기본 최적화 알고리즘인 Tree-structured Parzen Estimators (TPE)를 사용하고 있습니다. TPE는 베이지안 최적화의 한 종류입
sampler_type = "tpe"
n_trials = 2

class ModelTrainer:
    """모델 학습을 위한 클래스"""
    
    def __init__(self, mlflow_logger: MLflowLogger):
        self.mlflow_logger = mlflow_logger
        self.registry = ModelRegistry(metric_threshold={'eval_f1': model_register_threshold})
    
    def objective(self, 
                 trial: optuna.Trial, 
                 model_name: str, 
                 data_file: Path, 
                 model_dir: Path,
                 pretrained_model_name: str, 
                 pretrained_model_dir: Path) -> float:
        """Optuna 최적화를 위한 목적 함수"""
        # Define hyperparameters to optimize
        train_args = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'num_train_epochs': trial.suggest_int('num_train_epochs', 2, 5, 10),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32])
        }
        num_unfrozen_layers = trial.suggest_int('num_unfrozen_layers', 1, 4)
        
        # Create and train model
        factory = ModelFactory()
        model = factory.get_model(
            model_name, 
            data_file=data_file, 
            model_dir=model_dir,
            pretrained_model_name=pretrained_model_name,
            pretrained_model_dir=pretrained_model_dir
        )
        
        model.train(train_args=train_args, num_unfrozen_layers=num_unfrozen_layers)
        metrics = model.evaluate()
        
        return metrics['eval_f1']

    def create_study(self, study_name: str, sampler_type: str = "tpe") -> optuna.Study:
        """Optuna study 생성
        
        Args:
            study_name: study 이름
            sampler_type: 'tpe', 'random', 'grid', 'cmaes' 중 하나
        """
        if sampler_type == "tpe":
            sampler = optuna.samplers.TPESampler(seed=42)
        elif sampler_type == "random":
            sampler = optuna.samplers.RandomSampler(seed=42)
        elif sampler_type == "grid":
            sampler = optuna.samplers.GridSampler(
                {
                    'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
                    'num_train_epochs': [2, 3, 4, 5],
                    'batch_size': [16, 32],
                    'num_unfrozen_layers': [1, 2, 3, 4]
                }
            )
        elif sampler_type == "cmaes":
            sampler = optuna.samplers.CmaEsSampler(seed=42)
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")
        
        return optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler
        )

    def train_model(self, 
                   model_name: str,
                   pretrained_model_name: str,
                   dataset_name: str,
                   data_file: Path,
                   base_model_dir: Path,
                   pretrained_model_dir: Path,
                   n_trials: int = 2,
                   sampler_type: str = "tpe") -> Tuple[str, Dict[str, float]]:
        """모델 학습 및 평가 수행"""
        logger.info(f"\n=== 모델 학습 시작: {model_name} ===")
        
        # Optuna study 생성
        study_name = f"{model_name}_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = self.create_study(study_name, sampler_type)
        
        # 하이퍼파라미터 최적화
        study.optimize(
            lambda trial: self.objective(
                trial, 
                model_name, 
                data_file, 
                base_model_dir, 
                pretrained_model_name,
                pretrained_model_dir
            ), 
            n_trials=n_trials
        )
        
        # MLflow에 하이퍼파라미터 탐색 과정 로깅
        with self.mlflow_logger.start_run(
            run_name=self.mlflow_logger.generate_run_name(model_name, "hyperparameter_optimization")
        ) as run:
            # 기본 정보 로깅
            self.mlflow_logger.log_params({
                "sampler_type": sampler_type,
                "n_trials": n_trials,
                "study_name": study_name,
                "optimization_direction": study.direction.name,
                "search_space": {
                    "learning_rate": "log-uniform(1e-5, 1e-3)",
                    "num_train_epochs": "int(2, 5)",
                    "batch_size": "categorical[16, 32]",
                    "num_unfrozen_layers": "int(1, 4)"
                }
            })
            
            # 최적화 결과 로깅
            self.mlflow_logger.log_metrics({
                "best_value": study.best_value,
                "best_trial_number": study.best_trial.number,
                "total_trials": len(study.trials),
                "completed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                "failed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
            })
            
            # 모든 trial의 결과 로깅
            trial_results = []
            for trial in study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    result = {
                        "number": trial.number,
                        "value": trial.value,
                        **trial.params
                    }
                    trial_results.append(result)
            
            # Trial 결과를 DataFrame으로 변환하여 로깅
            if trial_results:
                trials_df = pd.DataFrame(trial_results)
                trials_df.to_csv("hyperparameter_trials.csv", index=False)
                self.mlflow_logger.log_artifacts("hyperparameter_trials.csv", "hyperparameter_optimization")
            
            # Best trial 정보 로깅
            self.mlflow_logger.log_params({
                "best_trial_params": study.best_trial.params,
                "best_trial_datetime": study.best_trial.datetime_start.strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # Optuna 시각화 저장 및 로깅
            try:
                import optuna.visualization as vis
                
                # 최적화 히스토리
                fig = vis.plot_optimization_history(study)
                fig.write_html("optimization_history.html")
                self.mlflow_logger.log_artifacts("optimization_history.html", "hyperparameter_optimization")
                
                # 파라미터 중요도
                fig = vis.plot_param_importances(study)
                fig.write_html("param_importances.html")
                self.mlflow_logger.log_artifacts("param_importances.html", "hyperparameter_optimization")
                
                # 파라미터 관계
                fig = vis.plot_parallel_coordinate(study)
                fig.write_html("parallel_coordinate.html")
                self.mlflow_logger.log_artifacts("parallel_coordinate.html", "hyperparameter_optimization")
                
                # 슬라이스 플롯
                fig = vis.plot_slice(study)
                fig.write_html("slice_plot.html")
                self.mlflow_logger.log_artifacts("slice_plot.html", "hyperparameter_optimization")
                
            except Exception as e:
                logger.warning(f"Optuna visualization failed: {str(e)}")
        
        # 최적의 하이퍼파라미터로 최종 모델 학습
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = self.mlflow_logger.generate_run_name(model_name, "training")
        
        with self.mlflow_logger.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            model_dir = base_model_dir / run_id
            model_dir.mkdir(parents=True, exist_ok=True)
            
            factory = ModelFactory()
            model = factory.get_model(
                model_name, 
                data_file=data_file, 
                model_dir=model_dir,
                pretrained_model_name=pretrained_model_name,
                pretrained_model_dir=pretrained_model_dir
            )
            
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
            self.mlflow_logger.log_params({
                **study.best_params,
                'model_name': model_name,
                'dataset_name': dataset_name,
                'run_id': run_id,
                'timestamp': timestamp
            })
            
            self.mlflow_logger.log_metrics({
                'validation_loss': metrics['eval_loss'],
                'validation_accuracy': metrics['eval_accuracy'],
                'validation_precision': metrics['eval_precision'],
                'validation_recall': metrics['eval_recall'],
                'validation_f1': metrics['eval_f1']
            })
            
            # 모델 저장
            save_path = model_dir / 'best_model'
            model.save_model(save_path)
            
            # MLflow에 모델 등록
            self.mlflow_logger.log_model_artifacts(
                model=model.model,
                tokenizer=model.tokenizer,
                model_name=model_name,
                dataset_name=dataset_name
            )
            
            # 데이터셋 정보 로깅
            train_df = model.train_dataset.to_pandas()
            val_df = model.val_dataset.to_pandas()
            
            self.mlflow_logger.log_dataset(
                train_df,
                context="training",
                dataset_name=dataset_name
            )
            self.mlflow_logger.log_dataset(
                val_df,
                context="validation",
                dataset_name=dataset_name
            )
            
            # 모델 평가 및 등록
            self.registry.evaluate_and_register(run_id, model_name)
            
            logger.info(f"\n=== 학습 완료 ===")
            logger.info(f"Run ID: {run_id}")
            logger.info(f"Model path: {model_dir}")
            logger.info(f"Best parameters: {study.best_params}")
            logger.info(f"Validation F1: {metrics['eval_f1']:.4f}")
            
            return run_id, metrics

def main():
    # MLflow 설정
    
    mlflow_config = MLflowConfig(experiment_name=exp_name)
    mlflow_logger = MLflowLogger(mlflow_config)
    mlflow_logger.setup_experiment()  # 실험 설정 명시적 호출
    
    trainer = ModelTrainer(mlflow_logger)
    
    # 데이터 준비
    
    
    
    base_dir = Path(__file__).parent
    data_dir = base_dir.parent / 'data'
    
    data_prep = DataPreprocessor(data_dir=data_dir)
    if data_name == 'naver_movie_review':
        data_prep.prep_naver_data(sampling_rate=sampling_rate)
        preped_file_path = data_prep.preprocess()
    else:
        logger.error('error. no data name')
        return
    
    # 모델 설정
    models_config = {
        'KcELECTRA': 'beomi/KcELECTRA-base',
        'KcBERT': 'beomi/kcbert-base'
    }
    
    # 모델 디렉토리 설정
    models_dir = base_dir.parent / 'models'
    pretrained_models_dir = models_dir / 'pretrained'
    fine_tuned_models_dir = models_dir / 'finetuned'
    models_dir.mkdir(parents=True, exist_ok=True)
    pretrained_models_dir.mkdir(parents=True, exist_ok=True)
    fine_tuned_models_dir.mkdir(parents=True, exist_ok=True)
    
    # 각 모델 학습
    results = {}
    total_models = len(models_config)
    for idx, (model_name, pretrained_model_name) in enumerate(models_config.items(), 1):
        logger.info(f"\n=== Training {model_name} ({idx}/{total_models}) ===")
        
        base_model_dir = fine_tuned_models_dir / model_name
        base_model_dir.mkdir(parents=True, exist_ok=True)
        
        pretrained_model_dir = pretrained_models_dir / model_name
        pretrained_model_dir.mkdir(parents=True, exist_ok=True)
        
        run_id, metrics = trainer.train_model(
            model_name=model_name,
            pretrained_model_name=pretrained_model_name,
            dataset_name=data_name,
            data_file=preped_file_path,
            base_model_dir=base_model_dir,
            pretrained_model_dir=pretrained_model_dir,
            n_trials=n_trials,
            sampler_type=sampler_type

        )
        
        results[model_name] = {'run_id': run_id, 'metrics': metrics}
    
    # 결과 요약 출력
    logger.info("\n=== 전체 학습 결과 요약 ===")
    for model_name, result in results.items():
        logger.info(f"\n{model_name}:")
        logger.info(f"Run ID: {result['run_id']}")
        logger.info(f"F1 Score: {result['metrics']['eval_f1']:.4f}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
