from utils.data_preprocessor import DataPreprocessor
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from models.model_factory import ModelFactory
import optuna
from datetime import datetime
import logging
from utils.mlflow_utils import MLflowLogger
from models.model_registry import ModelRegistry
import os
from utils.config import Config
from models.model_factory import ModelFactory
import matplotlib.pyplot as plt
import mlflow


config = Config()

# Optuna 설정 사용
n_trials = config.optuna['n_trials']
timeout = config.optuna['timeout']
search_space = config.optuna['search_space']
# config.mlflow['experiment_name'] += '_' + Path(__file__).stem + '_000'
# exp_name = config.mlflow['experiment_name']
sampling_rate = config.data['sampling_rate']
logger = logging.getLogger(__name__)
sampler_type = config.optuna['optimizer']
class HyperparamOptimizer:
    """하이퍼파라미터 최적화를 위한 클래스"""
    
    def __init__(self, mlflow_logger: MLflowLogger):
        self.mlflow_logger = mlflow_logger
        
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna optimization objective function"""
        # 하이퍼파라미터 샘플링
        train_args = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'num_train_epochs': trial.suggest_int('num_train_epochs', 2, 5),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'num_unfrozen_layers': trial.suggest_int('num_unfrozen_layers', 1, 4),
            'sampling_rate': 1.0  # 이미 샘플링된 데이터이므로 1.0으로 설정
        }
        num_unfrozen_layers = trial.suggest_int('num_unfrozen_layers', 1, 4)
        
        # 모델 생성 및 학습
        factory = ModelFactory()
        model = factory.get_model(
            model_name=self.model_name,
            data_file=self.data_file,
            model_dir=self.model_dir,
            pretrained_model_name=self.pretrained_model_name,
            pretrained_model_dir=self.pretrained_model_dir
        )
        
        model.set_mlflow_logger(self.mlflow_logger)
        model.train(train_args=train_args, num_unfrozen_layers=num_unfrozen_layers)
        metrics = model.evaluate()
        
        # 예측 및 혼동 행렬 생성
        predictions = model.predict(model.val_dataset)
        confusion_matrix_path = model.save_confusion_matrix(
            y_true=model.val_dataset.labels,
            y_pred=predictions
        )
        
        # MLflow 로깅
        self.mlflow_logger.log_evaluate(
            metrics=metrics,
            model=model.model,
            tokenizer=model.tokenizer,
            train_data=model.train_dataset,
            val_data=model.val_dataset,
            model_name=self.model_name,
            dataset_name=self.dataset_name,
            sampling_rate=self.sampling_rate,
            confusion_matrix_path=confusion_matrix_path
        )
        
        # 모델 레지스트리에 등록 시도
        if metrics['f1'] > config.model['register_threshold']:
            registry = ModelRegistry()
            registry.add_model(
                model_name=self.model_name,
                run_id=mlflow.active_run().info.run_id,
                metrics=metrics,
                dataset_name=self.dataset_name,
                sampling_rate=self.sampling_rate,
                threshold=config.model['register_threshold']
            )
        
        return metrics['f1']

    def create_study(self, sampler_type: str = "tpe") -> optuna.Study:
        """Optuna study 생성"""
        if sampler_type == "tpe":
            sampler = optuna.samplers.TPESampler(seed=42)
        elif sampler_type == "random":
            sampler = optuna.samplers.RandomSampler(seed=42)
        elif sampler_type == "grid":
            sampler = optuna.samplers.GridSampler(
                {
                    'learning_rate': [1e-5,  1e-4, 1e-3],
                    'num_train_epochs': [2, 4, 6],
                    'batch_size': [16, 32],
                    'num_unfrozen_layers': [1, 3, 6]
                }
            )
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")
        
        return optuna.create_study(
            study_name=f"hyperparam_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            direction="maximize",
            sampler=sampler
        )

    def optimize(self, 
                model_name: str,
                pretrained_model_name: str,
                dataset_name: str,
                data_file: Path,
                base_model_dir: Path,
                pretrained_model_dir: Path,
                n_trials: int = 10,
                sampler_type: str = "tpe",
                sampling_rate: float = 0.001) -> Tuple[Dict, float]:
        """하이퍼파라미터 최적화 수행"""
        
        # 데이터 준비
        data_prep = DataPreprocessor(data_dir=data_file.parent.parent)
        
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
        preped_file_path = data_file.parent / 'preprocessed_naver_review.csv'
        df.to_csv(preped_file_path, index=False)
        logger.info(f"전처리된 데이터 저장 완료: {preped_file_path}")
        
        # 클래스 속성으로 저장
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.data_file = preped_file_path
        self.model_dir = base_model_dir
        self.pretrained_model_name = pretrained_model_name
        self.pretrained_model_dir = pretrained_model_dir
        self.sampling_rate = 1.0  # 이미 샘플링된 데이터이므로 1.0으로 설정
        
        study = self.create_study(sampler_type)
        
        # MLflow 실행 이름 생성 - 단순화
        run_name = f"{model_name}_hpo_{datetime.now().strftime('%Y%m%d')}"
        
        with self.mlflow_logger.run_with_logging(
            "hyperparameter_optimization",
            model_name,
            dataset_name,
            sampling_rate,
            run_name=run_name
        ) as run:
            # 실험 설정 로깅
            self.mlflow_logger.log_params({
                "model_name": model_name,
                "pretrained_model_name": pretrained_model_name,
                "dataset_name": dataset_name,
                "data_file": str(preped_file_path),
                "n_trials": n_trials,
                "sampler_type": sampler_type,
                "dataset_size": len(df),
                "num_labels": len(df['label'].unique()),
                "label_distribution": str(df['label'].value_counts().to_dict())
            })
            
            study.optimize(lambda trial: self.objective(trial), n_trials=n_trials)
            
            # Optuna 결과 로깅
            self.mlflow_logger.log_params({
                "best_trial_params": study.best_trial.params,
                "optimization_direction": study.direction.name
            })
            
            self.mlflow_logger.log_metrics({
                "best_value": study.best_value,
                "best_trial_number": study.best_trial.number
            })
            
            # 플롯 기능 임시 비활성화
            # self._plot_optimization_history(study)
            # self._plot_param_importances(study)
            
        return study.best_params, study.best_value

    def _plot_optimization_history(self, study):
        """최적화 히스토리 플롯 저장"""
        try:
            # matplotlib을 사용한 플롯 생성
            plt.figure(figsize=(10, 6))
            trials = study.trials
            values = [t.value if t.value is not None else float('nan') for t in trials]
            plt.plot(range(len(trials)), values, marker='o')
            plt.xlabel('Trial number')
            plt.ylabel('Objective value (f1)')
            plt.title('Optimization History')
            
            # 파일 저장
            plt.savefig("optimization_history.png")
            plt.close()
            
            self.mlflow_logger.log_artifacts("optimization_history.png")
            os.remove("optimization_history.png")
        except Exception as e:
            logger.error(f"최적화 히스토리 플롯 저장 중 오류: {str(e)}")

    def _plot_param_importances(self, study):
        """파라미터 중요도 플롯 저장"""
        try:
            # 파라미터 중요도 계산
            importances = optuna.importance.get_param_importances(study)
            
            # matplotlib을 사용한 플롯 생성
            plt.figure(figsize=(10, 6))
            params = list(importances.keys())
            values = list(importances.values())
            
            # 가로 막대 그래프
            plt.barh(range(len(params)), values)
            plt.yticks(range(len(params)), params)
            plt.xlabel('Importance')
            plt.title('Parameter Importances')
            
            # 파일 저장
            plt.savefig("param_importances.png")
            plt.close()
            
            self.mlflow_logger.log_artifacts("param_importances.png")
            os.remove("param_importances.png")
        except Exception as e:
            logger.error(f"파라미터 중요도 플롯 저장  오류: {str(e)}")

def main():
    # MLflow 설정
    file_path = Path(__file__)
    file_name = file_path.stem
    config.mlflow['experiment_name'] = config.mlflow['experiment_name'] + '_' + file_name
    print(f'Experiment name: {config.mlflow["experiment_name"]}')
    
    mlflow_logger = MLflowLogger()
    
    # HyperparamOptimizer 인스턴스 생성
    optimizer = HyperparamOptimizer(mlflow_logger)
    
    # 데이터 준비
    base_dir = Path(__file__).parent
    data_dir = base_dir.parent / 'data'
    
    data_prep = DataPreprocessor(data_dir=data_dir)
    data_prep.prep_naver_data(sampling_rate=0.001)
    preped_file_path = data_prep.preprocess()
    
    # 모델 설정
    models_config = {
        'KcELECTRA': 'beomi/KcELECTRA-base',
        'KcBERT': 'beomi/kcbert-base'
    }
    
    # 모델 디렉토리 설정
    models_dir = base_dir.parent / 'models'
    pretrained_models_dir = models_dir / 'pretrained'
    fine_tuned_models_dir = models_dir / 'finetuned'
    
    # 각 모델에 대해 하이퍼파라미터 최적
    results = {}
    for model_name, pretrained_model_name in models_config.items():
        logger.info(f"\n=== 하이퍼파라미터 최적화 시작: {model_name} ===")
        
        base_model_dir = fine_tuned_models_dir / model_name
        pretrained_model_dir = pretrained_models_dir / model_name
        
        best_params, best_value = optimizer.optimize(
            model_name=model_name,
            pretrained_model_name=pretrained_model_name,
            dataset_name='naver_movie_review',
            data_file=preped_file_path,
            base_model_dir=base_model_dir,
            pretrained_model_dir=pretrained_model_dir,
            n_trials=n_trials,
            sampler_type=sampler_type,
            sampling_rate=sampling_rate
        )
        
        results[model_name] = {
            'best_params': best_params,
            'best_f1': best_value
        }
    
    # 결과 요약 출력
    logger.info("\n=== 하이퍼파라미터 최적화 결과 요약 ===")
    for model_name, result in results.items():
        logger.info(f"\n{model_name}:")
        logger.info(f"Best F1: {result['best_f1']:.4f}")
        logger.info("Best Parameters:")
        for param, value in result['best_params'].items():
            logger.info(f"- {param}: {value}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main() 