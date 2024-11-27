from pathlib import Path
import mlflow
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
from contextlib import contextmanager

@dataclass
class MLflowConfig:
    """MLflow 설정을 위한 데이터 클래스"""
    tracking_uri: str = "http://127.0.0.1:5000"
    experiment_name: str = "sentiment_analysis"
    model_registry_stage: str = "Production"

class MLflowLogger:
    """MLflow 로깅을 위한 유틸리티 클래스"""
    
    def __init__(self, config: MLflowConfig):
        self.config = config
        mlflow.set_tracking_uri(config.tracking_uri)
        self.active_run = None
        
    def setup_experiment(self, experiment_name: Optional[str] = None) -> None:
        """MLflow 실험 설정"""
        exp_name = experiment_name or self.config.experiment_name
        mlflow.set_experiment(exp_name)
    
    @contextmanager
    def start_run(self, run_name: Optional[str] = None, nested: bool = False):
        """MLflow 실행을 위한 컨텍스트 매니저"""
        active_run = mlflow.active_run()
        
        if active_run and not nested:
            mlflow.end_run()  # 활성 run이 있고 nested가 아니면 종료
            
        run = mlflow.start_run(run_name=run_name, nested=nested)
        self.active_run = run
        
        try:
            yield run
        finally:
            if not nested:  # nested run이 아닌 경우에만 자동으로 종료
                mlflow.end_run()
                self.active_run = None
    
    def end_run(self):
        """현재 실행 중인 run 종료"""
        if mlflow.active_run():
            mlflow.end_run()
            self.active_run = None
    
    def log_model_artifacts(self, 
                          model: Any,
                          tokenizer: Any,
                          model_name: str,
                          dataset_name: str,
                          task: str = "sentiment-analysis") -> None:
        """모델 아티팩트 로깅"""
        mlflow.transformers.log_model(
            transformers_model={
                "model": model,
                "tokenizer": tokenizer
            },
            artifact_path="model",
            task=task,
            registered_model_name=f"{model_name}_{dataset_name}",
            metadata={
                "source_model": model_name,
                "task": task,
                "language": "ko",
                "dataset": dataset_name,
                "framework": "pytorch"
            }
        )
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """메트릭 로깅"""
        mlflow.log_metrics(metrics)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """파라미터 로깅"""
        # 모든 값을 문자열로 변환
        str_params = {k: str(v) for k, v in params.items()}
        mlflow.log_params(str_params)
    
    def log_artifacts(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """아티팩트 로깅"""
        mlflow.log_artifact(local_path, artifact_path)
    
    def log_dataset(self, 
                    df: Any,
                    context: str,
                    dataset_name: str,
                    tags: Optional[Dict[str, str]] = None) -> None:
        """데이터셋 로깅"""
        # 기본 태그 설정
        default_tags = {
            "format": "pandas",
            "dataset_name": dataset_name
        }
        
        # 추가 태그가 있으면 모든 값을 문자열로 변환
        if tags:
            tags = {k: str(v) for k, v in tags.items()}
            default_tags.update(tags)
        
        mlflow.log_input(
            dataset=mlflow.data.from_pandas(df),
            context=context,
            tags=default_tags
        )
    
    @staticmethod
    def generate_run_name(model_name: str, prefix: str = "") -> str:
        """실행 이름 생성"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{prefix}_{model_name}_{timestamp}".strip('_')