from pathlib import Path
import mlflow
from datetime import datetime
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
import pandas as pd
from contextlib import contextmanager
import os
import logging
import json
import torch
from utils.config import Config
from mlflow.tracking import MlflowClient
import shutil

logger = logging.getLogger(__name__)

class MLflowLogger:
    """MLflow 로깅을 위한 유틸리티 클래스"""
    
    def __init__(self):
        self.config = Config()
        self.client = MlflowClient()
        self.experiment_id = self._setup_mlflow()
        
    def _setup_mlflow(self):
        """MLflow 실험 설정"""
        mlflow.set_tracking_uri(self.config.mlflow['tracking_uri'])
        if 'registry_uri' in self.config.mlflow:
            mlflow.set_registry_uri(self.config.mlflow['registry_uri'])
        
        # 실험 생성 또는 가져오기
        experiment = mlflow.get_experiment_by_name(self.config.mlflow['experiment_name'])
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                name=self.config.mlflow['experiment_name'],
                artifact_location=self.config.mlflow['artifact_location']
            )
        else:
            experiment_id = experiment.experiment_id
        
        mlflow.set_experiment(experiment_id=experiment_id)
        logger.info(f"MLflow 실험 설정 완료: {self.config.mlflow['experiment_name']} (ID: {experiment_id})")
        return experiment_id
    
    def find_run_in_all_experiments(self, run_id: str):
        """모든 실험에서 특정 run_id 검색"""
        logger.info(f"\n=== 모든 실험에서 Run ID 검색: {run_id} ===")
        experiments = self.client.search_experiments()
        
        for exp in experiments:
            logger.info(f"\n실험 검색 중: {exp.name} (ID: {exp.experiment_id})")
            runs = self.client.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string=f"attributes.run_id = '{run_id}'"
            )
            if runs:
                found_run = runs[0]
                logger.info(f"Run을 찾았습니다:")
                logger.info(f"- 실험명: {exp.name}")
                logger.info(f"- 실험 ID: {exp.experiment_id}")
                logger.info(f"- Run 상태: {found_run.info.status}")
                logger.info(f"- 아티팩트 URI: {found_run.info.artifact_uri}")
                return found_run, exp.experiment_id
                
        logger.error(f"어떤 실험에서도 Run ID {run_id}를 찾을 수 없습니다.")
        return None, None
    
    def get_model_path(self, run_id: str, model_path: str = "model"):
        """특정 run의 모델 경로 반환"""
        run, experiment_id = self.find_run_in_all_experiments(run_id)
        if run is None:
            return None
            
        artifact_uri = run.info.artifact_uri
        return f"{artifact_uri}/{model_path}"
    
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
        """모델과 토크나이저를 MLflow에 저장"""
        model_version = mlflow.transformers.log_model(
            transformers_model={
                "model": model,
                "tokenizer": tokenizer
            },
            artifact_path="model",
            task=task,
            registered_model_name=model_name,
            metadata={
                "source_model": model_name,
                "task": task,
                "language": "ko",
                "dataset": dataset_name,
                "framework": "pytorch"
            }
        )
        
        # 모델 별칭 설정
        client = mlflow.tracking.MlflowClient()
        try:
            client.set_registered_model_alias(
                name=model_name,
                alias="latest",
                version=model_version.version
            )
        except Exception as e:
            logger.warning(f"모델 별칭 설정 중 오류 발생: {str(e)}")
    
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
                    df: pd.DataFrame,
                    context: str,
                    dataset_name: str,
                    sampling_rate: float,
                    split_type: str = "train",  # train/val/test
                    tags: Optional[Dict[str, str]] = None):
        """데이터셋 상세 정보 및 파일 로깅"""
        if tags is None:
            tags = {}
        
        # SentimentDataset을 DataFrame으로 변환
        if hasattr(df, 'to_pandas'):
            df = df.to_pandas()
        
        # 텍스트 컬럼 이름 확인
        text_column = 'text' if 'text' in df.columns else 'document'
        
        # 데이터셋 통계 정보 계산
        dataset_stats = {
            "total_samples": int(len(df)),
            "label_distribution": {k: int(v) for k, v in df['label'].value_counts().to_dict().items()},
            "text_length_stats": {
                "mean": float(df[text_column].str.len().mean()),
                "min": int(df[text_column].str.len().min()),
                "max": int(df[text_column].str.len().max())
            },
            "sampling_rate": sampling_rate,
            "split_type": split_type
        }
        
        # 데이터셋 메타데이터 저장
        metadata_path = f"{split_type}_dataset_info.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_stats, f, ensure_ascii=False, indent=2)
        self.log_artifacts(metadata_path, f"datasets/{split_type}/metadata")
        os.remove(metadata_path)
        
        # 실제 데이터셋 파일 저장 (CSV 형식)
        dataset_path = f"{split_type}_data.csv"
        df.to_csv(dataset_path, index=False)
        self.log_artifacts(dataset_path, f"datasets/{split_type}/data")
        os.remove(dataset_path)
        
        # MLflow Datasets API를 통한 로깅
        mlflow.log_input(
            dataset=mlflow.data.from_pandas(df),
            context=context,
            tags={
                "format": "pandas",
                "dataset_name": dataset_name,
                "split_type": split_type,
                "sampling_rate": str(sampling_rate),
                "total_samples": str(len(df)),
                "label_distribution": str(dataset_stats["label_distribution"]),
                **tags
            }
        )
    
    def generate_run_name(self, 
                     run_type: str, 
                     model_name: str, 
                     dataset_name: str, 
                     sampling_rate: float) -> str:
        """MLflow 실행 이름 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{run_type}_{model_name}_{dataset_name}_{sampling_rate}_{timestamp}"

    def log_evaluate(self, metrics: Dict[str, float], model, tokenizer, 
                    train_data: pd.DataFrame, val_data: pd.DataFrame,
                    model_name: str, dataset_name: str, sampling_rate: float,
                    confusion_matrix_path: Optional[str] = None):
        """평가 결과 로깅"""
        try:
            # 메트릭스 로깅
            self.log_metrics(metrics)
            
            # 데이터셋 정보 로깅 (메서드 이름 수정)
            self.log_dataset(train_data, val_data, dataset_name, sampling_rate)
            
            # 모델 아티팩트 로깅
            self.log_model_artifacts(model, tokenizer, model_name)
            
            # Confusion Matrix 로깅 (있는 경우에만)
            if confusion_matrix_path:
                try:
                    mlflow.log_artifacts(confusion_matrix_path, "confusion_matrices")
                    logger.info(f"Confusion Matrix 아티팩트가 로깅되었습니다: {confusion_matrix_path}")
                except Exception as e:
                    logger.warning(f"Confusion Matrix 로깅 중 오류 발생: {str(e)}")
                
        except Exception as e:
            logger.error(f"평가 결과 로깅 중 오류 발생: {str(e)}")
            logger.error(f"상세 오류: {type(e).__name__}")

    @contextmanager
    def run_with_logging(self, 
                        run_type: str, 
                        model_name: str, 
                        dataset_name: str, 
                        sampling_rate: float,
                        run_name: str = None):
        """MLflow 실행 컨텍스트 매니저"""
        if run_name is None:
            run_name = f"{model_name}_{run_type}_{datetime.now().strftime('%Y%m%d')}"
        
        with mlflow.start_run(run_name=run_name) as run:
            # 기본 정보 로깅
            self.log_params({
                "model_name": model_name,  # 단순화된 모델 이름
                "dataset_name": dataset_name,
                "sampling_rate": sampling_rate,
                "run_type": run_type
            })
            
            yield run
            
            logger.info(f"MLflow 실행 종료: {run.info.run_id}")

    def generate_model_name(self, 
                          model_name: str, 
                          task: str = "sentiment",
                          dataset_name: str = None,
                          version: str = None,
                          sampling_rate: float = None) -> str:
        """모델 레지스트리에 등록할 이름 생성
        
        Args:
            model_name: 기본 모델 이름 (예: KcBERT)
            
        Returns:
            str: 생성된 모델 이름 (예: "KcBERT")
        """
        # 단순화된 모델 이름만 반환
        return model_name

