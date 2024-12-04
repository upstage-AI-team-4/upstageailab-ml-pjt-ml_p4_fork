import os
import sys
from pathlib import Path
import torch
import mlflow
from typing import Optional

from src.config import Config
from src.data.nsmc_dataset import NSMCDataModule
from src.models.kcbert_model import KcBERT
from src.models.kcelectra_model import KcELECTRA
from src.utils.mlflow_utils import MLflowModelManager
import pytorch_lightning as pl

def train_model(config: Optional[Config] = None) -> None:
    """모델 학습 실행
    
    Args:
        config: 설정 객체 (없으면 새로 생성)
    """
    if config is None:
        config = Config()
    
    # MLflow 설정
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)
    
    # 데이터 모듈 준비
    data_module = NSMCDataModule(config=config)
    data_module.prepare_data()
    data_module.setup(stage='fit')
    
    # 모델 선택 및 초기화
    model_name = config.project.model_name
    model_config = config.models[model_name]
    
    if model_name == "KcBERT":
        model = KcBERT(
            pretrained_model=model_config.pretrained_model,
            num_labels=model_config.training.num_labels,
            num_unfreeze_layers=model_config.training.num_unfreeze_layers
        )
        # 학습 파라미터 설정
        model.lr = model_config.training.lr
        model.optimizer_name = model_config.training.optimizer
        model.scheduler_name = model_config.training.lr_scheduler
        
    elif model_name == "KcELECTRA":
        model = KcELECTRA(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Trainer 설정
    trainer = pl.Trainer(**config.common.trainer)
    
    # MLflow 실험 시작
    with mlflow.start_run() as run:
        # 하이퍼파라미터 로깅
        mlflow.log_params({
            "model_name": model_name,
            "pretrained_model": model_config.pretrained_model,
            "batch_size": model_config.training.batch_size,
            "learning_rate": model_config.training.lr,
            "epochs": model_config.training.epochs,
            "max_length": model_config.training.max_length,
            "optimizer": model_config.training.optimizer,
            "lr_scheduler": model_config.training.lr_scheduler,
            "num_unfreeze_layers": model_config.training.num_unfreeze_layers
        })
        
        # 모델 학습
        trainer.fit(model, data_module)
        
        # 검증 결과 로깅
        metrics = trainer.callback_metrics
        mlflow.log_metrics({
            "val_loss": metrics["val_loss"].item(),
            "val_accuracy": metrics["val_accuracy"].item(),
            "val_f1": metrics["val_f1"].item(),
            "val_precision": metrics["val_precision"].item(),
            "val_recall": metrics["val_recall"].item()
        })
        
        # 모델 아티팩트 저장
        mlflow.pytorch.log_model(
            model,
            "model",
            registered_model_name=model_name
        )
        
        # 모델 정보 저장
        model_manager = MLflowModelManager(config)
        model_manager.register_model(
            model_name=model_name,
            run_id=run.info.run_id
        )

if __name__ == "__main__":
    train_model() 