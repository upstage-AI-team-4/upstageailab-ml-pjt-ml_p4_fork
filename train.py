import os
import warnings
from pathlib import Path
import json
from datetime import datetime
import mlflow
import torch  # torch import 추가
from pytorch_lightning import Trainer, seed_everything
from transformers import AutoTokenizer
from src.utils.visualization import plot_confusion_matrix
from src.models.kcbert_model import KcBERT
from src.data.nsmc_dataset import NSMCDataModule
from src.utils.config import Config
from src.utils.mlflow_utils import MLflowModelManager
# torchvision 관련 경고 무시
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
# Tensor Core 최적화를 위한 precision 설정
torch.set_float32_matmul_precision('medium')  # 또는 'high'

def get_local_path_from_uri(uri: str) -> Path:
    """MLflow URI를 로컬 경로로 변환"""
    # file:/// 또는 file:// 또는 file: 제거
    if uri.startswith(('file:///', 'file://', 'file:')):
        # 운영체제 상관없이 처리
        path = uri.split('file:')[-1].lstrip('/')
        # Windows에서 드라이브 문자(예: C:) 처리
        if os.name == 'nt' and len(path) >= 2 and path[1] == ':':
            return Path(path)
        return Path('/' + path)
    return Path(uri)

def cleanup_artifacts(config, metrics, run_id):
    """체크포인트와 아티팩트 정리"""
    # 체크포인트 폴더 정리
    checkpoint_dir = config.base_path / config.base_training.checkpoint['dirpath']
    if checkpoint_dir.exists():
        import shutil
        shutil.rmtree(checkpoint_dir)  # 폴더 자체를 삭제
    
    # MLflow 아티팩트 정리
    if metrics["val_accuracy"] <= config.mlflow.model_registry_metric_threshold:
        # threshold를 넘지 못한 모델의 아티팩트는 삭제
        artifact_path = config.mlflow.mlrun_path / run_id / "artifacts"
        if artifact_path.exists():
            shutil.rmtree(artifact_path)

def train(config):
    # MLflow 설정
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    
    # 실험 설정
    experiment = mlflow.get_experiment_by_name(config.mlflow.experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            config.mlflow.experiment_name,
            artifact_location=f"file://{config.mlflow.mlrun_path}"
        )
    else:
        experiment_id = experiment.experiment_id
    
    mlflow.set_experiment(experiment_id)
    
    # 시드 설정
    seed_everything(config.base_training.random_seed)
    
    # Run 이름 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.project.model_name}_{config.project.dataset_name}_{timestamp}"
    
    # MLflow 실험 시작
    with mlflow.start_run(run_name=run_name) as run:
        print(f"\n=== Starting new run: {run_name} ===")
        
        # 토크나이저 설정
        tokenizer = AutoTokenizer.from_pretrained(config.base_training.pretrained_model)
        
        # 데이터 모듈 설정
        data_module = NSMCDataModule(
            tokenizer=tokenizer,
            batch_size=config.base_training.batch_size,
            max_length=config.base_training.max_length,
            sampling_rate=config.data.sampling_rate,
            config=config,
            data_dir=str(config.data_path),
            train_file=config.data.train_data_path,
            val_file=config.data.val_data_path
        )
        
        # 모이터 준비 및 설정
        data_module.prepare_data()  # 데이터 다운로드 등 준비 작업
        data_module.setup(stage='fit')  # 데이터셋 설정
        
        # 모델 설정
        model = KcBERT(**config.get_model_kwargs())
        
        # Trainer 설정 및 학습
        trainer = Trainer(**config.get_trainer_kwargs())
        trainer.fit(model, data_module)
        
        # 검증 성능 평가
        val_results = trainer.validate(model, datamodule=data_module)[0]
        
        # 예측값과 실제값 수집 (confusion matrix 용)
        val_predictions = []
        val_labels = []
        for batch in data_module.val_dataloader():
            with torch.no_grad():
                outputs = model(**{k: v for k, v in batch.items() if k != 'labels'})
                predictions = torch.argmax(outputs.logits, dim=-1)
                val_predictions.extend(predictions.cpu().numpy())
                val_labels.extend(batch['labels'].cpu().numpy())
        
        # Confusion matrix 생성 및 MLflow에 로깅
        cm_path = plot_confusion_matrix(val_labels, val_predictions)
        mlflow.log_artifact(cm_path, "confusion_matrix")
        os.remove(cm_path)  # 임시 파일 삭제
        
        # MLflow에 메트릭 로깅
        metrics = {
            "val_accuracy": val_results["val_accuracy"],
            "val_precision": val_results["val_precision"],
            "val_recall": val_results["val_recall"],
            "val_f1": val_results["val_f1"]
        }
        mlflow.log_metrics(metrics)
        
        # MLflow에 파라미터 로깅
        params = {
            "model_name": config.project.model_name,
            "pretrained_model": config.base_training.pretrained_model,
            "batch_size": config.base_training.batch_size,
            "learning_rate": config.base_training.lr,
            "num_epochs": config.base_training.epochs,
            "max_length": config.base_training.max_length,
            "optimizer": config.base_training.optimizer,
            "lr_scheduler": config.base_training.lr_scheduler,
            "num_unfreeze_layers": config.base_training.num_unfreeze_layers,
            "precision": config.base_training.precision
        }
        mlflow.log_params(params)
        
        # threshold를 넘는 모델만 아티팩트로 저장
        if metrics["val_accuracy"] > config.mlflow.model_registry_metric_threshold:
            # 모델 아티팩트 저장
            artifact_uri = mlflow.get_artifact_uri()
            artifact_path = get_local_path_from_uri(artifact_uri) / "model"
            os.makedirs(artifact_path, exist_ok=True)
            
            # 모델 저장
            torch.save(model.state_dict(), artifact_path / "model.pt")
            
            # 설정 저장
            config_dict = {
                "model_type": type(model).__name__,
                "pretrained_model": config.base_training.pretrained_model,
                "num_labels": config.base_training.num_labels,
                "max_length": config.base_training.max_length
            }
            with open(artifact_path / "config.json", "w") as f:
                json.dump(config_dict, f, indent=2)
            
            # MLflow에 모델 등록
            register_path = str(artifact_path).replace('\\', '/')
            mlflow.register_model(
                f"file://{register_path}",
                config.project.model_name
            )
            
            # model_info.json에 저장
            model_manager = MLflowModelManager(config)
            model_manager.save_model_info(
                run_id=run.info.run_id,
                metrics=metrics,
                params=params
            )
        
        return run.info.run_id, metrics, run_name

def main():
    # Config 설정
    config = Config()
    
    # 로그 디렉토리 생성
    log_dir = config.base_path / "logs" / "lightning_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # MLflow 경로 정보 출력
    print("\n=== MLflow Configuration ===")
    print(f"MLflow Tracking URI: {config.mlflow.tracking_uri}")
    print(f"MLflow Run Path: {config.mlflow.mlrun_path}")
    print(f"MLflow Experiment Name: {config.mlflow.experiment_name}")
    print("=" * 50 + "\n")
    
    # 학습 실행
    run_id, metrics, run_name = train(config)
    
    print("\n=== Training completed ===")
    print(f"Run Name: {run_name}")
    print(f"Run ID: {run_id}")
    print("Validation metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    # MLflow 실험 결과 위치 출력
    print("\n=== MLflow Run Information ===")
    print(f"Run logs and artifacts: {config.mlflow.mlrun_path / run_id}")
    print("=" * 50)
    
    # 모델 관리 인터페이스 실행 (선택적)
    if input("\nWould you like to manage models? (y/n): ").lower() == 'y':
        model_manager = MLflowModelManager(config)
        model_manager.manage_model(config.project.model_name)
    
    # 체크포인트와 아티팩트 정리
    cleanup_artifacts(config, metrics, run_id)

if __name__ == '__main__':
    main()
