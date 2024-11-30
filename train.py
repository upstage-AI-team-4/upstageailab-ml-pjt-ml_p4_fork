import os
import warnings
from pathlib import Path
import json
from datetime import datetime
import mlflow
import torch
from pytorch_lightning import Trainer, seed_everything
from transformers import AutoTokenizer
from src.utils.visualization import plot_confusion_matrix
from src.models.kcbert_model import KcBERT
from src.data.nsmc_dataset import NSMCDataModule, log_data_info
from src.utils.config import Config
from src.utils.mlflow_utils import MLflowModelManager, cleanup_artifacts
from src.utils.evaluator import ModelEvaluator
from src.utils.inferencer import ModelInferencer
import pandas as pd
import numpy as np

# torchvision 관련 경고 무시
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
# Tensor Core 최적화를 위한 precision 설정
torch.set_float32_matmul_precision('medium')  # 또는 'high'

def get_local_path_from_uri(uri: str) -> Path:
    """MLflow URI를 로컬 경로로 변환"""
    if uri.startswith(('file:///', 'file://', 'file:')):
        path = uri.split('file:')[-1].lstrip('/')
        if os.name == 'nt' and len(path) >= 2 and path[1] == ':':
            return Path(path)
        return Path('/' + path)
    return Path(uri)

def train(config):
    print("=" * 50)
    print("\n=== Training Configuration ===")
    print(f"Pretrained Model: {config.base_training.pretrained_model}")
    print(f"Batch Size: {config.base_training.batch_size}")
    print(f"Learning Rate: {config.base_training.lr}")
    print(f"Epochs: {config.base_training.epochs}")
    print(f"Max Length: {config.base_training.max_length}")
    print("=" * 50 + "\n")
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    
    # Experiment 설정
    mlflow.set_experiment(config.mlflow.experiment_name)
    
    seed_everything(config.base_training.random_seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.project.model_name}_{config.project.dataset_name}_{timestamp}"
    
    with mlflow.start_run(run_name=run_name) as run:
        print(f"\n=== Starting new run: {run_name} ===")
        
        tokenizer = AutoTokenizer.from_pretrained(config.base_training.pretrained_model)
        
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
        
        data_module.prepare_data()
        data_module.setup(stage='fit')
        
        # 데이터 정보 로깅 및 출력
        log_data_info(data_module, config)
        
        model = KcBERT(**config.get_model_kwargs())
        
        trainer = Trainer(**config.get_trainer_kwargs())
        trainer.fit(model, data_module)
        
        # 평가기 초기화 및 사용
        evaluator = ModelEvaluator(model, tokenizer)
        eval_metrics = evaluator.evaluate_dataset(data_module)
        
        val_accuracy = eval_metrics['accuracy']
        val_f1 = eval_metrics['f1']
        
        print("\n=== Validation Results ===")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation F1 Score: {val_f1:.4f}")
        
        mlflow.log_metrics({
            "val_accuracy": val_accuracy,
            "val_f1": val_f1
        })
        
        # Confusion matrix 생성 및 로깅
        all_preds = []
        all_labels = []
        model.eval()
        with torch.no_grad():
            for batch in data_module.val_dataloader():
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**{k: v for k, v in batch.items() if k != 'labels'})
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        cm_path = plot_confusion_matrix(all_labels, all_preds, log_path='confusion_matrix.png')
        mlflow.log_artifact(cm_path, "confusion_matrix")
        os.remove(cm_path)
        
        cm_norm_path = plot_confusion_matrix(all_labels, all_preds, normalize=True, log_path='confusion_matrix_normalized.png')
        mlflow.log_artifact(cm_norm_path, "confusion_matrix_normalized")
        os.remove(cm_norm_path)
        
        # 모델 등록 및 정보 저장
        if val_f1 > config.mlflow.model_registry_metric_threshold:
            model_manager = MLflowModelManager(config)
            model_version = model_manager.register_model(config.project.model_name, run.info.run_id)
            model_manager.save_model_info(run.info.run_id, {"val_f1": val_f1}, config.get_model_kwargs())
        
        return run.info.run_id, {"val_accuracy": val_accuracy, "val_f1": val_f1}, run_name, data_module, model, tokenizer

def main():
    config = Config()
    
    log_dir = config.base_path / "logs" / "lightning_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    print("=" * 50)
    print("\n=== MLflow Configuration ===")
    print(f"MLflow Tracking URI: {config.mlflow.tracking_uri}")
    print(f"MLflow Run Path: {config.mlflow.mlrun_path}")
    print(f"MLflow Experiment Name: {config.mlflow.experiment_name}")
    print("=" * 50 + "\n")
    print(f"Model Name: {config.project.model_name}")
    print(f"Dataset Name: {config.data.dataset_name}")
    print(f"Sampling Rate: {config.data.sampling_rate}")
    print(f"Test Size: {config.data.test_size}")

    run_id, metrics, run_name, data_module, model, tokenizer = train(config)
    
    print("\n=== Training completed ===")
    print(f"Run Name: {run_name}")
    print(f"Run ID: {run_id}")
    print("Validation metrics:")
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric_name}: {value:.4f}")
    
    print("\n=== Validation Sample Predictions ===")
    
    val_dataset = data_module.val_dataset
    n_samples = 5
    indices = torch.randperm(len(val_dataset))[:n_samples].tolist()
    
    model.eval()
    with torch.no_grad():
        for idx in indices:
            text = val_dataset.documents[idx]
            true_label = val_dataset.labels[idx]
            
            sample = val_dataset[idx]
            inputs = {
                'input_ids': sample['input_ids'].unsqueeze(0).to(model.device),
                'attention_mask': sample['attention_mask'].unsqueeze(0).to(model.device)
            }
            
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_label = torch.argmax(logits, dim=-1).item()
            confidence = probs[0][pred_label].item()
            
            print("\nText:", text)
            print(f"True Label: {'긍정' if true_label == 1 else '부정'}")
            print(f"Prediction: {'긍정' if pred_label == 1 else '부정'}")
            print(f"Confidence: {confidence:.4f}")
            print(f"Correct: {'O' if pred_label == true_label else 'X'}")
            print("-" * 80)
    
    print("\n=== Interactive Inference ===")
    print("Enter your text (or 'q' to quit):")
    
    inferencer = ModelInferencer(model, tokenizer)
    
    while True:
        user_input = input("\nText: ").strip()
        if user_input.lower() == 'q':
            break
            
        if not user_input:
            continue
            
        result = inferencer.predict(user_input)[0]
        print(f"Prediction: {'긍정' if result['prediction'] == 1 else '부정'}")
        print(f"Confidence: {result['confidence']:.4f}")
    
    print("\n=== MLflow Run Information ===")
    print(f"Run logs and artifacts: {config.mlflow.mlrun_path / run_id}")
    print("=" * 50)
    
    if input("\nWould you like to manage models? (y/n): ").lower() == 'y':
        model_manager = MLflowModelManager(config)
        model_manager.manage_model(config.project.model_name)
    
    cleanup_artifacts(config, metrics, run_id)

if __name__ == '__main__':
    main()
