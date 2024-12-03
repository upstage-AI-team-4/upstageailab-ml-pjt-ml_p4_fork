import os
import warnings
from pathlib import Path
import json
from datetime import datetime
import mlflow
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
import transformers
from transformers import AutoTokenizer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import shutil

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from src.config import Config
from src.data.nsmc_dataset import NSMCDataModule, log_data_info
from src.utils.mlflow_utils import MLflowModelManager, cleanup_artifacts, initialize_mlflow, setup_mlflow_server
from src.utils.evaluator import ModelEvaluator
from src.utils.inferencer import ModelInferencer
from src.utils.visualization import plot_confusion_matrix

# torchvision 관련 경고 무시
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
# Tensor Core 최적화를 위한 precision 설정
torch.set_float32_matmul_precision('medium')  # 또는 'high'

def cleanup_training_artifacts(config: Config):
    """학습 완료 후 임시 파일 정리"""
    print("\nCleaning up training artifacts...")
    
    # 체크포인트 폴더 삭제
    checkpoint_path = config.paths['model_checkpoints']
    if checkpoint_path.exists():
        shutil.rmtree(checkpoint_path)
        print(f"Removed checkpoints directory: {checkpoint_path}")
    
    # 모델 폴더 삭제
    model_path = config.paths['model']
    if model_path.exists():
        shutil.rmtree(model_path)
        print(f"Removed model directory: {model_path}")

def train(config):
    print("=" * 50)
    print("\n=== Training Configuration ===")
    print(f"Model: {config.project['model_name']}")
    print(f"Pretrained Model: {config.model_config['pretrained_model']}")
    print(f"Batch Size: {config.training_config['batch_size']}")
    print(f"Learning Rate: {config.training_config['lr']}")
    print(f"Epochs: {config.training_config['epochs']}")
    print(f"Max Length: {config.training_config['max_length']}")
    print("=" * 50 + "\n")
    
    # MLflow 설정
    setup_mlflow_server(config)
    experiment_id = initialize_mlflow(config)
    
    seed_everything(config.project['random_state'])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.project['model_name']}_{config.project['dataset_name']}_{timestamp}"
    
    with mlflow.start_run(run_name=run_name) as run:
        print(f"\n=== Starting new run: {run_name} ===")
        
        tokenizer = AutoTokenizer.from_pretrained(config.model_config['pretrained_model'])
        
        data_module = NSMCDataModule(
            config=config,
            tokenizer=tokenizer
        )
        
        data_module.prepare_data()
        data_module.setup(stage='fit')
        
        # Log data info
        log_data_info(data_module)
        
        # Model initialization
        if config.project['model_name'].startswith('KcBERT'):
            from src.models.kcbert_model import KcBERT
            model_class = KcBERT
        elif config.project['model_name'].startswith('KcELECTRA'):
            from src.models.kcelectra_model import KcELECTRA
            model_class = KcELECTRA
        else:
            raise ValueError(f"Unknown model: {config.project['model_name']}")
        
        model = model_class(**config.get_model_kwargs())
        
        # Optimized training configuration
        callbacks = [
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                mode='min',
                verbose=True
            ),
            # Model checkpoint
            ModelCheckpoint(
                dirpath=config.checkpoint['dirpath'],
                filename=config.checkpoint['filename'],
                monitor=config.checkpoint['monitor'],
                mode=config.checkpoint['mode'],
                save_top_k=config.checkpoint['save_top_k'],
                save_last=config.checkpoint['save_last'],
                every_n_epochs=config.checkpoint['every_n_epochs']
            ),
            # Learning rate monitor
            LearningRateMonitor(logging_interval='step')
        ]
        
        # Trainer configuration with optimizations for single GPU
        trainer_kwargs = {
            'max_epochs': config.training_config['epochs'],
            'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
            'devices': 1,
            'precision': config.training_config.get('precision', 16),
            'deterministic': True,
            'gradient_clip_val': 1.0,
            'accumulate_grad_batches': config.training_config.get('accumulate_grad_batches', 1),
            'strategy': 'auto',  # Use auto strategy instead of DDP
            'enable_progress_bar': True,
            'log_every_n_steps': 100,
            'callbacks': callbacks,
            'num_sanity_val_steps': 0,
            'enable_checkpointing': True,
            'detect_anomaly': False,
            'inference_mode': True,
            'move_metrics_to_cpu': True,
            'logger': True
        }
        
        # Logger 설정
        logger = TensorBoardLogger(
            save_dir=config.common['trainer']['logger']['save_dir'],
            name=config.common['trainer']['logger']['name'],
            version=config.common['trainer']['logger']['version']
        )
        trainer_kwargs['logger'] = logger
        
        # Trainer 초기화
        trainer = Trainer(**trainer_kwargs)
        
        # Train the model
        trainer.fit(model, data_module)
        
        # Evaluation
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
        
        # Model registration if performance meets threshold
        if val_f1 > config.mlflow.model_registry_metric_threshold:
            print("\nSaving model artifacts...")
            
            try:
                # 모델 저장 경로 구성
                model_path = Path("mlruns") / run.info.experiment_id / run.info.run_id / "artifacts/model"
                model_path.mkdir(parents=True, exist_ok=True)
                print(f"Debug: Saving model to path: {model_path}")
                print(f"Debug: Run ID: {run.info.run_id}")
                print(f"Debug: Experiment ID: {run.info.experiment_id}")
                
                # MLflow 포맷으로 모델 저장
                mlflow.pytorch.save_model(
                    pytorch_model=model,
                    path=str(model_path)
                )
                print("Debug: Model saved in MLflow format")
                
                # 커스텀 포맷으로 모델 저장 (model.pt, config.json)
                torch.save(model.state_dict(), model_path / "model.pt")
                
                # 설정 저장
                model_config = {
                    "model_type": config.project['model_name'],
                    "pretrained_model": config.model_config['pretrained_model'],
                    "num_labels": config.training_config['num_labels'],
                    "max_length": config.training_config['max_length']
                }
                
                with open(model_path / "config.json", 'w', encoding='utf-8') as f:
                    json.dump(model_config, f, indent=2, ensure_ascii=False)
                    
                print("Debug: Model saved in custom format (model.pt and config.json)")
                
                # MLflow에 모델 경로 기록
                mlflow.log_param("model_path", str(model_path))
                print("Debug: Model path logged to MLflow")
                
                # 혼동 행렬 저장
                print("\nGenerating and logging confusion matrix...")
                confusion_matrix_img = plot_confusion_matrix(data_module.val_dataset, model, tokenizer)
                confusion_matrix_path = model_path.parent / "confusion_matrix.png"
                confusion_matrix_img.save(str(confusion_matrix_path))
                print("Debug: Confusion matrix saved successfully")
                
                # 모델 정보 저장
                print("Debug: Saving model registry information...")
                model_manager = MLflowModelManager(config)
                model_version = model_manager.register_model(config.project['model_name'], run.info.run_id)
                model_manager.save_model_info(
                    run_id=run.info.run_id,
                    metrics={"val_f1": val_f1},
                    params=config.get_model_kwargs(),
                    version=model_version.version
                )
                print("Debug: Model registry information saved successfully")
                
            except Exception as e:
                print(f"Debug: Error during model saving: {str(e)}")
                print(f"Debug: Error type: {type(e)}")
                import traceback
                print("Debug: Full traceback:")
                traceback.print_exc()
                raise
        
        # 학습 완료 후 임시 파일 정리
        cleanup_training_artifacts(config)
        
        return run.info.run_id, {"val_accuracy": val_accuracy, "val_f1": val_f1}, run_name, data_module, model, tokenizer

def main():
    config = Config()
    
    print('\n' * 3)
    print("=" * 50)
    print("\n=== MLflow Configuration ===")
    print(f"MLflow Tracking URI: {config.mlflow.tracking_uri}")
    print(f"MLflow Run Path: {config.mlflow.mlrun_path}")
    print(f"MLflow Experiment Name: {config.mlflow.experiment_name}")
    print("=" * 50 + "\n")
    print(f"Model Name: {config.project['model_name']}")
    print(f"Dataset Name: {config.data['dataset_name']}")
    print(f"Sampling Rate: {config.data['sampling_rate']}")
    print(f"Test Size: {config.data['test_size']}")

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
            text = val_dataset.texts[idx]
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
    print(f"Run logs and artifacts: {Path(config.mlflow.mlrun_path) / run_id}")
    print("=" * 50)
    
    if input("\nWould you like to manage models? (y/n): ").lower() == 'y':
        model_manager = MLflowModelManager(config)
        model_manager.manage_model(config.project['model_name'])
    
    cleanup_artifacts(config, metrics, run_id)

if __name__ == '__main__':
    main()

