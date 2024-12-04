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

class SentimentTrainer:
    """감성 분석 모델 학습기"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Args:
            config_path: 설정 파일 경로
        """
        self.config = Config(config_path)
        self.setup_mlflow()
        
    def setup_mlflow(self):
        """MLflow 설정"""
        setup_mlflow_server(self.config)
        self.experiment_id = initialize_mlflow(self.config)
        
    def cleanup_training_artifacts(self):
        """학습 완료 후 임시 파일 정리"""
        print("\nCleaning up training artifacts...")
        
        # 체크포인트 폴더 삭제
        checkpoint_path = self.config.paths['model_checkpoints']
        if checkpoint_path.exists():
            shutil.rmtree(checkpoint_path)
            print(f"Removed checkpoints directory: {checkpoint_path}")
        
        # 모델 폴더 삭제
        model_path = self.config.paths['model']
        if model_path.exists():
            shutil.rmtree(model_path)
            print(f"Removed model directory: {model_path}")
            
    def train(self, interactive: bool = False) -> dict:
        """모델 학습 실행
        
        Args:
            interactive: 대화형 추론 및 모델 관리 기능 활성화 여부
            
        Returns:
            dict: 학습 결과 정보
            {
                'run_id': str,
                'metrics': dict,
                'run_name': str,
                'model': PreTrainedModel,
                'tokenizer': PreTrainedTokenizer,
                'data_module': NSMCDataModule
            }
        """
        print("=" * 50)
        print("\n=== Training Configuration ===")
        print(f"Model: {self.config.project['model_name']}")
        print(f"Pretrained Model: {self.config.model_config['pretrained_model']}")
        print(f"Batch Size: {self.config.training_config['batch_size']}")
        print(f"Learning Rate: {self.config.training_config['lr']}")
        print(f"Epochs: {self.config.training_config['epochs']}")
        print(f"Max Length: {self.config.training_config['max_length']}")
        print("=" * 50 + "\n")
        
        seed_everything(self.config.project['random_state'])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{self.config.project['model_name']}_{self.config.project['dataset_name']}_{timestamp}"
        
        with mlflow.start_run(run_name=run_name) as run:
            print(f"\n=== Starting new run: {run_name} ===")
            
            # 데이터 모듈 준비
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_config['pretrained_model'])
            data_module = NSMCDataModule(config=self.config, tokenizer=tokenizer)
            data_module.prepare_data()
            data_module.setup(stage='fit')
            log_data_info(data_module)
            
            # 모델 초기화
            model = self._initialize_model()
            
            # 학습 실행
            trainer = self._create_trainer()
            trainer.fit(model, data_module)
            
            # 평가
            metrics = self._evaluate_model(model, tokenizer, data_module)
            
            # 모델 저장 및 등록
            if metrics['val_f1'] > self.config.mlflow.model_registry_metric_threshold:
                self._save_model(run, model, metrics)
            
            # 임시 파일 정리
            self.cleanup_training_artifacts()
            
            result = {
                'run_id': run.info.run_id,
                'metrics': metrics,
                'run_name': run_name,
                'model': model,
                'tokenizer': tokenizer,
                'data_module': data_module
            }
            
            if interactive:
                self._run_interactive_features(result)
            
            return result
            
    def _initialize_model(self):
        """모델 초기화"""
        if self.config.project['model_name'].startswith('KcBERT'):
            from src.models.kcbert_model import KcBERT
            model_class = KcBERT
        elif self.config.project['model_name'].startswith('KcELECTRA'):
            from src.models.kcelectra_model import KcELECTRA
            model_class = KcELECTRA
        else:
            raise ValueError(f"Unknown model: {self.config.project['model_name']}")
        
        return model_class(**self.config.get_model_kwargs())
        
    def _create_trainer(self) -> Trainer:
        """Trainer 객체 생성"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                mode='min',
                verbose=True
            ),
            ModelCheckpoint(
                dirpath=self.config.checkpoint['dirpath'],
                filename=self.config.checkpoint['filename'],
                monitor=self.config.checkpoint['monitor'],
                mode=self.config.checkpoint['mode'],
                save_top_k=self.config.checkpoint['save_top_k'],
                save_last=self.config.checkpoint['save_last'],
                every_n_epochs=self.config.checkpoint['every_n_epochs']
            ),
            LearningRateMonitor(logging_interval='step')
        ]
        
        trainer_kwargs = {
            'max_epochs': self.config.training_config['epochs'],
            'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
            'devices': 1,
            'precision': self.config.training_config.get('precision', 16),
            'deterministic': True,
            'gradient_clip_val': 1.0,
            'accumulate_grad_batches': self.config.training_config.get('accumulate_grad_batches', 1),
            'strategy': 'auto',
            'enable_progress_bar': True,
            'log_every_n_steps': 100,
            'callbacks': callbacks,
            'num_sanity_val_steps': 0,
            'enable_checkpointing': True,
            'detect_anomaly': False,
            'inference_mode': True,
            'logger': TensorBoardLogger(
                save_dir=self.config.common['trainer']['logger']['save_dir'],
                name=self.config.common['trainer']['logger']['name'],
                version=self.config.common['trainer']['logger']['version']
            )
        }
        
        return Trainer(**trainer_kwargs)
        
    def _evaluate_model(self, model, tokenizer, data_module):
        """모델 평가"""
        evaluator = ModelEvaluator(model, tokenizer)
        metrics = evaluator.evaluate_dataset(data_module)
        
        print("\n=== Validation Results ===")
        print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
        print(f"Validation F1 Score: {metrics['f1']:.4f}")
        
        mlflow.log_metrics({
            "val_accuracy": metrics['accuracy'],
            "val_f1": metrics['f1']
        })
        
        return metrics
        
    def _save_model(self, run, model, metrics):
        """모델 저장 및 MLflow에 등록"""
        try:
            model_path = Path("mlruns") / run.info.experiment_id / run.info.run_id / "artifacts/model"
            model_path.mkdir(parents=True, exist_ok=True)
            
            # MLflow 포맷으로 저장
            mlflow.pytorch.save_model(pytorch_model=model, path=str(model_path))
            
            # 커스텀 포맷으로 저장
            torch.save(model.state_dict(), model_path / "model.pt")
            
            # 설정 저장
            model_config = {
                "model_type": self.config.project['model_name'],
                "pretrained_model": self.config.model_config['pretrained_model'],
                "num_labels": self.config.training_config['num_labels'],
                "max_length": self.config.training_config['max_length'],
                "batch_size": self.config.training_config['batch_size'],
                "lr": self.config.training_config['lr'],
                "epochs": self.config.training_config['epochs'],
                "random_state": self.config.project['random_state'],
                "dataset_name": self.config.project['dataset_name'],
                "sampling_rate": self.config.data['sampling_rate'],
                "test_size": self.config.data['test_size'],
                "train_data_path": self.config.data['train_data_path'],
                "val_data_path": self.config.data['val_data_path'],
                "column_mapping": self.config.data['column_mapping']
            }
            
            with open(model_path / "config.json", 'w', encoding='utf-8') as f:
                json.dump(model_config, f, indent=2, ensure_ascii=False)
            
            # MLflow에 아티팩트 등록
            mlflow.log_param("model_path", str(model_path))
            mlflow.log_artifact(str(model_path / "model.pt"))
            mlflow.log_artifact(str(model_path / "config.json"))
            
            # 모델 레지스트리에 등록
            model_manager = MLflowModelManager(self.config)
            model_version = model_manager.register_model(self.config.project['model_name'], run.info.run_id)
            model_manager.save_model_info(
                run_id=run.info.run_id,
                metrics={"val_f1": metrics['f1']},
                params=self.config.get_model_kwargs(),
                version=model_version.version
            )
            
        except Exception as e:
            print(f"Error during model saving: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
            
    def _run_interactive_features(self, result):
        """대화형 추론 및 모델 관리 기능 실행"""
        # 샘플 예측 출력
        self._show_sample_predictions(result)
        
        # 대화형 추론
        print("\n=== Interactive Inference ===")
        print("Enter your text (or 'q' to quit):")
        
        inferencer = ModelInferencer(result['model'], result['tokenizer'])
        
        while True:
            user_input = input("\nText: ").strip()
            if user_input.lower() == 'q':
                break
                
            if not user_input:
                continue
                
            prediction = inferencer.predict(user_input)[0]
            print(f"Prediction: {'긍정' if prediction['prediction'] == 1 else '부정'}")
            print(f"Confidence: {prediction['confidence']:.4f}")
        
        # 모델 관리
        if input("\nWould you like to manage models? (y/n): ").lower() == 'y':
            model_manager = MLflowModelManager(self.config)
            model_manager.manage_model(self.config.project['model_name'])
            
    def _show_sample_predictions(self, result):
        """검증 데이터셋에서 샘플 예측 출력"""
        print("\n=== Validation Sample Predictions ===")
        
        val_dataset = result['data_module'].val_dataset
        n_samples = 5
        indices = torch.randperm(len(val_dataset))[:n_samples].tolist()
        
        model = result['model']
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

def main():
    """커맨드 라인에서 실행할 때의 메인 함수"""
    trainer = SentimentTrainer()
    
    print('\n' * 3)
    print("=" * 50)
    print("\n=== MLflow Configuration ===")
    print(f"MLflow Tracking URI: {trainer.config.mlflow.tracking_uri}")
    print(f"MLflow Run Path: {trainer.config.mlflow.mlrun_path}")
    print(f"MLflow Experiment Name: {trainer.config.mlflow.experiment_name}")
    print("=" * 50 + "\n")
    
    # 대화형 기능을 활성화하여 학습 실행
    result = trainer.train(interactive=True)
    
    print("\n=== Training completed ===")
    print(f"Run Name: {result['run_name']}")
    print(f"Run ID: {result['run_id']}")
    print("Validation metrics:")
    for metric_name, value in result['metrics'].items():
        if isinstance(value, float):
            print(f"  {metric_name}: {value:.4f}")
    
    print("\n=== MLflow Run Information ===")
    print(f"Run logs and artifacts: {Path(trainer.config.mlflow.mlrun_path) / result['run_id']}")
    print("=" * 50)
    
    cleanup_artifacts(trainer.config, result['metrics'], result['run_id'])

if __name__ == '__main__':
    main()

