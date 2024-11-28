from abc import ABC, abstractmethod
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, precision_score, recall_score
from transformers import Trainer, TrainingArguments
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset
from typing import Optional, Dict, Any
from datetime import datetime
import mlflow
from utils.mlflow_utils import MLflowLogger
import tempfile
import shutil
import logging
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

    def to_pandas(self) -> pd.DataFrame:
        """데이터셋을 pandas DataFrame으로 변환"""
        # input_ids를 numpy 배열로 변환
        input_ids = self.encodings['input_ids'].numpy()
        attention_mask = self.encodings['attention_mask'].numpy()
        
        # 라벨도 numpy 배열로 변환
        labels = np.array(self.labels)
        
        return pd.DataFrame({
            'text': input_ids.tolist(),  # 텍스트는 리스트로 저장
            'label': labels,
            'attention_mask': attention_mask.tolist()
        })

class BaseSentimentModel(ABC):
    def __init__(self, data_file: Path, pretrained_model_name: str, batch_size: int = 32):
        self.data_file = data_file
        self.pretrained_model_name = pretrained_model_name
        self.mlflow_logger = None
        self.batch_size = batch_size
        
        # 데이터셋과 DataLoader
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        
        # device 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {self.device}')
        
        # 모델 디렉토리 설정
        self.pretrained_model_dir = Path(__file__).parent.parent.parent / 'models' / 'pretrained' / pretrained_model_name.split('/')[-1]
        self.model_dir = Path(tempfile.mkdtemp(prefix='model_'))
        
        self.pretrained_model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f'=== Sentiment Model 인스턴스 생성 완료...')
        logger.info(f'Pretrained 모델 경로: {self.pretrained_model_dir}')
        logger.info(f'임시 작업 디렉토리: {self.model_dir}')
    
    def __del__(self):
        """임시 디렉토리 정리"""
        if hasattr(self, 'model_dir') and self.model_dir.exists():
            shutil.rmtree(self.model_dir)
            logger.info(f'임시 디렉토리 삭제 완료: {self.model_dir}')
    
    def set_mlflow_logger(self, mlflow_logger: 'MLflowLogger'):
        """MLflow 로거 설정"""
        self.mlflow_logger = mlflow_logger
    
    def log_model_info(self, dataset_name: str, run_name: Optional[str] = None):
        """모델 정보 로깅"""
        if not self.mlflow_logger:
            return
            
        with self.mlflow_logger.start_run(run_name=run_name) as run:
            # 기본 파라미터 로깅
            self.mlflow_logger.log_params({
                'model_name': self.pretrained_model_name,
                'dataset_name': dataset_name,
                'model_dir': str(self.model_dir),
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
            })
            
            return run.info.run_id
    
    def log_training_info(self, train_args: Dict[str, Any]):
        """학습 관련 정보 로깅"""
        if not self.mlflow_logger:
            return
            
        self.mlflow_logger.log_params({
            'learning_rate': train_args.get('learning_rate'),
            'num_train_epochs': train_args.get('num_train_epochs'),
            'batch_size': train_args.get('batch_size'),
            'num_unfrozen_layers': train_args.get('num_unfrozen_layers')
        })
    
    def log_metrics(self, metrics: Dict[str, float]):
        """평가 메트릭 로깅"""
        if not self.mlflow_logger:
            return
            
        self.mlflow_logger.log_metrics(metrics)
    
    def log_datasets(self, dataset_name: str):
        """데이터셋 정보 로깅"""
        if self.train_dataset is None or self.val_dataset is None:
            logger.error("데이터셋이 로드되지 않았습니다.")
            return
        
        try:
            # 데이터셋 정보 로깅
            dataset_info = {
                'dataset_name': dataset_name,
                'train_samples': len(self.train_dataset),
                'val_samples': len(self.val_dataset),
                'test_samples': len(self.test_dataset) if self.test_dataset else 0
            }
            
            if self.mlflow_logger:
                self.mlflow_logger.log_params(dataset_info)
                logger.info(f"데이터셋 정보 로깅 완료: {dataset_info}")
        except Exception as e:
            logger.error(f"데이터셋 정보 로깅 중 오류 발생: {str(e)}")
    
    def log_model_artifacts(self, dataset_name: str) -> Optional[str]:
        """
        모델 아티팩트 로깅 및 버전 관리
        Returns:
            등록된 모델 버전
        """
        if not self.mlflow_logger:
            return None
        
        # 모델 아티팩트 로깅
        self.mlflow_logger.log_model_artifacts(
            model=self.model,
            tokenizer=self.tokenizer,
            model_name=self.pretrained_model_name,
            dataset_name=dataset_name
        )
        
        # 새 버전 등록
        version = self.mlflow_logger.register_model_version(
            run_id=mlflow.active_run().info.run_id,
            model_name=self.pretrained_model_name,
            dataset_name=dataset_name
        )
        
        if version:
            logger.info(f"모델 버전 {version} 등록 완료")
            # 버전 정보 로깅
            self.mlflow_logger.log_params({
                'model_version': version,
                'model_registry_stage': self.mlflow_logger.config.model_registry_stage
            })
        
        return version
    
    def log_confusion_matrix(self):
        """Confusion Matrix 생성 및 로깅"""
        if not self.mlflow_logger:
            return
            
        # 일반 confusion matrix
        self.save_confusion_matrix(normalize=False)
        self.mlflow_logger.log_artifacts(
            str(self.model_dir / "confusion_matrix.png"), 
            "confusion_matrices"
        )
        
        # 정규화된 confusion matrix
        self.save_confusion_matrix(normalize=True)
        self.mlflow_logger.log_artifacts(
            str(self.model_dir / "confusion_matrix_normalized.png"), 
            "confusion_matrices"
        )
    
    def log_prediction(self, text: str, sentiment: str, confidence: float):
        """예측 결과 로깅"""
        if not self.mlflow_logger:
            return
            
        with self.mlflow_logger.start_run(nested=True):
            self.mlflow_logger.log_metrics({
                'prediction_confidence': confidence
            })
            self.mlflow_logger.log_params({
                'input_text': text,
                'predicted_sentiment': sentiment
            })
    
    def load_model(self, run_id: Optional[str] = None):
        """
        MLflow 또는 Hugging Face에서 모 
        Args:
            run_id: MLflow run ID (있으면 MLflow에서 로드)
        """
        try:
            if run_id:
                # MLflow에서 모델 로드
                logged_model = f"runs:/{run_id}/model"
                self.model = mlflow.transformers.load_model(logged_model)
                self.tokenizer = self.model['tokenizer']
                self.model = self.model['model']
                logger.info(f"MLflow에서 모델 로드 완료 (run_id: {run_id})")
            else:
                # Hugging Face에서 직접 다운로드
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.pretrained_model_name,
                    num_labels=2
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
                logger.info(f"Hugging Face에서 모델 로드 완료 ({self.pretrained_model_name})")
        except Exception as e:
            logger.error(f"모델 로드 실패: {str(e)}")
            raise
    
    def save_model(self):
        """모델 저장"""
        try:
            if self.mlflow_logger is None:
                logger.warning("MLflow logger가 설정되지 않았습니다.")
                return
            
            # Pipeline 생성
            from transformers import pipeline
            device = 0 if torch.cuda.is_available() else -1  # GPU 사용 가능하면 0, 아니면 -1
            
            nlp = pipeline(
                task="text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device,
                model_kwargs={"device_map": "auto"}  # device mapping 자동 설정
            )
            
            # 모델 카드 정보 추가
            model_card = {
                "license": "MIT",  # 라이선스 정보 추가
                "tags": ["text-classification", "sentiment-analysis", "korean"],
                "description": "Korean sentiment analysis model trained on Naver Movie Review dataset",
                "model-index": [
                    {
                        "name": self.pretrained_model_name,
                        "results": []
                    }
                ]
            }
            
            # MLflow에 모델 저장
            mlflow.transformers.log_model(
                transformers_model=nlp,
                artifact_path="model",
                task="text-classification",
                input_example=["이 영화 정 재미있어요!", "별로였습니다."],
                signature=mlflow.models.signature.infer_signature(
                    model_input=["이 영화 정말 재미있어요!"],
                    model_output=[{"label": "POSITIVE", "score": 0.9}]
                ),
                metadata=model_card  # 모델 카드 정보 추가
            )
            logger.info(f"모델이 MLflow에 저장되었습니다. (device: {device})")
            
        except Exception as e:
            logger.error(f"모델 저장 중 오류 발생: {str(e)}")
            raise
    
    def create_dataset(self, df: pd.DataFrame, max_length: int) -> SentimentDataset:
        """데이터셋 생성"""
        encodings = self.tokenizer(
            df['text'].tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        return SentimentDataset(encodings, df['label'].values)

    def load_data(self, sampling_rate: float = 1.0, max_length: int = 256):
        """데이터 로드 및 전처리"""
        try:
            logger.info(f"데이터 로드 시작 (max_length: {max_length})")
            
            # 데이터 로드
            df = pd.read_csv(self.data_file)
            
            if df.empty:
                raise ValueError("데이터 파일이 비어있습니다.")
            
            logger.info(f"데이터 크기: {len(df)}")
            
            # 샘플링 제거 (이미 전처리 단계에서 수행됨)
            
            # 학습/검증/테스트 분할
            train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
            val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
            
            # 텍스트 데이터가 'text' 또는 'document' 컬럼에 있는지 확인
            text_column = 'text' if 'text' in train_df.columns else 'document'
            
            # 데이터셋 및 인코딩 생성
            self.train_encodings = self.tokenizer(
                train_df[text_column].astype(str).tolist(),
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt"
            )
            self.val_encodings = self.tokenizer(
                val_df[text_column].astype(str).tolist(),
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt"
            )
            self.test_encodings = self.tokenizer(
                test_df[text_column].astype(str).tolist(),
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            # 데이터셋 생성
            self.train_dataset = SentimentDataset(self.train_encodings, train_df['label'].values)
            self.val_dataset = SentimentDataset(self.val_encodings, val_df['label'].values)
            self.test_dataset = SentimentDataset(self.test_encodings, test_df['label'].values)
            
            # 원본 데이터프레임도 저장
            self.train_df = train_df
            self.val_df = val_df
            self.test_df = test_df
            
            # DataLoader 설정
            self.setup_dataloaders()
            
            # 데이터셋 통계 로깅
            logger.info(f"데이터 로드 완료:")
            logger.info(f"- 학습 데이터: {len(self.train_dataset)} 샘플")
            logger.info(f"- 검증 데이터: {len(self.val_dataset)} 샘플")
            logger.info(f"- 테스트 데이터: {len(self.test_dataset)} 샘플")
            
        except Exception as e:
            logger.error(f"데이터 로드 중 오류 발생: {str(e)}")
            raise

    def compute_metrics(self, eval_pred):
        """평가 메트릭 계산"""
        try:
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            
            # 기본 메트릭 계산 (prefix 없이)
            metrics = {
                'accuracy': accuracy_score(labels, predictions),
                'f1': f1_score(labels, predictions, average='binary'),
                'precision': precision_score(labels, predictions, average='binary'),
                'recall': recall_score(labels, predictions, average='binary')
            }
            
            # MLflow 로깅을 위한 별도의 메트릭 딕셔너리 생성
            if self.mlflow_logger is not None:
                eval_metrics = {f'eval_{k}': v for k, v in metrics.items()}
                self.mlflow_logger.log_metrics(eval_metrics)
            
            # prefix 없는 원래 메트릭 반환
            return metrics
            
        except Exception as e:
            logger.error(f"메트릭 계산 중 오류 발생: {str(e)}")
            # 기���값도 prefix ��이 반환
            return {
                'accuracy': 0.0,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0
            }

    def freeze_layers(self, num_unfrozen_layers: int = 2):
        """
        Freezes all layers except the specified number of top layers.
        Args:
            num_unfrozen_layers (int): Number of top layers to unfreeze (keep trainable)
        """
        # Freeze embeddings
        for param in self.model.bert.embeddings.parameters():
            param.requires_grad = False
        print("임베딩 레이어가 고정되었습니다.")

        # Get total number of layers
        total_layers = len(self.model.bert.encoder.layer)
        
        # Ensure we don't try to unfreeze more layers than we have
        num_unfrozen_layers = min(num_unfrozen_layers, total_layers)
        num_layers_to_freeze = total_layers - num_unfrozen_layers
        
        # Freeze layers from bottom
        for layer in self.model.bert.encoder.layer[:num_layers_to_freeze]:
            for param in layer.parameters():
                param.requires_grad = False
                
        # Unfreeze top layers
        for layer in self.model.bert.encoder.layer[num_layers_to_freeze:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        print(f"\n=== 레이어 고정 상태 ===")
        print(f"전체 레이어 수: {total_layers}")
        print(f"고정된 레이어 수: {num_layers_to_freeze}")
        print(f"학습에 사용될 이어 수: {num_unfrozen_layers}")
        
        # Count and print trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"\n=== 파라미터 상태 ===")
        print(f"전체 파라미터: {total_params:,}")
        print(f"학습 가능한 파라미터: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        print(f"고정된 파라미터: {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")

    def predict_samples(self, num_samples: int = 5):
        """무작위 샘플에 대한 예측 수행"""
        try:
            if not hasattr(self, 'val_df') or self.val_df is None:
                logger.error("검증 데이터가 로드되지 않았습니다.")
                return
            
            # 무작위 샘플 선택
            sample_indices = np.random.choice(len(self.val_df), num_samples)
            
            logger.info("\n=== 샘플 예측 결과 ===")
            for idx in sample_indices:
                # 텍스트와 실제 레이블 가져오기
                text = self.val_df['text'].iloc[idx]  # 'sentence' 대신 'text' 사용
                true_label = self.val_df['label'].iloc[idx]
                
                # 예측 수행
                inputs = self.tokenizer(
                    text,
                    truncation=True,
                    padding=True,
                    max_length=256,
                    return_tensors="pt"
                ).to(self.device)
                
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=1)
                    pred_label = torch.argmax(probs, dim=1).item()
                    confidence = probs[0][pred_label].item()
                
                # 결과 출력
                sentiment = "긍정" if pred_label == 1 else "부정"
                true_sentiment = "긍정" if true_label == 1 else "부정"
                
                logger.info(f"\n텍스트: {text}")
                logger.info(f"실제 감성: {true_sentiment}")
                logger.info(f"예측 감성: {sentiment} (확률: {confidence:.2%})")
                
        except Exception as e:
            logger.error(f"샘플 예측 중 오류 발생: {str(e)}")

    def train(self, train_args: dict = None, num_unfrozen_layers: int = 2):
        """
        Train the model with given arguments
        Args:
            train_args (dict): Training arguments including learning_rate, num_train_epochs, batch_size
            num_unfrozen_layers (int): Number of layers to unfreeze for training
        """
        if train_args is None:
            train_args = {}
        
        # 데이터 로드 및 준비
        sampling_rate = train_args.get('sampling_rate', 1.0)
        max_length = train_args.get('max_length', 256)
        
        self.load_data(sampling_rate=sampling_rate, max_length=max_length)
        # prepare_data는 이제 필요 없음 (load_data에서 모든 처리를 수행)
        
        # Freeze layers before training
        self.freeze_layers(num_unfrozen_layers=num_unfrozen_layers)
        
        # Default training arguments
        default_args = {
            'learning_rate': 2e-5,
            'num_train_epochs': 3,
            'batch_size': 16
        }
        
        # Update with provided arguments
        default_args.update(train_args)
        
        training_args = TrainingArguments(
            output_dir=os.path.join(self.model_dir, 'checkpoints'),
            num_train_epochs=default_args['num_train_epochs'],
            per_device_train_batch_size=default_args['batch_size'],
            per_device_eval_batch_size=default_args['batch_size'],
            evaluation_strategy='epoch',
            save_strategy='epoch',
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            logging_dir=os.path.join(self.model_dir, 'logs'),
            learning_rate=default_args['learning_rate'],
            warmup_steps=500,
            weight_decay=0.01,
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        self.trainer.train()
        self.save_model()
        
        # After training, show some prediction samples
        self.predict_samples(num_samples=5)

    def evaluate(self) -> Dict[str, float]:
        """모델 평가 수행"""
        try:
            # 평가 수행
            eval_results = self.trainer.evaluate()
            
            # 예측값 얻기
            predictions = self.trainer.predict(self.val_dataset)
            y_pred = np.argmax(predictions.predictions, axis=-1)
            y_true = predictions.label_ids
            
            # 메트릭스 계산
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'f1': f1_score(y_true, y_pred, average='binary'),
                'precision': precision_score(y_true, y_pred, average='binary'),
                'recall': recall_score(y_true, y_pred, average='binary')
            }
            
            # Confusion Matrix 저장 디렉토리 설정
            confusion_matrix_dir = Path("confusion_matrices")
            confusion_matrix_dir.mkdir(parents=True, exist_ok=True)
            
            # 기본 Confusion Matrix 생성 및 저장
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(confusion_matrix_dir / "confusion_matrix.png")
            plt.close()
            
            # 정규화된 Confusion Matrix 생성 및 저장
            plt.figure(figsize=(8, 6))
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues')
            plt.title('Normalized Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(confusion_matrix_dir / "confusion_matrix_normalized.png")
            plt.close()
            
            # MLflow에 아티팩트 로깅
            if self.mlflow_logger is not None:
                self.mlflow_logger.log_metrics(metrics)
                mlflow.log_artifacts(str(confusion_matrix_dir), "confusion_matrices")
                logger.info(f"Confusion Matrix 이미지가 저장되었습니다: {confusion_matrix_dir}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"평가 중 오류 발생: {str(e)}")
            logger.error(f"예측 결과: {predictions if 'predictions' in locals() else 'Not available'}")
            return {
                'accuracy': 0.0,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0
            }

    def setup_dataloaders(self):
        """DataLoader 설정"""
        if self.train_dataset is not None:
            self.train_dataloader = DataLoader(
                self.train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True
            )
        if self.val_dataset is not None:
            self.val_dataloader = DataLoader(
                self.val_dataset, 
                batch_size=self.batch_size, 
                shuffle=False
            )
        if self.test_dataset is not None:
            self.test_dataloader = DataLoader(
                self.test_dataset, 
                batch_size=self.batch_size, 
                shuffle=False
            )

    @abstractmethod
    def predict(self, dataset):
        """예측 수행"""
        pass