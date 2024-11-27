from abc import ABC, abstractmethod
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels.iloc[idx]))
        return item

    def __len__(self):
        return len(self.labels)

    def to_pandas(self):
        """Convert dataset to pandas DataFrame"""
        return pd.DataFrame({
            'text': self.encodings['input_ids'],
            'label': self.labels
        })

class BaseSentimentModel(ABC):
    def __init__(self, data_file: Path, model_dir: Path, pretrained_model_name: str, pretrained_model_dir: Path):
        self.data_file = data_file
        self.model_dir = model_dir
        self.pretrained_model_name = pretrained_model_name
        self.pretrained_model_dir = pretrained_model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        print(f'\n===Sentiment Model 인스턴스 생성 완료...')
        
    @abstractmethod
    def load_model(self):
        pass
    
    def load_data(self):
        self.data = pd.read_csv(self.data_file)
        print(f"데이터를 {self.data_file}에서 로드했습니다.")

    def prepare_data(self):
        """
        Prepare data for training by converting tokens back to sentences
        """
        # Convert tokens (list of tuples) to sentence
        def tokens_to_sentence(tokens_str):
            try:
                tokens = eval(tokens_str)  # string to list of tuples
                # Extract only the tokens (first element of each tuple)
                words = [token[0] for token in tokens]
                return ' '.join(words)
            except:
                return ''

        print("데이터 준비 시작...")
        print(f"전체 데이터 크기: {len(self.data)}")
        
        # Convert tokens to sentences
        self.data['sentence'] = self.data['tokens'].apply(tokens_to_sentence)
        
        # Remove empty sentences
        empty_mask = self.data['sentence'].str.strip() != ''
        self.data = self.data[empty_mask]
        print(f"빈 문장 제거 후 데이터 크기: {len(self.data)}")
        # Prepare train/val split
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            self.data['sentence'], 
            self.data['label'], 
            test_size=0.2, 
            random_state=42
        )
        
        print("토큰화 시작...")
        self.train_encodings = self.tokenizer(
            list(train_texts), 
            truncation=True, 
            padding=True, 
            max_length=128
        )
        self.val_encodings = self.tokenizer(
            list(val_texts), 
            truncation=True, 
            padding=True, 
            max_length=128
        )
        
        self.train_labels = train_labels.reset_index(drop=True)
        self.val_labels = val_labels.reset_index(drop=True)
        
        # Create datasets
        self.train_dataset = SentimentDataset(self.train_encodings, self.train_labels)
        self.val_dataset = SentimentDataset(self.val_encodings, self.val_labels)
        
        print(f"학습 데이터: {len(train_texts)}개")
        print(f"검증 데이터: {len(val_texts)}개")
        
    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc, 
            'precision': precision, 
            'recall': recall, 
            'f1': f1
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
        print(f"학습에 사용될 레이어 수: {num_unfrozen_layers}")
        
        # Count and print trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"\n=== 파라미터 상태 ===")
        print(f"전체 파라미터: {total_params:,}")
        print(f"학습 가능한 파라미터: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        print(f"고정된 파라미터: {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")

    def predict_samples(self, num_samples: int = 5):
        """
        Predict sentiment for random samples and show results
        Args:
            num_samples (int): Number of samples to show
        """
        print("\n=== 예측 결과 샘플 ===")
        
        # Get random samples
        sample_indices = np.random.choice(len(self.val_encodings['input_ids']), num_samples)
        
        for idx in sample_indices:
            # Get original text
            text = self.data['sentence'].iloc[idx]
            true_label = self.val_labels.iloc[idx]
            
            # Prepare input
            inputs = {
                key: torch.tensor([self.val_encodings[key][idx]]).to(self.model.device)
                for key in self.val_encodings
            }
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                pred_label = outputs.logits.argmax(dim=-1).item()
            
            # Get prediction probability
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence = probs[0][pred_label].item()
            
            # Print results
            print("\n---")
            print(f"텍스트: {text[:100]}...")  # Show first 100 chars
            print(f"실제 감성: {'긍정' if true_label == 1 else '부정'}")
            print(f"예측 감성: {'긍정' if pred_label == 1 else '부정'} (확률: {confidence:.2%})")
            print(f"예측 결과: {'✓' if pred_label == true_label else '✗'}")

    def train(self, train_args: dict = None, num_unfrozen_layers: int = 2):
        """
        Train the model with given arguments
        Args:
            train_args (dict): Training arguments including learning_rate, num_train_epochs, batch_size
            num_unfrozen_layers (int): Number of layers to unfreeze for training
        """
        if train_args is None:
            train_args = {}
        
        self.load_data()
        self.prepare_data()
        
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

    def evaluate(self, normalize_cm: bool = True):
        eval_result = self.trainer.evaluate()
        print(f"모델 평가 결과: {eval_result}")
        
        # Confusion matrix 생성 및 저장
        self.save_confusion_matrix(normalize=normalize_cm)
        
        return eval_result

    def save_model(self, save_path: Path = None):
        """
        Save the model to the specified path
        Args:
            save_path (Path, optional): Path to save the model. If None, uses self.model_dir
        """
        if save_path is None:
            save_path = self.model_dir
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"모델을 {save_path}에 저장했습니다.")


    def save_confusion_matrix(self, normalize: bool = False):
        """Confusion matrix 생성 및 저장"""
        print("\n=== Confusion Matrix 생성 ===")
        
        # Matplotlib 백엔드를 Agg로 설정 (GUI 없이 동작)
        import matplotlib
        matplotlib.use('Agg')
        
        # 예측 및 실제 레이블 가져오기
        preds = self.trainer.predict(self.val_dataset).predictions.argmax(-1)
        true_labels = self.val_labels
        
        # Confusion matrix 계산
        cm = confusion_matrix(true_labels, preds, normalize='true' if normalize else None)
        
        # 시각화
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='.2%' if normalize else 'd', 
            cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive']
        )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        
        # 저장
        cm_file = os.path.join(self.model_dir, f'confusion_matrix{"_normalized" if normalize else ""}.png')
        plt.savefig(cm_file)
        plt.close()
        
        print(f"Confusion matrix를 {cm_file}에 저장했습니다.")

    def load_model(self):
        """
        Load the pretrained model from cache or download if not exists
        """
        try:
            # 먼저 로컬 pretrained 디렉토리에서 로드 시도
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.pretrained_model_dir,
                num_labels=2
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_dir)
            print(f"Loaded model from {self.pretrained_model_dir}")
        except:
            # 로컬에 없으면 다운로드 후 저장
            print(f"Downloading model {self.pretrained_model_name}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.pretrained_model_name,
                num_labels=2
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
            
            # pretrained 모델 저장
            self.model.save_pretrained(self.pretrained_model_dir)
            self.tokenizer.save_pretrained(self.pretrained_model_dir)
            print(f"Saved pretrained model to {self.pretrained_model_dir}")