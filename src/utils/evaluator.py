import torch
from typing import Dict, List, Tuple
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score

class ModelEvaluator:
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

    def evaluate_dataset(self, data_module, n_samples: int = 5) -> Dict:
        """전체 데이터셋에 대한 평가 수행"""
        metrics = {}
        
        # 전체 검증 세트에 대한 메트릭 계산
        predictions, labels, confidences = self._get_predictions(data_module.val_dataloader())
        
        # 기본 메트릭 계산
        metrics['accuracy'] = self._calculate_accuracy(predictions, labels)
        metrics['avg_confidence'] = np.mean(confidences)
        
        # F1 스코어 계산
        metrics['f1'] = f1_score(labels, predictions, average='weighted')
        
        # 신뢰도 구간별 정확도
        confidence_metrics = self._calculate_confidence_bins(predictions, labels, confidences)
        metrics.update(confidence_metrics)
        
        # 샘플 예측 결과
        sample_predictions = self._get_sample_predictions(data_module.val_dataset, n_samples)
        metrics['sample_predictions'] = sample_predictions
        
        return metrics
    
    def _get_predictions(self, dataloader) -> Tuple[List, List, List]:
        """데이터로더로부터 예측 수행"""
        predictions, labels, confidences = [], [], []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**{k: v for k, v in batch.items() if k != 'labels'})
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                preds = torch.argmax(logits, dim=-1)
                confs = torch.max(probs, dim=-1)[0]
                
                predictions.extend(preds.cpu().numpy())
                labels.extend(batch['labels'].cpu().numpy())
                confidences.extend(confs.cpu().numpy())
                
        return predictions, labels, confidences
    
    def _calculate_accuracy(self, predictions: List, labels: List) -> float:
        """정확도 계산"""
        return sum(p == l for p, l in zip(predictions, labels)) / len(labels)
    
    def _calculate_confidence_bins(self, predictions: List, labels: List, confidences: List) -> Dict:
        """신뢰도 구간별 정확도 계산"""
        bins = {}
        confidence_bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
        
        for low, high in confidence_bins:
            mask = [(c >= low and c < high) for c in confidences]
            if sum(mask) > 0:
                bin_acc = sum(p == l for p, l, m in zip(predictions, labels, mask) if m) / sum(mask)
                bin_count = sum(mask)
                bins[f'confidence_{int(low*100)}_{int(high*100)}'] = {
                    'accuracy': bin_acc,
                    'count': bin_count
                }
        
        return {'confidence_bins': bins}
    
    def _get_sample_predictions(self, dataset, n_samples=5):
        """샘플 데이터에 대한 예측 결과 반환"""
        indices = torch.randperm(len(dataset))[:n_samples].tolist()
        predictions = []
        
        self.model.eval()
        with torch.no_grad():
            for idx in indices:
                text = dataset.texts[idx]
                true_label = dataset.labels[idx]
                
                sample = dataset[idx]
                inputs = {
                    'input_ids': sample['input_ids'].unsqueeze(0).to(self.model.device),
                    'attention_mask': sample['attention_mask'].unsqueeze(0).to(self.model.device)
                }
                
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                pred_label = torch.argmax(logits, dim=-1).item()
                confidence = probs[0][pred_label].item()
                
                predictions.append({
                    'text': text,
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'confidence': confidence
                })
        
        return predictions