import torch
from typing import List, Dict, Union
from pathlib import Path

class ModelInferencer:
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def predict(self, texts: Union[str, List[str]], batch_size: int = 32) -> List[Dict]:
        """텍스트 또는 텍스트 리스트에 대한 예측 수행"""
        if isinstance(texts, str):
            texts = [texts]
        
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = self._predict_batch(batch_texts)
            results.extend(batch_results)
        
        return results
    
    def _predict_batch(self, texts: List[str]) -> List[Dict]:
        """배치 단위 예측"""
        # 토크나이징
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        # GPU로 이동
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 예측
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            confidences = torch.max(probs, dim=-1)[0]
        
        # 결과 정리
        results = []
        for text, pred, conf, prob in zip(texts, predictions, confidences, probs):
            results.append({
                'text': text,
                'prediction': pred.item(),
                'confidence': conf.item(),
                'probabilities': prob.cpu().numpy().tolist()
            })
        
        return results 