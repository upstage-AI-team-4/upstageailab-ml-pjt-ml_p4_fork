import torch
from typing import List, Dict, Union, Any
from pathlib import Path

class ModelInferencer:
    def __init__(self, model, tokenizer):
        """모델 추론기 초기화
        
        Args:
            model: 학습된 모델
            tokenizer: 토크나이저
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
    def predict(self, texts: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """텍스트 감정 분석
        
        Args:
            texts: 분석할 텍스트 또는 텍스트 리스트
            
        Returns:
            List[Dict]: 예측 결과 리스트 (prediction, confidence)
            
        Raises:
            ValueError: 입력 텍스트가 비어있는 경우
            RuntimeError: 모델 추론 중 오류 발생
        """
        if not texts:
            raise ValueError("Input texts cannot be empty")
            
        # 단일 텍스트를 리스트로 변환
        if isinstance(texts, str):
            texts = [texts]
            
        # 배치 크기로 나누어 처리
        batch_size = 32
        results = []
        
        try:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_results = self._predict_batch(batch_texts)
                results.extend(batch_results)
        except Exception as e:
            raise RuntimeError(f"Error during model inference: {str(e)}")
            
        return results
    
    def _predict_batch(self, batch_texts: List[str]) -> List[Dict[str, Any]]:
        """배치 단위 예측
        
        Args:
            batch_texts: 배치 텍스트 리스트
            
        Returns:
            List[Dict]: 예측 결과 리스트
        """
        # 모델을 평가 모드로 설정
        self.model.eval()
        
        # 텍스트 전처리
        try:
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
        except Exception as e:
            raise RuntimeError(f"Error during tokenization: {str(e)}")
        
        # token_type_ids 처리
        try:
            import inspect
            forward_params = inspect.signature(self.model.forward).parameters
            if 'token_type_ids' not in forward_params and 'token_type_ids' in inputs:
                del inputs['token_type_ids']
        except Exception as e:
            print(f"Warning: Error checking model signature: {e}")
            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']
        
        # 입력을 현재 디바이스로 이동
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 예측
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                confidences = torch.max(probs, dim=-1).values
        except Exception as e:
            raise RuntimeError(f"Error during model inference: {str(e)}")
        
        # 결과 변환
        results = []
        for pred, conf in zip(predictions.cpu().numpy(), confidences.cpu().numpy()):
            results.append({
                "prediction": int(pred),
                "confidence": float(conf)
            })
        
        return results 