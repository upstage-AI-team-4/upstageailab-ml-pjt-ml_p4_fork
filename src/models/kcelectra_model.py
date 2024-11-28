from transformers import AutoTokenizer, BertForSequenceClassification
from .base_model import BaseSentimentModel
import torch
import numpy as np

class KcELECTRAModel(BaseSentimentModel):
    def __init__(self, data_file, model_dir, pretrained_model_name, pretrained_model_dir):
        super().__init__(data_file=data_file, 
                        pretrained_model_name=pretrained_model_name)
        self.pretrained_model_dir = pretrained_model_dir
        self.load_model()
        
    def load_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_dir)
            self.model = BertForSequenceClassification.from_pretrained(self.pretrained_model_dir)
            print(f"모델을 {self.pretrained_model_dir}에서 로드했습니다.")
        except Exception as e:
            print(f"모델을 로드하는 데 실패했습니다: {e}")
            print(f"사전 학습된 모델을 {self.pretrained_model_name}에서 다운로드합니다.")
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
            self.model = BertForSequenceClassification.from_pretrained(
                self.pretrained_model_name, 
                num_labels=2
            )
            # pretrained 모델 저장
            self.model.save_pretrained(self.pretrained_model_dir)
            self.tokenizer.save_pretrained(self.pretrained_model_dir) 

    def predict(self, dataset) -> np.ndarray:
        """예측 수행"""
        self.model.eval()
        self.model.to(self.device)
        
        predictions = []
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # 배치 데이터를 GPU로 이동
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # 예측 수행
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # 예측값 추출
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())
        
        return np.array(predictions) 