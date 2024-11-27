from transformers import BertTokenizer, BertForSequenceClassification
from .base_model import BaseSentimentModel

class KcBERTModel(BaseSentimentModel):
    def __init__(self, data_file, model_dir, pretrained_model_name, pretrained_model_dir):
        super().__init__(data_file, model_dir, pretrained_model_name, pretrained_model_dir)
        self.load_model()
        
    def load_model(self):
        try:
            self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_dir)
            self.model = BertForSequenceClassification.from_pretrained(self.pretrained_model_dir)
            print(f"모델을 {self.pretrained_model_dir}에서 로드했습니다.")
        except:
            print(f"사전 학습된 모델을 {self.pretrained_model_name}에서 다운로드합니다.")
            self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_name)
            self.model = BertForSequenceClassification.from_pretrained(
                self.pretrained_model_name, 
                num_labels=2
            )
            # pretrained 모델 저장
            self.model.save_pretrained(self.pretrained_model_dir)
            self.tokenizer.save_pretrained(self.pretrained_model_dir) 