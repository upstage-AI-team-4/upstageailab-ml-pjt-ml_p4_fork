from transformers import BertTokenizer, BertForSequenceClassification
from .base_model import BaseSentimentModel

class KcBERTSentimentModel(BaseSentimentModel):
    def __init__(self, data_file, model_dir='./models/kcbert'):
        super().__init__(data_file, model_dir)
        self.model_name = 'beomi/kcbert-base'
        self.load_model()
        
    def load_model(self):
        try:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_dir)
            self.model = BertForSequenceClassification.from_pretrained(self.model_dir)
            print(f"모델을 {self.model_dir}에서 로드했습니다.")
        except:
            print(f"사전 학습된 모델을 {self.model_name}에서 다운로드합니다.")
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = BertForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=2
            )
            self.save_model() 