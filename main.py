
# main.py
from utils.twitter_collector import TwitterCollector
from utils.data_preprocessor import DataPreprocessor
from pathlib import Path
from typing import List
import pandas as pd
from models.model_factory import ModelFactory


def main():
    # 1. 데이터 수집
    # collector = TwitterCollector(api_key='YOUR_API_KEY',
    #                              api_secret='YOUR_API_SECRET',
    #                              access_token='YOUR_ACCESS_TOKEN',
    #                              access_secret='YOUR_ACCESS_SECRET')
    # collector.collect_tweets(query='감정 OR 기분 OR 행복 OR 슬픔', max_tweets=1000,
    #                          output_file='data/raw/tweets.csv')
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'data' 
    
    data_prep= DataPreprocessor(data_dir=data_dir)
    data_prep.prep_naver_data(sampling_rate=0.1)
    preped_file_path = data_prep.preprocess()
    
    # 3. 모델 학습 및 평가
     # 모델 이름에 따라 동적으로 모델 생성
    model_name = "KcBERT" ##"KcELECTRA"  # 또는 "KcBERT"
    model_dir = Path('e:/models')
    #model_kcbert = model_dir / 'KcBERT'
    model_dir = model_dir / model_name
    factory = ModelFactory()
    model = factory.get_model(model_name, data_file = preped_file_path, model_dir = model_dir)
    model.train(num_unfrozen_layers=3)
    model.evaluate()

if __name__ == '__main__':
    main()
