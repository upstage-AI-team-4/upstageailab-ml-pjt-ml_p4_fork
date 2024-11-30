import re
import emoji

def clean_text(text: str) -> str:
    """Clean text by removing special characters and emojis"""
    # 이모지 제거
    text = emoji.replace_emoji(text, '')
    
    # HTML 태그 제거
    text = re.sub(r'<[^>]+>', '', text)
    
    # 특수 문자 및 숫자 제거 (한글, 영문, 공백만 남김)
    text = re.sub(r'[^가-힣a-zA-Z\s]', '', text)
    
    # 연속된 공백 제거
    text = re.sub(r'\s+', ' ', text)
    
    # 앞뒤 공백 제거
    text = text.strip()
    
    return text

def preprocess_text(text: str) -> str:
    """Preprocess text for NSMC dataset"""
    # 기본 클리닝
    text = clean_text(text)
    
    # 빈 문자열 처리
    if not text:
        text = "빈 텍스트"
        
    return text
