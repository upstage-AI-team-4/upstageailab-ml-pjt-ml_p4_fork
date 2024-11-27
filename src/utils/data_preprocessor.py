# utils/data_preprocessor.py
import pandas as pd
import re
# from konlpy.tag import Mecab
import emoji
import os
from konlpy.tag import Okt
from kiwipiepy import Kiwi
from tqdm import tqdm
from pathlib import Path
from typing import List


def find_files_with_extension(directory: str, extension: str, recursive: bool = False) -> List[Path]:
    """
    Finds all files with a given extension in a directory.
    Args:
        directory (str): The directory to search for files.
        extension (str): The extension to search for.
        recursive (bool): Whether to search recursively.
    Returns:
        List[Path]: A list of Path objects representing the found files.
    """
    folder = Path(directory)
    if recursive:
        files = [file for file in folder.rglob(f'*{extension}') if file.is_file()]
    else:
        files = list(folder.glob(f'*{extension}'))

    print(f"Found {len(files)} files with extension {extension} in {directory}.")
    return files
def load_txt_as_dataframe(file_path: Path, delimiter: str = ',') -> pd.DataFrame:
    """
    Loads a text file as a pandas DataFrame.
    Args:
        file_path (Path): The path to the text file.
        delimiter (str): The delimiter used in the text file.
    Returns:
        pd.DataFrame: A pandas DataFrame containing the data from the text file.
    """
    file = Path(file_path)
    if not file.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")
    
    return pd.read_csv(file, delimiter = delimiter)


class DataPreprocessor:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        os.makedirs(self.processed_dir, exist_ok=True)
        self.output_file = None
        # 분석기 인스턴스 생성
        self.okt = Okt()
        self.kiwi = Kiwi()
        print(f'\n<<< Data preprocessing: {data_dir}')
        print(f'===분석기 인스턴스 생성 완료...')
        # self.mecab = Mecab()

    # def load_data(self):
    #     self.data = pd.read_csv(self.input_file)
    #     print(f"{self.input_file}에서 데이터를 로드했습니다.")
        
    def prep_naver_data(self, sampling_rate: float = 1.0) -> pd.DataFrame:
        sampling_rate_str = str(sampling_rate).replace('.', '_')
        self.output_file = self.processed_dir / f'naver_movie_review_sampling_{sampling_rate_str}.csv'
        if os.path.exists(self.output_file):
            print(f'{self.output_file} 파일이 이미 존재합니다.')
            self.data = pd.read_csv(self.output_file)
            return self.data

        
        data_naver_dir = self.raw_dir / 'naver_movie_review'
        data_naver_train = data_naver_dir / 'ratings_train.txt'
        data_naver_test = data_naver_dir / 'ratings_test.txt'
        print(f'===네이버 영화 리뷰 데이터 로드 시작... Sampling Rate: {sampling_rate}\n from {data_naver_dir}')

        train_df = load_txt_as_dataframe(data_naver_train, delimiter = '\t')
        test_df = load_txt_as_dataframe(data_naver_test, delimiter = '\t')
        train_df['is_test'] = 0
        test_df['is_test'] = 1
        print(f'데이터 로드 완료...{len(train_df)}개, {len(test_df)}개')
        if sampling_rate < 1.0:
            train_df = train_df.sample(frac=sampling_rate, random_state=42)
            test_df = test_df.sample(frac=sampling_rate, random_state=42)
            print(f'샘플링 완료...{len(train_df)}개, {len(test_df)}개')
        naver_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
        naver_df.to_csv(self.output_file, index=False, encoding='utf-8')
        print(naver_df.head())
        self.data = naver_df
        return self.data

    def clean_text(self, text):
        """Clean text by removing emojis, URLs, special characters, and numbers"""
        # NaN 값 처리
        if pd.isna(text):
            return ''
        
        text = str(text)  # 숫자 등의 다른 타입을 문자열로 변환
        text = emoji.replace_emoji(text, replace='')
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def prep_column_name(self):
        print(f'text, label 컬럼 확인...')
        if 'text' not in self.data.columns:
            if 'document' in self.data.columns:
                self.data.rename(columns={'document': 'text'}, inplace=True)
                print("text 컬럼이 없습니다. document 컬럼을 사용합니다.")
            else:
                raise ValueError("text 또는 document 컬럼이 없습니다.")
    # def tokenize(self, text):
    #     return self.mecab.morphs(text)

    def preprocess(self):
        print(f'===데이터 전처리 시작...')
        self.prep_column_name()
        
        # NaN 값 처리
        print("===NaN 값 확인 및 처리...")
        print(f"전처리 전 NaN 값 개수: {self.data['text'].isna().sum()}")
        self.data = self.data.dropna(subset=['text'])
        print(f"전처리 후 NaN 값 개수: {self.data['text'].isna().sum()}")
        
        print("===텍스트 정제 시작...")
        self.data['clean_text'] = self.data['text'].apply(self.clean_text)
        
        print("===토큰화 시작...")
        self.data['tokens'] = self.data['clean_text'].apply(self.combined_tokenize)
        
        # 빈 문자열 제거
        print("빈 문자열 확인 및 처리...")
        empty_texts = self.data['clean_text'].str.strip() == ''
        print(f"빈 문자열 개수: {empty_texts.sum()}")
        self.data = self.data[~empty_texts]
        
        # 저장
        prep_file = self.processed_dir / f'preped_{self.output_file.name}'
        self.data.to_csv(prep_file, index=False)
        print(f"전처리된 데이터를 {prep_file}에 저장했습니다.")
        print(f"최종 데이터 크기: {len(self.data)} 행")
        return prep_file

    def combined_tokenize(self, text):
        """Tokenize text using both Okt and Kiwi tokenizers"""
        if pd.isna(text) or text.strip() == '':
            return []
            
        # Okt 토큰화 및 품사 태깅
        okt_tokens = self.okt.pos(text)
        # Kiwi 토큰화 및 품사 태깅
        kiwi_tokens = [(token.form, token.tag) for token in self.kiwi.tokenize(text)]
        
        # 결과를 딕셔너리로 변환
        okt_dict = dict(okt_tokens)
        kiwi_dict = dict(kiwi_tokens)
        
        # 토큰 리스트 생성
        tokens = list(set(list(okt_dict.keys()) + list(kiwi_dict.keys())))
        
        # 품사 정보를 결합하여 새로운 리스트 생성
        combined_tokens = []
        for token in tokens:
            if token in okt_dict and token in kiwi_dict:
                # 두 분석기의 품사가 같은 경우 사용
                if okt_dict[token] == kiwi_dict[token]:
                    combined_tokens.append((token, okt_dict[token]))
                else:
                    # 품사가 다른 경우 Okt의 품사 사용
                    combined_tokens.append((token, okt_dict[token]))
            elif token in okt_dict:
                combined_tokens.append((token, okt_dict[token]))
            elif token in kiwi_dict:
                # Kiwi의 품사 태그를 Okt의 태그로 매핑 (필요 시)
                kiwi_pos = kiwi_dict[token]
                # 간단한 매핑 예시
                tag_map = {
                    'NNG': 'Noun',
                    'NNP': 'Noun',
                    'VV': 'Verb',
                    'VA': 'Adjective',
                    'MAG': 'Adverb',
                    'JKS': 'Josa',
                    'EF': 'Eomi',
                    'SW': 'Punctuation'
                }
                mapped_pos = tag_map.get(kiwi_pos, 'Unknown')
                combined_tokens.append((token, mapped_pos))
            else:
                combined_tokens.append((token, 'Unknown'))
        
        return combined_tokens

