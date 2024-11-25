import streamlit as st
import torch
import pandas as pd
from pathlib import Path
from models.kcbert_sentiment_model import KcBERTSentimentModel
from models.kcelectra_sentiment_model import KcELECTRASentimentModel
import plotly.graph_objects as go

class SentimentAnalysisApp:
    def __init__(self):
        self.data_dir = Path("data")
        self.models = {}
        self.current_dataset = None
        
    def load_available_datasets(self):
        """데이터 폴더에서 사용 가능한 CSV 파일 목록 가져오기"""
        csv_files = []
        for subdir in ['raw', 'processed']:
            data_path = self.data_dir / subdir
            if data_path.exists():
                csv_files.extend(list(data_path.glob("*.csv")))
        
        return {f.stem: f for f in csv_files}

    def initialize_models(self, dataset_path):
        """선택된 데이터셋으로 모델 초기화"""
        self.models = {
            'KcBERT': {
                'model': KcBERTSentimentModel(
                    data_file=str(dataset_path),
                    model_dir="models/kcbert"
                ),
                'color': '#1f77b4'
            },
            'KcELECTRA': {
                'model': KcELECTRASentimentModel(
                    data_file=str(dataset_path),
                    model_dir="models/kcelectra"
                ),
                'color': '#ff7f0e'
            }
        }

    def show_dataset_info(self, dataset_path):
        """데이터셋 정보 표시"""
        df = pd.read_csv(dataset_path)
        st.markdown("### 데이터셋 정보")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("전체 데이터 수", f"{len(df):,}")
        with col2:
            if 'label' in df.columns:
                positive = (df['label'] == 1).sum()
                st.metric("긍정 데이터 수", f"{positive:,}")
        with col3:
            if 'label' in df.columns:
                negative = (df['label'] == 0).sum()
                st.metric("부정 데이터 수", f"{negative:,}")
        
        # 샘플 데이터 표시
        st.markdown("### 데이터 샘플")
        st.dataframe(df.head())

    def predict_sentiment(self, text: str, model_name: str):
        """감성 분석 예측 수행"""
        model = self.models[model_name]['model']
        
        inputs = model.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = model.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_label = outputs.logits.argmax(dim=-1).item()
            confidence = probs[0][pred_label].item()
            
        return {
            'label': '긍정' if pred_label == 1 else '부정',
            'confidence': confidence,
            'probabilities': probs[0].tolist()
        }

    def create_gauge_chart(self, value: float, color: str):
        """게이지 차트 생성"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 100], 'color': "white"}
                ]
            },
            number={'suffix': "%"}
        ))
        fig.update_layout(height=200)
        return fig

    def run(self):
        """Streamlit 앱 실행"""
        st.title("한국어 감성 분석 데모")
        
        # 데이터셋 선택
        datasets = self.load_available_datasets()
        dataset_name = st.selectbox(
            "데이터셋을 선택하세요:",
            options=list(datasets.keys()),
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        dataset_path = datasets[dataset_name]
        
        # 선택된 데이터셋이 변경되면 모델 재초기화
        if self.current_dataset != dataset_path:
            self.current_dataset = dataset_path
            self.initialize_models(dataset_path)
            self.show_dataset_info(dataset_path)
        
        st.markdown("""
        ### 텍스트 감성 분석
        선택한 데이터셋으로 학습된 모델을 사용하여 텍스트의 감성을 분석합니다.
        """)
        
        # 텍스트 입력
        text = st.text_area(
            "분석할 텍스트를 입력하세요:",
            height=100,
            placeholder="예: 이 영화 정말 재미있게 봤어요. 배우들의 연기도 훌륭했고 스토리도 좋았습니다."
        )
        
        # 모델 선택
        col1, col2 = st.columns([2, 1])
        with col1:
            model_name = st.selectbox(
                "사용할 모델을 선택하세요:",
                list(self.models.keys())
            )
        
        with col2:
            analyze_button = st.button("분석하기", use_container_width=True)
        
        if analyze_button:
            if text:
                with st.spinner('분석 중...'):
                    result = self.predict_sentiment(text, model_name)
                    
                    st.markdown(f"### 분석 결과")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**예측된 감성:** {result['label']}")
                    with col2:
                        st.markdown(f"**신뢰도:** {result['confidence']:.2%}")
                    
                    st.plotly_chart(
                        self.create_gauge_chart(
                            result['confidence'],
                            self.models[model_name]['color']
                        ),
                        use_container_width=True
                    )
                    
                    st.markdown("### 확률 분포")
                    probs_df = pd.DataFrame({
                        '감성': ['부정', '긍정'],
                        '확률': result['probabilities']
                    })
                    st.bar_chart(probs_df.set_index('감성'))
                    
            else:
                st.warning("텍스트를 입력해주세요.")

if __name__ == "__main__":
    app = SentimentAnalysisApp()
    app.run() 