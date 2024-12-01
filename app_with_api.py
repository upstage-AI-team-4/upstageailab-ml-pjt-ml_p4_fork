import streamlit as st
import requests
import plotly.graph_objects as go
from typing import Dict, Any

API_URL = "http://localhost:8000"

def get_production_models() -> Dict[str, Any]:
    """Get list of production models from API"""
    response = requests.get(f"{API_URL}/models")
    return response.json()

def predict_sentiment(text: str, model_id: str) -> Dict[str, Any]:
    """Send prediction request to API"""
    response = requests.post(
        f"{API_URL}/predict",
        json={"text": text, "model_id": model_id}
    )
    return response.json()

def display_model_info(model_info: Dict[str, Any]):
    """Display model information in sidebar"""
    st.sidebar.subheader("Selected Model Info")
    st.sidebar.write(f"Model: {model_info['run_name']}")
    st.sidebar.write(f"Stage: {model_info['stage']}")
    
    st.sidebar.subheader("Model Metrics")
    for metric, value in model_info['metrics'].items():
        st.sidebar.metric(metric, f"{value:.4f}")
    
    st.sidebar.write(f"Registered: {model_info['timestamp']}")

def main():
    st.set_page_config(
        page_title="Sentiment Analysis Demo",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("한국어 감성 분석 데모")
    st.write("텍스트를 입력하면 긍정/부정을 판단합니다.")
    
    try:
        # Get available production models
        production_models = get_production_models()
        
        if not production_models:
            st.error("No production models found. Please train and promote a model first.")
            st.stop()
        
        # Model selection
        model_options = {
            f"{model['run_name']} ({model['timestamp']})": model 
            for model in production_models
        }
        
        selected_model_name = st.sidebar.selectbox(
            "Select Production Model",
            options=list(model_options.keys())
        )
        
        selected_model_info = model_options[selected_model_name]
        display_model_info(selected_model_info)
        
        # Text input
        text = st.text_area(
            "분석할 텍스트를 입력하세요:",
            height=100,
            help="여러 줄의 텍스트를 입력할 수 있습니다."
        )
        
        if st.button("분석하기", type="primary"):
            if not text:
                st.warning("텍스트를 입력해주세요.")
                return
            
            with st.spinner("분석 중..."):
                result = predict_sentiment(text, selected_model_info['run_id'])
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("분석 결과")
                    st.metric("감성", result["sentiment"])
                    st.metric("확신도", f"{result['confidence']:.1%}")
                
                with col2:
                    st.subheader("확률 분포")
                    fig = go.Figure(go.Bar(
                        x=['부정', '긍정'],
                        y=result['probabilities'],
                        marker_color=['#ff9999', '#99ff99']
                    ))
                    fig.update_layout(
                        title="감성 분석 확률 분포",
                        yaxis_title="확률",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Additional analysis
                st.subheader("상세 분석")
                with st.expander("자세히 보기"):
                    st.write("입력 텍스트 길이:", len(text))
                    st.write("토큰 수:", result.get('num_tokens', 'N/A'))
                    st.json({
                        "prediction": {
                            "label": result["sentiment"],
                            "confidence": f"{result['confidence']:.4f}",
                            "probabilities": {
                                "negative": f"{result['probabilities'][0]:.4f}",
                                "positive": f"{result['probabilities'][1]:.4f}"
                            }
                        }
                    })
                    
    except requests.exceptions.ConnectionError:
        st.error("API 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.")
    except Exception as e:
        st.error(f"오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    main() 