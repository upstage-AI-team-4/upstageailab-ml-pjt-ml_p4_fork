import streamlit as st
from src.config import Config
from src.utils.mlflow_utils import MLflowModelManager
from transformers import AutoTokenizer
import torch
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import time

def load_model_and_tokenizer(model_info, config):
    """Load selected model and tokenizer"""
    model_manager = MLflowModelManager(config)
    
    # Load model
    model = model_manager.load_production_model(config.project['model_name'])
    if model is None:
        st.error("Failed to load the model. Please check if the model files exist.")
        st.stop()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_info['params']['pretrained_model'])
    
    return model, tokenizer

def predict_sentiment(text: str, model, tokenizer, config):
    """Predict sentiment for given text"""
    # Tokenize
    inputs = tokenizer(
        text,
        padding='max_length',
        max_length=config.training_config['max_length'],
        truncation=True,
        return_tensors='pt'
    )
    
    # Check if model needs token_type_ids
    try:
        import inspect
        forward_params = inspect.signature(model.forward).parameters
        if 'token_type_ids' not in forward_params and 'token_type_ids' in inputs:
            del inputs['token_type_ids']
    except Exception as e:
        print(f"Warning: Error checking model signature: {e}")
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
    
    # Move inputs to the same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_label = torch.argmax(outputs.logits, dim=-1).item()
        confidence = probs[0][pred_label].item()
    
    return pred_label, confidence, probs[0].cpu().numpy()

def display_model_info(model_info):
    """Display model information in sidebar"""
    st.sidebar.subheader("Selected Model Info")
    st.sidebar.write(f"Model: {model_info['run_name']}")
    st.sidebar.write(f"Stage: {model_info['stage']}")
    
    st.sidebar.subheader("Model Metrics")
    for metric, value in model_info['metrics'].items():
        st.sidebar.metric(metric, f"{value:.4f}")
    
    st.sidebar.write(f"Registered: {model_info['timestamp']}")

def initialize_session_state():
    """Initialize session state variables"""
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'total_predictions' not in st.session_state:
        st.session_state.total_predictions = 0
    if 'positive_count' not in st.session_state:
        st.session_state.positive_count = 0
    if 'negative_count' not in st.session_state:
        st.session_state.negative_count = 0

def update_statistics(sentiment: str, confidence: float):
    """Update prediction statistics"""
    st.session_state.total_predictions += 1
    if sentiment == "긍정":
        st.session_state.positive_count += 1
    else:
        st.session_state.negative_count += 1

def add_to_history(text: str, sentiment: str, confidence: float, probabilities: list, model_id: int):
    """Add prediction to history"""
    st.session_state.history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text": text,
        "sentiment": sentiment,
        "confidence": confidence,
        "negative_prob": probabilities[0],
        "positive_prob": probabilities[1],
        "model_id": model_id
    })

def display_statistics():
    """Display prediction statistics"""
    st.sidebar.subheader("Prediction Statistics")
    total = st.session_state.total_predictions
    if total > 0:
        pos_ratio = (st.session_state.positive_count / total) * 100
        neg_ratio = (st.session_state.negative_count / total) * 100
        
        col1, col2, col3 = st.sidebar.columns(3)
        col1.metric("Total", total)
        col2.metric("긍정", f"{pos_ratio:.1f}%")
        col3.metric("부정", f"{neg_ratio:.1f}%")

def display_model_management(model_manager, model_name: str):
    """Display model management interface"""
    st.subheader("모델 관리")
    
    # Get all model versions
    models = model_manager.load_model_info()
    if not models:
        st.warning("등록된 모델이 없습니다.")
        return
    
    # Create DataFrame for better display
    df = pd.DataFrame(models)
    df['model_id'] = df.index + 1
    
    # Reorder columns
    columns = [
        'model_id', 'run_name', 'stage', 'metrics', 
        'timestamp', 'version', 'run_id'
    ]
    df = df[columns]
    
    # Format metrics column
    df['metrics'] = df['metrics'].apply(
        lambda x: ', '.join([f"{k}: {v:.4f}" for k, v in x.items()])
    )
    
    # Add styling
    def color_stage(val):
        colors = {
            'Production': '#99ff99',
            'Staging': '#ffeb99',
            'Archived': '#ff9999'
        }
        return f'background-color: {colors.get(val, "#ffffff")}; color: black'
    
    styled_df = df.style.applymap(
        color_stage,
        subset=['stage']
    )
    
    # Display models table
    st.dataframe(
        styled_df,
        column_config={
            "model_id": "모델 ID",
            "run_name": "모델 이름",
            "stage": "스테이지",
            "metrics": "성능 지표",
            "timestamp": "등록 시간",
            "version": "버전",
            "run_id": "실행 ID"
        },
        hide_index=True,
        use_container_width=True
    )
    
    # Model management controls
    st.markdown("---")
    st.subheader("스테이지 관리")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_model_id = st.selectbox(
            "관리할 모델 선택",
            options=df['model_id'].tolist(),
            format_func=lambda x: f"Model {x}: {df[df['model_id']==x]['run_name'].iloc[0]}"
        )
        
        selected_model = df[df['model_id'] == selected_model_id].iloc[0]
        
        st.write("현재 정보:")
        st.write(f"- 모델: {selected_model['run_name']}")
        st.write(f"- 스테이지: {selected_model['stage']}")
        st.write(f"- 버전: {selected_model['version']}")
    
    with col2:
        new_stage = st.selectbox(
            "변경할 스테이지",
            options=['Staging', 'Production', 'Archived']
        )
        
        if st.button("스테이지 변경", type="primary"):
            try:
                if new_stage == 'Production':
                    model_manager.promote_to_production(
                        model_name,
                        selected_model['version']
                    )
                elif new_stage == 'Archived':
                    model_manager.archive_model(
                        model_name,
                        selected_model['version']
                    )
                elif new_stage == 'Staging':
                    model_manager.promote_to_staging(
                        model_name,
                        selected_model['run_id']
                    )
                
                st.success(f"모델 스테이지가 {new_stage}로 변경되었습니다.")
                time.sleep(5) 
                st.rerun()
                
            except Exception as e:
                st.error(f"스테이지 변경 중 오류가 발생했습니다: {str(e)}")

def main():
    st.set_page_config(
        page_title="Sentiment Analysis Demo",
        page_icon="🤖",
        layout="wide"
    )
    
    initialize_session_state()
    
    # Initialize config and model manager
    config = Config()
    model_manager = MLflowModelManager(config)
    
    # Create tabs
    tab_predict, tab_manage = st.tabs(["감성 분석", "모델 관리"])
    
    with tab_predict:
        st.title("한국어 감성 분석 데모")
        st.write("텍스트를 입력하면 긍정/부정을 판단합니다.")
        
        # Get production models
        production_models = model_manager.get_production_models()
        
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
        
        # Get model_id from selected model
        model_id = production_models.index(selected_model_info) + 1
        
        # Display statistics
        display_statistics()
        
        # Cache model loading
        @st.cache_resource
        def load_resources(model_info):
            return load_model_and_tokenizer(model_info, config)
        
        model, tokenizer = load_resources(selected_model_info)
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
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
                    pred_label, confidence, probs = predict_sentiment(
                        text, model, tokenizer, config
                    )
                    
                    # Update statistics and history
                    sentiment = "긍정" if pred_label == 1 else "부정"
                    update_statistics(sentiment, confidence)
                    add_to_history(text, sentiment, confidence, probs, model_id)
                    
                    # Display results
                    st.subheader("분석 결과")
                    col_result1, col_result2 = st.columns(2)
                    
                    with col_result1:
                        st.metric("감성", sentiment)
                        st.metric("확신도", f"{confidence:.1%}")
                    
                    with col_result2:
                        fig = go.Figure(go.Bar(
                            x=['부정', '긍정'],
                            y=probs,
                            marker_color=['#ff9999', '#99ff99']
                        ))
                        fig.update_layout(
                            title="감성 분석 확률 분포",
                            yaxis_title="확률",
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("분석 상세 정보")
            with st.expander("자세히 보기", expanded=True):
                st.write("입력 텍스트 길이:", len(text) if text else 0)
                st.write("토큰 수:", len(tokenizer.encode(text)) if text else 0)
                if text:
                    st.json({
                        "prediction": {
                            "label": sentiment if 'sentiment' in locals() else None,
                            "confidence": f"{confidence:.4f}" if 'confidence' in locals() else None,
                            "probabilities": {
                                "negative": f"{probs[0]:.4f}" if 'probs' in locals() else None,
                                "positive": f"{probs[1]:.4f}" if 'probs' in locals() else None
                            }
                        }
                    })
        
        # History section
        st.markdown("---")
        st.subheader("분석 히스토리")
        
        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history)
            df = df.sort_values('timestamp', ascending=False)
            
            # Add styling
            def color_sentiment(val):
                color = '#99ff99' if val == '긍정' else '#ff9999'
                return f'background-color: {color}; color: black'
            
            styled_df = df.style.applymap(
                color_sentiment, 
                subset=['sentiment']
            ).format({
                'confidence': '{:.1%}',
                'negative_prob': '{:.4f}',
                'positive_prob': '{:.4f}'
            })
            
            st.dataframe(
                styled_df,
                column_config={
                    "timestamp": "시간",
                    "text": "텍스트",
                    "sentiment": "감성",
                    "confidence": "확신도",
                    "negative_prob": "부정 확률",
                    "positive_prob": "긍정 확률",
                    "model_id": "모델 ID"
                },
                hide_index=True,
                use_container_width=True
            )
            
            if st.button("히스토리 초기화"):
                st.session_state.history = []
                st.session_state.total_predictions = 0
                st.session_state.positive_count = 0
                st.session_state.negative_count = 0
                st.rerun()
        else:
            st.info("아직 분석 기록이 없습니다.")
    
    with tab_manage:
        display_model_management(model_manager, config.project['model_name'])

if __name__ == "__main__":
    main() 