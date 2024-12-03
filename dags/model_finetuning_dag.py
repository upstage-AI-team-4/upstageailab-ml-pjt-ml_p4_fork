from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from datetime import datetime, timedelta
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import mlflow
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl

# 프로젝트 루트 경로 추가
sys.path.append("/usr/local/ml4")

from src.config import Config
from src.utils.mlflow_utils import MLflowModelManager
from src.utils.inferencer import ModelInferencer
from src.train import train_model
from transformers import AutoTokenizer


os.environ['NO_PROXY'] = '*' # mac에서 airflow로 외부 요청할 때 이슈가 있음. 하여 해당 코드 추가 필요

def load_production_model(**context):
    """프로덕션 모델 로드"""
    config = Config()
    model_manager = MLflowModelManager(config)
    
    model_info = model_manager.load_production_model_info()
    if model_info is None:
        raise RuntimeError("No production model found")
        
    model = model_manager.load_production_model(config.project['model_name'])
    if model is None:
        raise RuntimeError("Failed to load production model")
        
    tokenizer = AutoTokenizer.from_pretrained(model_info['params']['pretrained_model'])
    
    return model, tokenizer, model_info

def prepare_wild_data(**context):
    """in-the-wild 데이터 준비 및 분할"""
    config = Config()
    
    # 모델 로드
    model, tokenizer, model_info = load_production_model()
    inferencer = ModelInferencer(model, tokenizer)
    
    # 데이터 로드
    data_path = Path(config.data_path) / config.dataset['in_the_wild']['wild_data_path']
    df = pd.read_csv(data_path)
    
    # 텍스트 컬럼명
    text_col = config.dataset['in_the_wild']['column_mapping']['text']
    label_col = config.dataset['in_the_wild']['column_mapping']['label']
    
    print(f"Loaded {len(df)} samples from {data_path}")
    
    # 추론 실행
    texts = df[text_col].tolist()
    results = inferencer.predict(texts)
    
    # 레이블 및 신뢰도 추가
    df[label_col] = [r['prediction'] for r in results]
    df['confidence'] = [r['confidence'] for r in results]
    
    # 높은 신뢰도(0.8 이상)의 데이터만 선택
    df_filtered = df[df['confidence'] >= 0.8].copy()
    print(f"Selected {len(df_filtered)} samples with high confidence")
    
    # 학습/검증 데이터 분할
    test_size = config.dataset['in_the_wild']['test_size']
    train_df, val_df = train_test_split(
        df_filtered,
        test_size=test_size,
        random_state=config.project['random_state'],
        stratify=df_filtered[label_col]
    )
    
    # 데이터 저장
    train_path = Path(config.data_path) / config.dataset['in_the_wild']['train_data_path']
    val_path = Path(config.data_path) / config.dataset['in_the_wild']['val_data_path']
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print(f"Saved {len(train_df)} training samples to {train_path}")
    print(f"Saved {len(val_df)} validation samples to {val_path}")
    
    # 데이터 통계
    stats = {
        'total_samples': len(df),
        'filtered_samples': len(df_filtered),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'positive_ratio_train': (train_df[label_col] == 1).mean(),
        'positive_ratio_val': (val_df[label_col] == 1).mean(),
    }
    
    return stats

def finetune_model(**context):
    """레이블링된 데이터로 모델 파인튜닝"""
    config = Config()
    
    # 기존 설정을 in-the-wild 데이터셋으로 변경
    config.project['dataset_name'] = 'in_the_wild'
    
    # 모델 학습 실행
    train_model(config)

def evaluate_and_promote(**context):
    """새로운 모델 평가 및 승격"""
    config = Config()
    model_manager = MLflowModelManager(config)
    
    # 최신 모델 정보 가져오기
    models = model_manager.list_models()
    if not models:
        raise RuntimeError("No models found")
    
    latest_model = models[-1]
    current_f1 = latest_model.get('metrics', {}).get('val_f1', 0)
    
    # 성능 임계값 확인
    threshold = config.mlflow.model_registry_metric_threshold
    
    if current_f1 > threshold:
        # Staging으로 승격
        latest_idx = len(models) - 1
        model_manager.stage_model_by_index(latest_idx, "Staging")
        print(f"New model promoted to Staging (F1: {current_f1} > threshold {threshold})")
    else:
        print(f"Model performance below threshold (F1: {current_f1} <= {threshold})")

def send_training_start_notification(**context):
    """학습 시작 알림"""
    config = Config()
    
    message = f"""
🚀 *모델 학습 파이프라인 시작*
• 모델: {config.project['model_name']}
• 데이터셋: {config.project['dataset_name']}
• 학습 설정:
  - Epochs: {config.models[config.project['model_name']]['training']['epochs']}
  - Batch Size: {config.models[config.project['model_name']]['training']['batch_size']}
  - Learning Rate: {config.models[config.project['model_name']]['training']['lr']}
  - Max Length: {config.models[config.project['model_name']]['training']['max_length']}
  - Optimizer: {config.models[config.project['model_name']]['training']['optimizer']}
"""
    
    # task_id 수정
    notification = SlackWebhookOperator(
        task_id='slack_start_notification',  # 변경된 task_id
        slack_webhook_conn_id="slack_webhook",
        message=message,
        username='ML Pipeline Bot',
        icon_emoji=':robot_face:'
    )
    
    return notification.execute(context=context)

def send_training_complete_notification(**context):
    """학습 완료 알림"""
    config = Config()
    ti = context['task_instance']
    
    # 데이터 통계 가져오기
    data_stats = ti.xcom_pull(task_ids='prepare_wild_data')
    
    # 최신 모델 정보 가져오기
    model_manager = MLflowModelManager(config)
    models = model_manager.list_models()
    if models:
        latest_model = models[-1]
        metrics = latest_model.get('metrics', {})
        
        message = f"""
✅ *모델 학습 파이프라인 완료*

📊 *데이터셋 정보*
• 전체 샘플 수: {data_stats['total_samples']}
• 필터링된 샘플 수: {data_stats['filtered_samples']}
• 학습 데이터: {data_stats['train_samples']}
• 검증 데이터: {data_stats['val_samples']}
• 학습 데이터 긍정 비율: {data_stats['positive_ratio_train']:.2%}
• 검증 데이터 긍정 비율: {data_stats['positive_ratio_val']:.2%}

📈 *모델 성능*
• F1 Score: {metrics.get('val_f1', 0):.4f}
• Accuracy: {metrics.get('val_accuracy', 0):.4f}
• Loss: {metrics.get('val_loss', 0):.4f}

🏷️ *모델 정보*
• 이름: {latest_model['run_name']}
• 스테이지: {latest_model['stage']}
• 타임스탬프: {latest_model['timestamp']}
"""
    else:
        message = "❌ *모델 학습 실패*\n모델 정보를 찾을 수 없습니다."
    
    # task_id 수정
    notification = SlackWebhookOperator(
        task_id='slack_complete_notification',  # 변경된 task_id
        slack_webhook_conn_id="slack_webhook",
        message=message,
        username='ML Pipeline Bot',
        icon_emoji=':robot_face:'
    )
    
    return notification.execute(context=context)

# DAG 설정
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'model_finetuning_pipeline',
    default_args=default_args,
    description='In-the-wild 데이터를 이용한 모델 파인튜닝 파이프라인',
    schedule_interval='0 2 * * 1',
    catchup=False
)

# GPU 상태 확인
check_gpu = BashOperator(
    task_id='check_gpu',
    bash_command='nvidia-smi || echo "GPU not available"',
    dag=dag
)

# 학습 시작 알림
start_notification = PythonOperator(
    task_id='training_start_notification',  # 변경된 task_id
    python_callable=send_training_start_notification,
    dag=dag
)

# 데이터 준비
prepare_data = PythonOperator(
    task_id='prepare_wild_data',
    python_callable=prepare_wild_data,
    dag=dag
)

# 모델 파인튜닝
finetune_task = PythonOperator(
    task_id='finetune_model',
    python_callable=finetune_model,
    dag=dag
)

# 모델 평가 및 승격
promote_task = PythonOperator(
    task_id='evaluate_and_promote',
    python_callable=evaluate_and_promote,
    dag=dag
)

# 학습 완료 알림
complete_notification = PythonOperator(
    task_id='training_complete_notification',  # 변경된 task_id
    python_callable=send_training_complete_notification,
    dag=dag
)

# 작업 순서 설정
start_notification >> check_gpu >> prepare_data >> finetune_task >> promote_task >> complete_notification 