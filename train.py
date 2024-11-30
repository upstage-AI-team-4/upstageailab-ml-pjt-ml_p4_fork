import os
import warnings
from pathlib import Path
import json
from datetime import datetime
import mlflow
import torch  # torch import 추가
from pytorch_lightning import Trainer, seed_everything
from transformers import AutoTokenizer
from src.utils.visualization import plot_confusion_matrix
from src.models.kcbert_model import KcBERT
from src.data.nsmc_dataset import NSMCDataModule
from src.utils.config import Config
from src.utils.mlflow_utils import MLflowModelManager
# torchvision 관련 경고 무시
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
# Tensor Core 최적화를 위한 precision 설정
torch.set_float32_matmul_precision('medium')  # 또는 'high'

def get_local_path_from_uri(uri: str) -> Path:
    """MLflow URI를 로컬 경로로 변환"""
    # file:/// 또는 file:// 또는 file: 제거
    if uri.startswith(('file:///', 'file://', 'file:')):
        # 운영체제 상관없이 처리
        path = uri.split('file:')[-1].lstrip('/')
        # Windows에서 드라이브 문자(예: C:) 처리
        if os.name == 'nt' and len(path) >= 2 and path[1] == ':':
            return Path(path)
        return Path('/' + path)
    return Path(uri)

def cleanup_artifacts(config, metrics, run_id):
    """체크포인트와 아티팩트 정리"""
    # 체크포인트 폴더 정리
    checkpoint_dir = config.base_path / config.base_training.checkpoint['dirpath']
    if checkpoint_dir.exists():
        import shutil
        shutil.rmtree(checkpoint_dir)  # 폴더 자체를 삭제
    
    # MLflow 아티팩트 정리
    if metrics["val_accuracy"] <= config.mlflow.model_registry_metric_threshold:
        # threshold를 넘지 못한 모델의 아티팩트는 삭제
        artifact_path = config.mlflow.mlrun_path / run_id / "artifacts"
        if artifact_path.exists():
            shutil.rmtree(artifact_path)

def print_prediction_samples(model, data_module, tokenizer, n_samples=5):
    """샘플 예측 결과 출력"""
    print("\n=== Sample Predictions ===")
    print(f"Showing {n_samples} random samples from validation set")
    print("-" * 80)
    
    # 검증 데이터셋에서 랜덤 샘플 선택
    val_dataset = data_module.val_dataset
    indices = torch.randperm(len(val_dataset))[:n_samples]
    
    model.eval()
    with torch.no_grad():
        for idx in indices:
            # 원본 텍스트와 레이블 가져오기
            original_text = val_dataset.documents[idx]
            true_label = val_dataset.labels[idx]
            
            # 모델 입력 준비
            inputs = val_dataset[idx]
            input_ids = inputs['input_ids'].unsqueeze(0)
            attention_mask = inputs['attention_mask'].unsqueeze(0)
            
            # GPU 사용 중이면 텐서를 GPU로 이동
            if next(model.parameters()).is_cuda:
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
            
            # 예측
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_label = torch.argmax(logits, dim=-1).item()
            confidence = probs[0][pred_label].item()
            
            # 결과 출력
            print(f"Original Text: {original_text}")
            print(f"True Label: {true_label}")
            print(f"Predicted Label: {pred_label}")
            print(f"Confidence: {confidence:.4f}")
            print(f"Correct: {'✓' if pred_label == true_label else '✗'}")
            print("-" * 80)

def train(config):
    # MLflow 설정
    print("=" * 50)
    print("\n=== Training Configuration ===")
    print(f"Pretrained Model: {config.base_training.pretrained_model}")
    print(f"Batch Size: {config.base_training.batch_size}")
    print(f"Learning Rate: {config.base_training.lr}")
    print(f"Epochs: {config.base_training.epochs}")
    print(f"Max Length: {config.base_training.max_length}")
    print("=" * 50 + "\n")
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    
    # 실험 설정
    experiment = mlflow.get_experiment_by_name(config.mlflow.experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            config.mlflow.experiment_name,
            artifact_location=f"file://{config.mlflow.mlrun_path}"
        )
    else:
        experiment_id = experiment.experiment_id
    
    mlflow.set_experiment(experiment_id)
    
    # 시드 설정
    seed_everything(config.base_training.random_seed)
    
    # Run 이름 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.project.model_name}_{config.project.dataset_name}_{timestamp}"
    
    # MLflow 실험 시작
    with mlflow.start_run(run_name=run_name) as run:
        print(f"\n=== Starting new run: {run_name} ===")
        
        # 토크나이저 설정
        tokenizer = AutoTokenizer.from_pretrained(config.base_training.pretrained_model)
        
        # 데이터 모듈 설정
        data_module = NSMCDataModule(
            tokenizer=tokenizer,
            batch_size=config.base_training.batch_size,
            max_length=config.base_training.max_length,
            sampling_rate=config.data.sampling_rate,
            config=config,
            data_dir=str(config.data_path),
            train_file=config.data.train_data_path,
            val_file=config.data.val_data_path
        )
        
        # 모이터 준비 및 설정
        data_module.prepare_data()  # 데이터 다운로드 등 준비 작업
        data_module.setup(stage='fit')  # 데이터셋 설정
        
        # 모델 설정
        model = KcBERT(**config.get_model_kwargs())
        
        # Trainer 설정 및 학습
        trainer = Trainer(**config.get_trainer_kwargs())
        trainer.fit(model, data_module)
        
        # 평가기 초기화
        evaluator = ModelEvaluator(model, tokenizer)
        
        # 한 번에 모든 평가 수행
        eval_metrics = evaluator.evaluate_dataset(data_module)
        
        # 결과 출력
        print("\n=== Evaluation Results ===")
        print(f"Accuracy: {eval_metrics['accuracy']:.4f}")
        print(f"Average Confidence: {eval_metrics['avg_confidence']:.4f}")
        
        # 신뢰도 구간별 정확도 출력
        print("\n=== Accuracy by Confidence Level ===")
        for bin_name, bin_data in eval_metrics['confidence_bins'].items():
            print(f"{bin_name}: {bin_data['accuracy']:.4f} ({bin_data['count']} samples)")
        
        # 샘플 예측 결과 출력
        print("\n=== Sample Predictions ===")
        for sample in eval_metrics['sample_predictions']:
            print(f"Text: {sample['text']}")
            print(f"True Label: {sample['true_label']}")
            print(f"Predicted: {sample['predicted_label']} (confidence: {sample['confidence']:.4f})")
            print(f"Correct: {'✓' if sample['correct'] else '✗'}")
            print("-" * 80)
        
        # MLflow에 메트릭 로깅
        mlflow.log_metrics({
            "val_accuracy": eval_metrics['accuracy'],
            "val_avg_confidence": eval_metrics['avg_confidence']
        })
        
        # Confusion matrix 수정
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        # 전체 validation 데이터에 대한 예측 수집
        all_preds = []
        all_labels = []
        model.eval()
        with torch.no_grad():
            for batch in data_module.val_dataloader():
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**{k: v for k, v in batch.items() if k != 'labels'})
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        # Confusion Matrix 생성 및 저장
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 임시 파일로 저장
        cm_path = 'confusion_matrix.png'
        plt.savefig(cm_path)
        plt.close()
        
        # MLflow에 로깅
        mlflow.log_artifact(cm_path, "confusion_matrix")
        os.remove(cm_path)  # 임시 파일 삭제
        
        return run.info.run_id, eval_metrics, run_name

def main():
    # Config 설정
    config = Config()
    
    # 로그 디렉토리 생성
    log_dir = config.base_path / "logs" / "lightning_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # MLflow 경로 정보 출력
    print("=" * 50)
    print("\n=== MLflow Configuration ===")
    print(f"MLflow Tracking URI: {config.mlflow.tracking_uri}")
    print(f"MLflow Run Path: {config.mlflow.mlrun_path}")
    print(f"MLflow Experiment Name: {config.mlflow.experiment_name}")
    print("=" * 50 + "\n")
    print(f"Model Name: {config.project.model_name}")
    print(f"Dataset Name: {config.data.dataset_name}")
    print(f"Sampling Rate: {config.data.sampling_rate}")
    print(f"Test Size: {config.data.test_size}")

    

    # 학습 실행
    run_id, metrics, run_name = train(config)
    
    print("\n=== Training completed ===")
    print(f"Run Name: {run_name}")
    print(f"Run ID: {run_id}")
    print("Validation metrics:")
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric_name}: {value:.4f}")
    
    # 학습된 모델로 추가 평가
    print("\n=== Validation Sample Predictions ===")
    
    # validation 데이터셋에서 랜덤 샘플 선택
    val_dataset = data_module.val_dataset
    n_samples = 10  # 보여줄 샘플 수
    indices = torch.randperm(len(val_dataset))[:n_samples].tolist()
    
    model.eval()
    with torch.no_grad():
        for idx in indices:
            # 원본 텍스트와 레이블 가져오기
            text = val_dataset.documents[idx]
            true_label = val_dataset.labels[idx]
            
            # 모델 입력 준비
            sample = val_dataset[idx]
            inputs = {
                'input_ids': sample['input_ids'].unsqueeze(0).to(model.device),
                'attention_mask': sample['attention_mask'].unsqueeze(0).to(model.device)
            }
            
            # 예측
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_label = torch.argmax(logits, dim=-1).item()
            confidence = probs[0][pred_label].item()
            
            # 결과 출력
            print("\nText:", text)
            print(f"True Label: {'긍정' if true_label == 1 else '부정'}")
            print(f"Prediction: {'긍정' if pred_label == 1 else '부정'}")
            print(f"Confidence: {confidence:.4f}")
            print(f"Correct: {'O' if pred_label == true_label else 'X'}")
            print("-" * 80)
    
    # 2. 사용자 입력 받아서 실시간 추론
    print("\n=== Interactive Inference ===")
    print("Enter your text (or 'q' to quit):")
    
    while True:
        user_input = input("\nText: ").strip()
        if user_input.lower() == 'q':
            break
            
        if not user_input:
            continue
            
        result = inferencer.predict(user_input)[0]
        print(f"Prediction: {'긍정' if result['prediction'] == 1 else '부정'}")
        print(f"Confidence: {result['confidence']:.4f}")
    
    # MLflow 실험 결과 위치 출력
    print("\n=== MLflow Run Information ===")
    print(f"Run logs and artifacts: {config.mlflow.mlrun_path / run_id}")
    print("=" * 50)
    
    # 모델 관리 인터페이스 실행 (선택적)
    if input("\nWould you like to manage models? (y/n): ").lower() == 'y':
        model_manager = MLflowModelManager(config)
        model_manager.manage_model(config.project.model_name)
    
    # 체크포인트와 아티팩트 정리
    cleanup_artifacts(config, metrics, run_id)

if __name__ == '__main__':
    main()
