import os
from pathlib import Path
import subprocess
from src.config import Config

def start_mlflow_server(config: Config):
    """MLflow 서버 시작"""
    
    # MLflow 저장 경로 설정
    mlruns_path = config.base_path / config.mlflow.mlrun_path
    mlartifacts_path = config.base_path / config.mlflow.mlartifact_path
    
    # 디렉토리 생성
    os.makedirs(mlruns_path, exist_ok=True)
    os.makedirs(mlartifacts_path, exist_ok=True)
    
    # MLflow 서버 실행 명령
    cmd = [
        "mlflow", "server",
        "--backend-store-uri", f"file://{config.mlflow.mlrun_path}",
        "--default-artifact-root", f"file://{config.mlflow.mlartifact_path}",
        "--host", "127.0.0.1",
        "--port", "5050"
    ]
    
    print(f"\nStarting MLflow server...")
    print(f"Tracking URI: {config.mlflow.tracking_uri}")
    print(f"Experiment data: {mlruns_path}")
    print(f"Artifacts: {mlartifacts_path}\n")
    
    # 서버 실행
    subprocess.Popen(cmd)

if __name__ == "__main__":
    config = Config()
    start_mlflow_server(config) 

# MLflow UI 실행 옵션:
# mlflow ui --host 0.0.0.0 --port 5050 &
# 
# 주요 옵션:
# --host: 호스트 주소 (0.0.0.0은 모든 IP에서 접근 허용)
# --port: UI 서버 포트 지정
# --backend-store-uri: MLflow 실험 데이터 저장 위치
# --default-artifact-root: 모델 아티팩트 저장 위치
# --workers: Gunicorn worker 프로세스 수
# --gunicorn-opts: Gunicorn 서버 추가 옵션 