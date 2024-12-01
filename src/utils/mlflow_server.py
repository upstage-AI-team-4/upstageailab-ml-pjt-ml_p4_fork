import os
from pathlib import Path
import subprocess
from src.config import Config

def start_mlflow_server(config: Config):
    """MLflow 서버 시작"""
    
    # MLflow 저장 경로 설정
    mlruns_path = config.project_root / config.mlflow.mlrun_path
    mlartifacts_path = config.project_root / config.mlflow.artifact_location
    
    # 디렉토리 생성
    os.makedirs(mlruns_path, exist_ok=True)
    os.makedirs(mlartifacts_path, exist_ok=True)
    
    # MLflow 서버 실행 명령
    cmd = [
        "mlflow", "server",
        "--backend-store-uri", f"file://{mlruns_path}",
        "--default-artifact-root", f"file://{mlartifacts_path}",
        "--host", "127.0.0.1",
        "--port", "5000"
    ]
    
    print(f"\nStarting MLflow server...")
    print(f"Tracking URI: {config.mlflow.tracking_uri}")
    print(f"Experiment name: {config.mlflow.experiment_name}")
    print(f"Experiment data: {mlruns_path}")
    print(f"Artifacts: {mlartifacts_path}\n")
    
    # 서버 실행
    subprocess.Popen(cmd)

if __name__ == "__main__":
    config = Config()
    start_mlflow_server(config) 