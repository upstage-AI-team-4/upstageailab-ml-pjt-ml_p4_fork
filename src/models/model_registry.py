import mlflow
from pathlib import Path
import json
from datetime import datetime

class ModelRegistry:
    def __init__(self, tracking_uri="http://127.0.0.1:5000", 
                 experiment_name="sentiment_classification",
                 metric_threshold={'eval_f1': 0.85}):
        """
        Initialize ModelRegistry
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: MLflow experiment name
            metric_threshold: Dictionary of metric names and their threshold values
        """
        mlflow.set_tracking_uri(tracking_uri)
        self.experiment_name = experiment_name
        self.metric_threshold = metric_threshold
        self.client = mlflow.tracking.MlflowClient()
        
        # 설정 파일 저장 경로
        self.config_dir = Path(__file__).parent.parent / 'configs'
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.production_config_path = self.config_dir / 'production_models.json'

    def evaluate_and_register(self, run_id: str, model_name: str) -> bool:
        """
        Evaluate model metrics and register if they meet thresholds
        """
        print(f"\n=== 모델 평가 및 등록 시작 ===")
        run = self.client.get_run(run_id)
        metrics = run.data.metrics
        
        # 메트릭 평가
        meets_threshold = all(
            metrics.get(metric_name, 0) >= threshold_value 
            for metric_name, threshold_value in self.metric_threshold.items()
        )
        
        if meets_threshold:
            # 모델 등록 또는 업데이트
            model_details = self._register_model(run_id, model_name)
            if model_details:
                self._save_production_config(model_details)
                return True
        
        print(f"모델이 기준을 충족하지 못했습니다. (F1 score: {metrics.get('eval_f1', 0):.4f})")
        return False

    def _register_model(self, run_id: str, model_name: str) -> dict:
        """
        Register model to MLflow Model Registry
        """
        try:
            registered_name = f"{model_name}_sentiment_classifier"
            
            # 새 모델 버전 등록
            result = mlflow.register_model(f"runs:/{run_id}/model", registered_name)
            new_version = result.version
            
            # Production으로 설정
            self.client.transition_model_version_stage(
                name=registered_name,
                version=new_version,
                stage="Production"
            )
            
            print(f"모델이 성공적으로 등록되었습니다: {registered_name} (version {new_version})")
            
            return {
                "model_name": model_name,
                "registered_name": registered_name,
                "run_id": run_id,
                "version": new_version,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "metrics": self.client.get_run(run_id).data.metrics
            }
            
        except Exception as e:
            print(f"모델 등록 중 오류 발생: {e}")
            return None

    def _save_production_config(self, model_details: dict):
        """
        Save production model details to config file
        """
        # 기존 설정 로드 또는 새로 생성
        if self.production_config_path.exists():
            with open(self.production_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = {"models": {}}
        
        # 새 모델 정보 추가/업데이트
        model_name = model_details["model_name"]
        if model_name not in config["models"]:
            config["models"][model_name] = []
        
        # 최신 버전 추가
        config["models"][model_name].append(model_details)
        
        # 타임스탬프 순으로 정렬
        config["models"][model_name].sort(key=lambda x: x["timestamp"], reverse=True)
        
        with open(self.production_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"Production 모델 설정이 저장되었습니다: {self.production_config_path}")

    def get_production_models(self, model_name: str = None) -> dict:
        """
        Get production models
        Args:
            model_name: Optional model name to filter results
        Returns:
            Dictionary of production models
        """
        if not self.production_config_path.exists():
            return {}
            
        with open(self.production_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        if model_name:
            return {model_name: config["models"].get(model_name, [])}
        return config["models"]

    def get_best_production_model(self, model_name: str = None, metric: str = "eval_f1") -> dict:
        """
        Get the best production model based on specified metric
        Args:
            model_name: Optional model name to filter results
            metric: Metric to use for comparison
        Returns:
            Best model details
        """
        models = self.get_production_models(model_name)
        if not models:
            return None
            
        best_model = None
        best_metric = float('-inf')
        
        for name, versions in models.items():
            for version in versions:
                current_metric = version["metrics"].get(metric, float('-inf'))
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_model = version
                    
        return best_model