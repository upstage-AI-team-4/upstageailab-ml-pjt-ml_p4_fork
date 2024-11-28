from pathlib import Path
import yaml
from typing import Dict, Any, Union, List
import logging

logger = logging.getLogger(__name__)

class Config:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """설정 파일 로드"""
        config_path = Path(__file__).parent.parent.parent / 'config' / 'config.yaml'
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
            self._convert_types()
            logger.info(f"설정 파일 로드 완료: {config_path}")
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {e}")
            raise
    
    def _convert_types(self):
        """숫자 값들의 타입 변환"""
        # Data
        self._config['data']['sampling_rate'] = float(self._config['data']['sampling_rate'])
        self._config['data']['max_length'] = int(self._config['data']['max_length'])
        
        # Model
        self._config['model']['register_threshold'] = float(self._config['model']['register_threshold'])
        self._config['model']['num_unfrozen_layers'] = int(self._config['model']['num_unfrozen_layers'])
        
        # Train
        self._config['train']['learning_rate'] = float(self._config['train']['learning_rate'])
        self._config['train']['num_train_epochs'] = int(self._config['train']['num_train_epochs'])
        self._config['train']['batch_size'] = int(self._config['train']['batch_size'])
        self._config['train']['early_stopping_patience'] = int(self._config['train']['early_stopping_patience'])
        
        # Optuna
        self._config['optuna']['n_trials'] = int(self._config['optuna']['n_trials'])
        self._config['optuna']['timeout'] = int(self._config['optuna']['timeout'])
        
        # Search space
        search_space = self._config['optuna']['search_space']
        search_space['learning_rate'] = [float(x) for x in search_space['learning_rate']]
        search_space['batch_size'] = [int(x) for x in search_space['batch_size']]
        search_space['num_unfrozen_layers'] = [int(x) for x in search_space['num_unfrozen_layers']]
    
    @property
    def data(self) -> Dict[str, Union[str, float, int]]:
        return self._config['data']
    
    @property
    def model(self) -> Dict[str, Union[str, float, int, Dict]]:
        return self._config['model']
    
    @property
    def train(self) -> Dict[str, Union[float, int]]:
        return self._config['train']
    
    @property
    def optuna(self) -> Dict[str, Union[int, str, Dict[str, List[Union[float, int]]]]]:
        return self._config['optuna']
    
    @property
    def mlflow(self) -> Dict[str, str]:
        return self._config['mlflow'] 