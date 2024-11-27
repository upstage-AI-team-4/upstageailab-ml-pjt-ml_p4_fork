from pathlib import Path
from .base_model import BaseSentimentModel
class ModelFactory:
    def get_model(self, model_name: str, data_file: Path, model_dir: Path, 
                  pretrained_model_name: str, pretrained_model_dir: Path) -> BaseSentimentModel:
        """
        Create a sentiment model based on the model name
        Args:
            model_name (str): Name of the model to create
            data_file (Path): Path to the data file
            model_dir (Path): Directory to save the model
            pretrained_model_name (str): Name of the pretrained model to use
            pretrained_model_dir (Path): Directory to save the pretrained model
        Returns:
            BaseSentimentModel: An instance of the specified model
        """
        if model_name == 'KcELECTRA':
            from .kcelectra_model import KcELECTRAModel
            return KcELECTRAModel(data_file=data_file, 
                                 model_dir=model_dir, 
                                 pretrained_model_name=pretrained_model_name,
                                 pretrained_model_dir=pretrained_model_dir)
        elif model_name == 'KcBERT':
            from .kcbert_model import KcBERTModel
            return KcBERTModel(data_file=data_file, 
                              model_dir=model_dir, 
                              pretrained_model_name=pretrained_model_name,
                              pretrained_model_dir=pretrained_model_dir)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    

# 주요 기능:
# 1. ModelFactory 클래스:
# model_classes 딕셔너리를 사용하여 모델 이름과 클래스 간의 매핑을 관리합니다.
# get_model 메소드를 통해 모델 이름에 따라 적절한 모델 클래스를 반환합니다.
# 동적 클래스 호출:
# getattr을 사용하지 않고, 딕셔너리를 통해 클래스 참조를 관리하여 코드의 가독성을 높였습니다.
# 이렇게 하면 모델 이름에 따라 적절한 모델 클래스를 동적으로 선택하고 인스턴스를 생성할 수 있습니다. getattr 대신 딕셔너리를 사용하여 클래스 참조를 관리하는 것이 더 명확하고 안전한 방법입니다.