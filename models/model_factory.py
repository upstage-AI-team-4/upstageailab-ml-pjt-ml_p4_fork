from .kcbert_model import KcBERTSentimentModel
from .kcelectra_model import KcELECTRASentimentModel

class ModelFactory:
    def __init__(self):
        self.model_classes = {
            'KcBERT': KcBERTSentimentModel,
            'KcELECTRA': KcELECTRASentimentModel
        }

    def get_model(self, model_name, data_file, model_dir):
        """
        모델 이름에 따라 적절한 모델 클래스를 반환합니다.
        Args:
            model_name (str): 모델 이름 ('KcBERT' 또는 'KcELECTRA')
            data_file (str): 데이터 파일 경로
            model_dir (str): 모델 디렉토리 경로
        Returns:
            BaseSentimentModel: 선택된 모델 클래스의 인스턴스
        """
        model_class = self.model_classes.get(model_name)
        if not model_class:
            raise ValueError(f"지원되지 않는 모델 이름: {model_name}")
        
        return model_class(data_file=data_file, model_dir=model_dir)
    

# 주요 기능:
# 1. ModelFactory 클래스:
# model_classes 딕셔너리를 사용하여 모델 이름과 클래스 간의 매핑을 관리합니다.
# get_model 메소드를 통해 모델 이름에 따라 적절한 모델 클래스를 반환합니다.
# 동적 클래스 호출:
# getattr을 사용하지 않고, 딕셔너리를 통해 클래스 참조를 관리하여 코드의 가독성을 높였습니다.
# 이렇게 하면 모델 이름에 따라 적절한 모델 클래스를 동적으로 선택하고 인스턴스를 생성할 수 있습니다. getattr 대신 딕셔너리를 사용하여 클래스 참조를 관리하는 것이 더 명확하고 안전한 방법입니다.