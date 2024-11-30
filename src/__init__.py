# src/__init__.py 파일을 만들어서 전역적으로 경고를 비활성화하겠습니다:
import warnings
import torchvision
# 경고 메시지 비활성화
warnings.filterwarnings('ignore')
torchvision.disable_beta_transforms_warning()

# # torchvision image 관련 경고 비활성화
# warnings.filterwarnings('ignore', message='Failed to load image Python extension')

# # datapoints 관련 경고 비활성화
# warnings.filterwarnings('ignore', message='The torchvision.datapoints')

# # transforms.v2 관련 경고 비활성화
# warnings.filterwarnings('ignore', message='The torchvision.transforms.v2')

# # libjpeg/libpng 관련 경고 비활성화
# warnings.filterwarnings('ignore', message='.*libjpeg.*')
# warnings.filterwarnings('ignore', message='.*libpng.*') 