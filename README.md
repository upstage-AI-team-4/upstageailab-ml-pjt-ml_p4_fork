# MLOps:

# I. Model Management

## 1. 프로젝트 구조 및 설정

```plaintext
project_root/
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py          # Configuration management
│   ├── models/
│   │   ├── __init__.py
│   │   └── model.py          # Model architecture definitions
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── inferencer.py     # Model inference utilities
│   │   └── mlflow_utils.py   # MLflow integration utilities
│   └── train/
│       ├── __init__.py
│       └── trainer.py        # Model training logic
├── examples/
│   └── inference.py          # Example inference script
├── tests/
│   ├── __init__.py
│   └── test_*.py            # Test files
├── configs/
│   └── config.yaml          # Configuration files
├── requirements.txt
├── README.md
└── .env                     # Environment variables
```

## 주요 컴포넌트 설명

### 📁 src
- **config**: 프로젝트 설정 관리
- **models**: 모델 아키텍처 정의
- **utils**: 유틸리티 함수 모음
- **train**: 학습 관련 로직

### 📁 examples
- 모델 추론 예제 스크립트

### 📁 tests
- 단위 테스트 및 통합 테스트

### 📁 configs
- YAML 기반 설정 파일

### 📄 주요 파일
- `requirements.txt`: 프로젝트 의존성
- `.env`: 환경 변수
- `README.md`: 프로젝트 문서

## 개발 환경 설정
- Python 3.8+
- Rye를 통한 의존성 관리
- MLflow를 통한 실험 관리

### 1.1 주요 폴더 및 파일 구조

- **config/**
    - `config.yaml`: 기본 설정을 구성하는 파일
        - **데이터셋 종류**: 사용할 데이터셋 종류 설정 (기본값: NSMC - 네이버 영화 리뷰)
        - **모델 설정**: 사용할 모델 및 학습 파라미터 설정 (기본값: KcBERT)
        - **기타 파라미터**:
            - `dataset_sampling_rate`: 빠른 실험을 위한 데이터셋 샘플링 비율
            - `max_length`: 모델 입력의 최대 길이
            - `register_threshold`: 모델 등록을 위한 최소 기준
            - `unfrozen_layers`: 학습 시 언프리즈할 레이어 수
- **src/**
    - **models/**: 모델 관련 코드
    - **data/**: 데이터 처리 관련 코드
    - **utils/**: 유틸리티 코드
    - `train.py`: 기본 학습 및 모델 관리 실행 스크립트
- data/
    - raw/: raw data
    - processed/: processed data
- models/
    - Pretrained models
- 

`data, models 폴더 및 파일이 없는 경우에도 [train.py](http://train.py) 실행시 저절로 데이터,모델 다운받아 실행`

---

## 2. 실행 순서

### 2.1 Conda 환경 생성

Python 3.10 버전의 Conda 가상 환경을 생성하고 활성화.

```bash
conda create -n ml4 python=3.10
conda activate ml4
```

### 2.2 필요 모듈 설치

프로젝트에 필요한 의존성 모듈을 설치

```bash
pip install -r requirements.txt
```

### 2.3 설정 파일 확인 및 수정

`config/config.yaml` 파일을 열어 필요한 설정을 확인하고 실험에 맞게 수정

- **데이터셋 설정**: `dataset` 섹션에서 데이터셋 종류와 샘플링 비율 등을 설정
- **모델 설정**: `model` 섹션에서 사용할 모델 이름과 학습 파라미터 등을 설정
- **학습 설정**: `train` 섹션에서 에포크 수, 배치 크기 등을 설정

### 2.4 MLflow 서버 실행

프로젝트 루트 디렉토리에서 다음 명령어를 실행하여 MLflow UI를 시작

```bash
mlflow ui
```

브라우저에서 [http://127.0.0.1:5000](http://127.0.0.1:5000/) 에 접속하여 MLflow UI에 접근

### 2.5 모델 학습 시작

터미널에서 다음 명령어를 실행하여 모델 학습을 시작하거나, IDE에서 `train.py`를 실행:

```bash
python train.py
```

### 2.6 모델 관리

학습 완료 후 터미널에 나타나는 모델 관리 관련 메시지에 따라 CLI에서 숫자 또는 `y/n`을 입력하여 모델을 관리.

- **모델 등록**: 모델을 레지스트리에 등록할지 여부 선택
- **모델 단계 설정**: 모델의 단계(stage)를 설정 (예: None, Staging, Production)

### 2.7 결과 확인

- **MLflow UI**: 브라우저에서 실험 결과, 메트릭, 파라미터 및 아티팩트를 확인.
- **폴더 구조**:
    - `mlruns/` 폴더에 실행(run) 관련 로그와 메트릭이 저장.
    - `mlartifacts/` 폴더에 모델 파일 등 아티팩트가 저장.
    - `config/model_info.json` 파일에서 등록된 모델의 단계(stage)를 확인.

### 2.8 Streamlit App 실행
```bash
streamlit run app.py
```

---

이 가이드를 따라 프로젝트를 실행하고 모델을 학습 및 관리. 필요에 따라 `config.yaml` 파일의 설정을 조정하여 실험을 진행

# 프로젝트 세부 사항

## 주요 설정 항목 설명

- **데이터셋 종류** (`dataset.name`): 사용할 데이터셋의 이름을 지정. 기본값은 `nsmc`
- **모델 이름** (`model.name`): 사용할 사전 학습된 모델의 이름을 지정. 기본값은 `KcBER`
- **데이터셋 샘플링 비율** (`dataset.sampling_rate`): 데이터셋의 일부만 사용하여 빠른 실험을 진행
- **최대 입력 길이** (`dataset.max_length`): 모델 입력 시퀀스의 최대 길이를 지정
- **모델 등록 최소 기준** (`model.register_threshold`): 모델을 레지스트리에 등록하기 위한 최소 성능 기준을 설정
- **언프리즈할 레이어 수** (`model.unfrozen_layers`): 모델 학습 시 업데이트할 레이어의 수를 지정

## 추가 참고 사항

- **환경 설정**: 가상 환경을 사용하여 의존성 충돌을 방지
- **설정 조정**: `config.yaml` 파일을 수정하여 다양한 실험을 진행
- **모델 관리 자동화**: 학습 스크립트 실행 후 자동으로 모델 등록 및 관리 메시지
- **MLflow 사용**: 실험 추적, 모델 관리 및 배포

# II. Docker for Airflow Setup

## 사용법 및 명령어

Airflow를 Docker로 설정하려면 아래 명령어를 실행:

```bash
docker-compose up --build -d

```

## Airflow 계정 자동 생성

Airflow 초기 설정 시 다음 기본 계정이 자동으로 생성:

- **ID**: `admin`
- **Password**: `admin`

---

## Slack Webhook 설정

Airflow에서 Slack Webhook을 사용하려면 다음 정보를 `.env` 파일에 저장:

### `.env` 파일 예시

```
env
코드 복사
# Slack Webhook Token 설정
SLACK_WEBHOOK_TOKEN=PUT YOUR SLACK TOKEN

# Airflow 설정
AIRFLOW__CORE__LOAD_EXAMPLES=False
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=sqlite:////usr/local/ml4/airflow.db
AIRFLOW__PROVIDERS__SLACK__WEBHOOK_CONN_ID=slack_webhook

```

Slack Webhook 연결을 위해 Airflow의 Connection ID를 아래와 같이 설정:

- **Connection ID**: `slack_webhook`
- **Token**: `.env` 파일에 설정된 `SLACK_WEBHOOK_TOKEN` 값 사용

---

## 추가 참고 사항

- `docker-compose.yml` 파일이 제대로 구성되어 있는지 확인.
- Airflow를 실행한 후 웹 UI에서 Slack Webhook Connection 설정이 올바르게 등록되었는지 확인.
