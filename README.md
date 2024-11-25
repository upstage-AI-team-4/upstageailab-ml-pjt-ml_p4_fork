[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/znc2XbtA)
# 트위터 트렌드 분석 및 감성 분류 (모델 생성 및 서빙 & 워크플로우 구축)
## Team ML 4 < Walk into AI >

| ![박정준](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이다언](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김동완](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김묘정](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이현지](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [박정준](https://github.com/UpstageAILab)             |            [이다언](https://github.com/danielinjesus/AI_Portfolio/tree/main/AI_Projects/House_Price_Prediction)             |            [김묘정](https://github.com/UpstageAILab)             |            [이현지](https://github.com/UpstageAILab)             |            [김동완A](https://github.com/UpstageAILab)             |
|                            팀장, 피처조정 및 선별                             |                         피처엔지니어링                             |           모델링, AutoML                       |                                       베이스라인 코드 분석                   |                                  좌표 데이터 API 수집                        |
## 0. Overview

### Environment

- **IDE 및 코드 편집기**
    - Visual Studio Code (VS Code)
    - Cursor IDE
- **개발 환경**
    - Linux Docker 서버 및 로컬 환경에서 Python, Jupyter Notebook 사용
    - Anaconda로 패키지 및 환경 관리
- **버전 관리 및 협업 도구**
    - Git을 사용하여 코드 및 파일 버전 관리
    - GitHub를 통해 팀원들과 협업

### Requirements

- **프로그래밍 언어**
    - Python 3.x
- **주요 라이브러리 및 패키지**
    - Pandas
    - NumPy
    - Scikit-learn
    - XGBoost
    - WandB (Weights & Biases)
    - Sweetviz
    - Pygalker
    - Dataprep
    - Matplotlib
    - Seaborn
- **기타**
    - Jupyter Notebook
    - Docker (서버 환경 구축용)

---

## 1. Competition Info

### Overview

- **경진대회 주제:** 부동산 가격 예측 모델 개발
- **목표:** 주어진 데이터를 활용하여 부동산 가격을 정확하게 예측하는 모델을 구축하고 성능을 향상시키는 것
- **추가 세부 목표: **
-   프로젝트 수행 역량 기르기

### Timeline

- **프로젝트 기간:** 총 2주 (수강 1주 포함)
    - **1주차:** 팀원 역량 강화 및 베이스라인 코드 실행
    - **2주차:** 모델 개선 및 성능 향상 작업

---

## 2. Components

### Directory

```Python

├── src
│   ├── preprocess.py
│   │── feature.py
│   ├── train.py
│   └── utils.py
├── config
│   └── config.yaml
│   └── configs.py
├── output
│   ├── plots
│   │── reports
├── data
│   ├── preprocessed
├── notebook
├── data
├── main_sweep.py
├── main_feat_select.py
├── requirements.txt
└── README.md


```

---

## 3. Data Description

### Dataset Overview

- **메인 데이터셋:** 부동산 거래 데이터
    - **주요 변수:** 전용면적, 위치 정보, 거래 가격 등
- **추가 데이터:**
    - **버스 및 지하철 위치 데이터:** 역세권 여부 및 최단 거리 계산에 활용
    - **강남 대장 아파트 정보:** 지역적 특성 반영을 위해 추가

### EDA (Exploratory Data Analysis)

- **프로세스:**
    - **Pandas**를 사용하여 기본 통계량 분석 및 데이터 구조 파악
    - **Sweetviz**, **Pygalker**, **Dataprep** 등의 라이브러리로 데이터 시각화
    - 변수 간 상관관계 분석 및 분포 확인
- **결론:**
    - **전용면적의 중요성:** 반복적으로 중요한 변수로 선정됨
        - 분포가 우측으로 치우쳐 있어 상한을 완화하여 성능 개선
    - **이상치 처리의 영향:** IQR을 활용한 이상치 제거 시 성능 향상 미미
        - 이상치를 제거하지 않고 모델에 반영하였을 때 성능이 크게 향상됨
    - **추가 피처의 영향:** 외부 데이터를 활용한 피처 추가는 성능 개선에 큰 영향이 없었음

### Data Processing

- **데이터 정제:**
    - 결측치 처리 및 데이터 일관성 확보
- **피처 엔지니어링:**
    - **새로운 피처 생성:** 역세권 여부, 최단 거리 등
    - **추가 데이터 통합:** 강남 대장 아파트 정보 등
- **스케일링 및 변환:**
    - **Robust Scaling**, **Power Transform** 적용 시도
        - 성능 개선 효과는 미미하여 다른 방법 모색

---

## 4. Modeling

### Model Description

- **사용 모델:** XGBoost
- **선택 이유:**
    - **우수한 예측 성능:** 순차적 학습 방식을 통해 오차를 보완하며 Random Forest보다 성능 우수
    - **학습 속도 최적화:** 병렬 처리 및 캐시 최적화를 통한 빠른 학습 가능
    - **과적합 방지 기능:** L1, L2 정규화 및 다양한 규제 파라미터 제공

### 사용 예시

- 데이터 로드 완료...150000개, 50000개
샘플링 완료...1500개, 500개
        id                                           document  label  is_test
0  8932939                                        수OO만에 다시보네여      1        0
1  3681731                              일방적인 영화다. 관객 좀 고려해주시길      0        0
2  9847174                               세상을 초월하는 한 사람의 선한 마음      1        0
3  8506899             멍하다.. 여러생각이 겹치는데 오랜만에 영화 보고 이런 느낌 느껴본다      1        0
4  9991656  우와 별 반개도 아까운판에 밑에 CJ 알바생들 쩐다.. 전부 만점이야 ㅎㅎㅎ..,....      0        0
데이터 전처리 시작...
d:\dev\twitter\data\naver_movie_review.csv에서 데이터를 로드했습니다.
text, label 컬럼 확인...
text 컬럼이 없습니다. document 컬럼을 사용합니다.
NaN 값 확인 및 처리...
전처리 전 NaN 값 개수: 0
전처리 후 NaN 값 개수: 0
텍스트 정제 시작...
토큰화 시작...
빈 문자열 확인 및 처리...
빈 문자열 개수: 7
전처리된 데이터를 d:\dev\twitter\data\processed\processed_naver_movie_review.csv에 저장했습니다.
최종 데이터 크기: 1993 행
모델을 e:\models\KcBERT에서 로드했습니다.
데이터를 d:\dev\twitter\data\processed\processed_naver_movie_review.csv에서 로드했습니다.
데이터 준비 시작...
전체 데이터 크기: 1993
빈 문장 제거 후 데이터 크기: 1993
토큰화 시작...
학습 데이터: 1594개
검증 데이터: 399개
임베딩 레이어가 고정되었습니다.

=== 레이어 고정 상태 ===
전체 레이어 수: 12
고정된 레이어 수: 9
학습에 사용될 레이어 수: 3

=== 파라미터 상태 ===
전체 파라미터: 108,920,066
학습 가능한 파라미터: 21,855,746 (20.07%)
고정된 파라미터: 87,064,320 (79.93%)
---

## 5. Result

### Leader Board

- **베이스라인 점수:** 46,433
- **최종 점수:** 14,844 (베이스라인 대비 약 2배 향상)
- **순위:** 3등

*(리더보드 캡처 이미지가 있을 경우 첨부)*

### Presentation

- **발표 자료:** [Google Slide 링크](https://docs.google.com/presentation/d/1yZgRoott_eZnF6p2f0CFERun2Yl78FbtjVDjerJ6Zms/edit#slide=id.g314fe61daed_10_89)

### Report
- **Fast-Up Report:**[Notion 링크](https://www.notion.so/Fast-Up-Team-Report-e11312a2a7f9433eb421031b8e7b337a?pvs=4)

### Meeting Log

- **회의록:** [Notion 링크](https://www.notion.so/6289d8a63f724563b0a470ddece7ff24?v=697ed335fa56441a82df84cdac6099e7&pvs=4)

### Reference

- **모델 관련 문서:**
    - [XGBoost 공식 문서](https://xgboost.readthedocs.io/)
    - Scikit-learn Feature Selection
- **관련 블로그 및 자료:**
    - 부동산 가격 예측 프로젝트 관련 블로그
    - 데이터 분석 및 전처리 기법 자료
