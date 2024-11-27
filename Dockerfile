FROM python:3.10-slim

# 환경 변수 설정
ENV AIRFLOW_HOME=/usr/local/ml4
ENV AIRFLOW_INIT_FLAG=$AIRFLOW_HOME/initialized

# Airflow 홈 디렉터리 생성 및 권한 설정
RUN mkdir -p $AIRFLOW_HOME && \
    chown -R root:root $AIRFLOW_HOME && \
    chmod -R 755 $AIRFLOW_HOME

WORKDIR $AIRFLOW_HOME

# 시스템 패키지 설치
RUN apt-get update && \
    apt-get install -y gcc libc-dev

# Airflow 설치
RUN pip install apache-airflow apache-airflow-providers-slack pandas scikit-learn joblib mlflow==2.16.2 tweepy numpy emoji konlpy transformers torch kiwipiepy matplotlib seaborn transformers[torch] plotly streamlit pytorch-lightning soynlp

# init-scripts 디렉토리 생성 및 권한 설정
RUN mkdir -p /init-scripts && \
    chmod -R 755 /init-scripts

# 초기화 스크립트 복사
COPY init-scripts/init.sh /init-scripts/init.sh

RUN chmod +x /init-scripts/init.sh && \
    sed -i 's/\r$//' /init-scripts/init.sh

# dags 폴더 복사
COPY src/ $AIRFLOW_HOME/src/

EXPOSE 8080 5050