FROM python:3.10-slim

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    gcc \
    libc-dev \
    libgomp1 \
    curl \
    git \
    wget \
    tzdata \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 타임존 설정
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 환경 변수 설정
ENV AIRFLOW_HOME=/usr/local/ml4
ENV AIRFLOW_INIT_FLAG=$AIRFLOW_HOME/initialized
ENV PYTHONPATH="${PYTHONPATH}:${AIRFLOW_HOME}"

# Airflow 홈 디렉터리 생성 및 권한 설정
RUN mkdir -p $AIRFLOW_HOME && \
    useradd -ms /bin/bash -d $AIRFLOW_HOME airflow && \
    chown -R airflow:airflow $AIRFLOW_HOME && \
    chmod -R 775 $AIRFLOW_HOME

WORKDIR $AIRFLOW_HOME

# Python 패키지 설치
RUN pip install --no-cache-dir \
    apache-airflow==2.8.1 \
    apache-airflow-providers-slack \
    mlflow==2.8.1 \
    pandas \
    scikit-learn \
    joblib \
    tweepy \
    numpy \
    torch torchvision torchaudio \
    transformers \
    streamlit \
    plotly \
    pytorch-lightning \
    emoji \
    konlpy \
    kiwipiepy \
    matplotlib \
    seaborn \
    soynlp \
    python-dotenv \
    "pendulum>=2.0.0,<3.0.0" \
    Flask-Session==0.5.0 \
    connexion==2.14.2 \
    swagger-ui-bundle==0.0.9 \
    apispec==6.3.0

# 디렉토리 생성 및 권한 설정
RUN mkdir -p /usr/local/ml4/dags && \
    mkdir -p /usr/local/ml4/logs && \
    mkdir -p /usr/local/ml4/mlruns && \
    mkdir -p /usr/local/ml4/config && \
    mkdir -p /usr/local/ml4/models && \
    mkdir -p /usr/local/ml4/data && \
    mkdir -p /usr/local/ml4/connections && \
    mkdir -p /init-scripts && \
    mkdir -p /usr/local/test && \
    touch /usr/local/ml4/logs/airflow.log && \
    touch /usr/local/ml4/logs/mlflow.log && \
    touch /usr/local/ml4/logs/streamlit.log && \
    chown -R airflow:airflow /usr/local/ml4/dags && \
    chown -R airflow:airflow /usr/local/ml4/logs && \
    chown -R airflow:airflow /usr/local/ml4/mlruns && \
    chown -R airflow:airflow /usr/local/ml4/config && \
    chown -R airflow:airflow /usr/local/ml4/models && \
    chown -R airflow:airflow /usr/local/ml4/data && \
    chown -R airflow:airflow /usr/local/ml4/connections && \
    chmod -R 777 /usr/local/ml4/dags && \
    chmod -R 777 /usr/local/ml4/logs && \
    chmod -R 777 /usr/local/ml4/mlruns && \
    chmod -R 777 /usr/local/ml4/config && \
    chmod -R 777 /usr/local/ml4/models && \
    chmod -R 777 /usr/local/ml4/data && \
    chmod -R 777 /usr/local/ml4/connections && \
    chmod -R 777 /init-scripts && \
    chown -R airflow:airflow /usr/local/test && \
    chmod -R 777 /usr/local/test && \
    chmod 666 /usr/local/ml4/logs/airflow.log && \
    chmod 666 /usr/local/ml4/logs/mlflow.log && \
    chmod 666 /usr/local/ml4/logs/streamlit.log

# 로그 파일 생성 및 권한 설정 부분 제거 (위에서 이미 처리됨)

# 포트 노출
EXPOSE 8080 5050 8501

# .env 파일 복사 및 권한 설정
COPY .env $AIRFLOW_HOME/.env
RUN chown airflow:airflow $AIRFLOW_HOME/.env && \
    chmod 600 $AIRFLOW_HOME/.env

# Slack Webhook 설정 스크립트 생성
RUN echo '#!/bin/bash\n\
airflow connections add "slack_webhook" \\\n\
    --conn-type "slack_webhook" \\\n\
    --conn-host "https://hooks.slack.com/services" \\\n\
    --conn-password "${SLACK_WEBHOOK_TOKEN}"' > $AIRFLOW_HOME/connections/setup_slack.sh && \
    chmod +x $AIRFLOW_HOME/connections/setup_slack.sh

# 초기화 스크립트 복사 및 권한 설정
COPY init-scripts/init.sh /init-scripts/init.sh
RUN chown airflow:airflow /init-scripts/init.sh && \
    chmod +x /init-scripts/init.sh && \
    sed -i 's/\r$//' /init-scripts/init.sh

# 소스 코드 복사 및 권한 설정
COPY src/ $AIRFLOW_HOME/src/
COPY dags/ $AIRFLOW_HOME/dags/
RUN chown -R airflow:airflow $AIRFLOW_HOME/src $AIRFLOW_HOME/dags

# airflow 사용자로 전환
USER airflow

CMD ["/init-scripts/init.sh"]