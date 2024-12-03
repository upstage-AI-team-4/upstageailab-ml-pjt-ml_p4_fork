FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    gcc \
    libc-dev \
    libgomp1 \
    curl \
    git \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 환경 변수 설정
ENV AIRFLOW_HOME=/usr/local/ml4
ENV AIRFLOW_INIT_FLAG=$AIRFLOW_HOME/initialized
ENV PYTHONPATH="${PYTHONPATH}:${AIRFLOW_HOME}"
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Airflow 홈 디렉터리 생성 및 권한 설정
RUN mkdir -p $AIRFLOW_HOME && \
    useradd -ms /bin/bash -d $AIRFLOW_HOME airflow && \
    chown -R airflow:airflow $AIRFLOW_HOME && \
    chmod -R 775 $AIRFLOW_HOME

WORKDIR $AIRFLOW_HOME

# Python 패키지 설치
RUN pip install --no-cache-dir \
    apache-airflow==2.7.1 \
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
    python-dotenv

# 디렉토리 생성 및 권한 설정
RUN mkdir -p $AIRFLOW_HOME/{dags,logs,mlruns,config,models,data,connections} && \
    mkdir -p /init-scripts && \
    chown -R airflow:airflow $AIRFLOW_HOME/{dags,logs,mlruns,config,models,data,connections} && \
    chmod -R 775 $AIRFLOW_HOME/{dags,logs,mlruns,config,models,data,connections} && \
    chmod -R 775 /init-scripts

# 로그 파일 생성 및 권한 설정
RUN touch $AIRFLOW_HOME/logs/{airflow.log,mlflow.log,streamlit.log} && \
    chown -R airflow:airflow $AIRFLOW_HOME/logs && \
    chmod -R 664 $AIRFLOW_HOME/logs/*.log

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