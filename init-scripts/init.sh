#!/bin/bash

set -e  # 에러 발생시 스크립트 중단

INIT_FLAG="${AIRFLOW_INIT_FLAG:-/usr/local/test/initialized}"
echo "Init flag path: $INIT_FLAG"

# 디버토리 권한 확인
ls -la $(dirname $INIT_FLAG)

if [ ! -f "$INIT_FLAG" ]; then
    echo "Starting initialization..."
    
    # DB 초기화 전 잠시 대기
    sleep 5
    
    # 데이터베이스 초기화
    airflow db init
    
    # admin 사용자 생성
    airflow users create \
        --username admin \
        --firstname jungjoon \
        --lastname park \
        --role Admin \
        --email biasdrive@gmail.com \
        --password admin
    
    touch "$INIT_FLAG"
    echo "Initialization completed."
else
    echo "Found existing init flag at: $INIT_FLAG"
fi

# 서비스 순차적 시작
echo "Starting Airflow services..."
airflow scheduler &
sleep 10  # scheduler가 완전히 시작될 때까지 대기
airflow webserver --port 8080 & 
sleep 5   # webserver가 시작될 때까지 대기 #& background ㅊ
mlflow ui --host 0.0.0.0 --port 5050 &

wait