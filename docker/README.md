# Docker Setting

## test ##

- .env
    - HOST_IP: HOST IP 지정
- common-compose.yml
    - /workflow:/workflow: Host의 Source Root 경로
    - ${MLFLOW_RUNS_PATH}: MLFlow Experiment 저장 Host Volume 정보

## MLFlow ##
- .env
    - Database 정보, MLFlow Port 정보
    - ${MLFLOW_DATABASE_PATH}: MySQL 저장 Host Volume 정보
    - ${MLFLOW_RUNS_PATH}: MLFlow Experiment 저장 Host Volume 정보

## MongoDB ##

- mongo/.env
    - ${MONGODB_PATH}: Database Volume
- mongo/select_list.json
    - 머신러닝 관리콘솔의 select list 항목 고정을 위한 설정 값
