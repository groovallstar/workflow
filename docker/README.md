# Docker Setting

## develop ##

- .env
    - HOST_IP: HOST IP 지정
- develop/common-compose.yml
    - /workflow:/workflow: Host의 Source Root 경로
    - ${MLFLOW_RUNS_PATH}: MLFlow Experiment 저장 Host Volume 정보

## MLFlow ##
- .env
    - Database 정보, MLFlow Port 정보
- mlflow/docker-compose.yml
    - ${MLFLOW_DATABASE_PATH}: MySQL 저장 Host Volume 정보
    - ${MLFLOW_RUNS_PATH}: MLFlow Experiment 저장 Host Volume 정보

## MongoDB ##

- mongo/docker-compose.yml
    - : MongoDB Replication 저장 Host Volume 정보
        - ${MONGODB_REPL_1_PATH}
        - ${MONGODB_REPL_2_PATH}
        - ${MONGODB_REPL_3_PATH}

## Sacred + Omniboard ##

- sacred/docker-compose.yml
    - ${SACRED_DATABASE_PATH}: Sacred Experiment 저장 Host Volume 정보

- sacred/omniboard/config.json
    - ${HOST_IP} 지정
        - 여러개 사용할 경우 각각 지정
