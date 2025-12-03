# Docker 설정 가이드

ML 파이프라인 시스템의 Docker 컨테이너 설정

## 디렉토리 구조

```
docker/
├── test/              # 개발/테스트 환경 (Client + Server)
├── mongo/             # MongoDB 서버
├── redis/             # Redis 서버
├── mlflow/            # MLflow 추적 서버
└── pytorch/           # PyTorch 환경 (선택사항)
```

## test (개발/테스트 환경)

Client(Celery Worker)와 Server(FastAPI) 컨테이너를 포함한 통합 개발 환경

### 주요 파일

**`.env`** - 환경 변수 설정

| 변수명 | 설명 | 예시 |
|--------|------|------|
| `HOST_IP` | 호스트 머신의 IP 주소 (필수) | `192.168.1.10` |
| `PYTHONPATH` | Python 소스 루트 경로 | `/workflow/src` |
| `MONGODB_URL` | MongoDB 연결 URL | `mongodb://root:root@${HOST_IP}:37017` |
| `MLFLOW_URL` | MLflow 서버 URL | `http://${HOST_IP}:8088` |
| `MLFLOW_RUNS_PATH` | MLflow 실험 저장 경로 | `/mlruns` |
| `CELERY_BROKER_URL` | Celery 브로커 URL | `redis://${HOST_IP}:6379/0` |
| `CELERY_RESULT_BACKEND` | Celery 결과 백엔드 URL | `redis://${HOST_IP}:6379/0` |
| `FASTAPI_PORT` | FastAPI 서버 포트 | `8008` |
| `REDIS_SUBSCRIBE_CHANNEL` | Redis 구독 채널 | `fastapi-event` |
| `REDIS_PUBLISH_CHANNEL` | Redis 발행 채널 | `fastapi-event` |

**`common-compose.yml`** - 공통 서비스 설정

- **볼륨 마운트**: `/workflow:/workflow` - 호스트의 소스 코드를 컨테이너에 마운트
- **MLflow 경로**: `${MLFLOW_RUNS_PATH}` - 실험 결과 저장 위치

**`docker-compose.yml`** - 서비스 정의

- **client_test**: Celery Worker 컨테이너
  - 포트: `2221:22` (SSH)
  - Shared Memory: 20GB (Ray Tune 사용)
- **server_test**: FastAPI 서버 컨테이너
  - 포트: `8008:8008` (HTTP), `2222:22` (SSH)

### 실행 방법

```bash
cd docker/test

# 초기 설정 (최초 1회)
./init.sh

# 서비스 시작
./start.sh

# 서비스 중지
docker-compose down
```

### 컨테이너 접속

```bash
# Client 컨테이너 접속
docker exec -it client_test /bin/bash

# Server 컨테이너 접속
docker exec -it server_test /bin/bash
```

## MLflow

MLflow 추적 서버 및 웹 UI를 제공

### 주요 파일

**`.env`** - 환경 변수 설정

| 변수명 | 설명 |
|--------|------|
| `MLFLOW_DATABASE_PATH` | MySQL 데이터베이스 저장 Host Volume 경로 |
| `MLFLOW_RUNS_PATH` | MLflow 실험 결과 저장 Host Volume 경로 |
| `MLFLOW_PORT` | MLflow UI 포트 (기본: 8088) |

### 구성 요소

- **MLflow 서버**: 실험 추적 및 모델 레지스트리
- **MySQL**: MLflow 메타데이터 저장소
- **Nginx**: 리버스 프록시

### 실행 방법

```bash
cd docker/mlflow

# 초기 설정
./init.sh

# 서비스 시작
./start.sh

# 웹 UI 접속
# http://{HOST_IP}:8088
```

## MongoDB

데이터셋, 모델 정보, 웹 설정을 저장하는 NoSQL 데이터베이스

### 주요 파일

**`.env`** - 환경 변수 설정

| 변수명 | 설명 |
|--------|------|
| `MONGODB_PATH` | MongoDB 데이터 저장 Host Volume 경로 |

**`select_list.json`** - 웹 관리 콘솔 설정

머신러닝 관리 콘솔의 select list 항목을 고정하기 위한 설정 파일  
이 파일의 내용은 `web.select_list` 컬렉션에 자동으로 삽입  

### 실행 방법

```bash
cd docker/mongo

# 서비스 시작
./start.sh

# MongoDB 접속
mongo mongodb://root:root@{HOST_IP}:37017
```

### 포트

- **37017**: MongoDB 서버 포트

## Redis

Celery 메시지 브로커 및 SSE Pub/Sub 채널로 사용됨

### 실행 방법

```bash
cd docker/redis

# 서비스 시작
./start.sh

# Redis CLI 접속
redis-cli -h localhost -p 6379
```

### 포트

- **6379**: Redis 서버 포트

## 전체 시스템 시작

모든 서비스를 순서대로 시작하는 방법:

```bash
# 1. MongoDB 시작
cd docker/mongo && ./start.sh && cd ../..

# 2. Redis 시작
cd docker/redis && ./start.sh && cd ../..

# 3. MLflow 시작
cd docker/mlflow && ./init.sh && ./start.sh && cd ../..

# 4. 개발 환경 시작 (Client + Server)
cd docker/test && ./init.sh && ./start.sh && cd ../..
```

또는 `docker/test/start.sh`만 실행하면 의존성 서비스가 이미 실행 중이라고 가정하고 Client와 Server를 시작함

## 포트 매핑 요약

| 서비스 | 포트 | 설명 |
|--------|------|------|
| MongoDB | 37017 | MongoDB 서버 |
| Redis | 6379 | Redis 서버 |
| MLflow | 8088 | MLflow 웹 UI |
| FastAPI | 8008 | 웹 관리 콘솔 |
| Client SSH | 2221 | Client 컨테이너 SSH |
| Server SSH | 2222 | Server 컨테이너 SSH |
