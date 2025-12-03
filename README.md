# ML Pipeline with Scikit-learn Toy Datasets

Scikit-learn의 Toy 데이터셋을 활용한 머신러닝 파이프라인 시스템입니다. 웹 기반 관리 콘솔을 통해 데이터 관리, 모델 학습, 예측 및 평가를 수행할 수 있습니다.

## 주요 기능

- **데이터 관리**: sklearn Toy 데이터셋(iris, digits, wine, breast_cancer)을 MongoDB에 저장 및 관리
- **모델 학습**: RandomForest, LightGBM, XGBoost, CatBoost 알고리즘 지원
- **하이퍼파라미터 튜닝**: Ray Tune을 이용한 Bayesian Optimization
- **실험 추적**: MLflow를 통한 실험 결과 및 모델 관리
- **웹 관리 콘솔**: FastAPI 기반 실시간 작업 모니터링
- **비동기 작업 처리**: Celery를 통한 백그라운드 작업 실행

## 프로젝트 구조

```
.
├── docker/               # Docker 컨테이너 설정
│   ├── test/            # 개발/테스트 환경 설정 (client, server)
│   ├── mongo/           # MongoDB 설정
│   ├── redis/           # Redis 설정
│   └── mlflow/          # MLflow 서버 설정
│
└── src/
   ├── celery/          # Celery 비동기 작업 처리
   ├── common/          # 공통 유틸리티 및 데이터 컨테이너
   ├── learning/        # ML 학습/예측 파이프라인
   ├── make_data/       # 데이터 저장 및 가공
   ├── scripts/         # 실행 스크립트 (insert_data, insert_table, train, predict)
   ├── tests/           # 단위 테스트
   └── web/             # FastAPI 웹 서버 및 관리 콘솔
```

## 빠른 시작

### 1. 환경 설정

`docker/test/.env` 파일에서 HOST_IP를 설정:

```bash
HOST_IP=your_host_ip_address
```

### 2. Docker 서비스 시작

```bash
cd docker/test
./start.sh
```

이 명령은 MongoDB, Redis, MLflow, Client(Celery), Server(FastAPI)를 모두 시작합니다.

### 3. 웹 관리 콘솔 접속

브라우저에서 `http://{HOST_IP}:8008` 접속

## 상세 문서

- [Docker 설정 가이드](docker/README.md)
- [스크립트 사용법](src/scripts/README.md)
- [개발자 가이드](CLAUDE.md)

## 기술 스택

- **언어**: Python 3.x
- **웹 프레임워크**: FastAPI
- **작업 큐**: Celery + Redis
- **데이터베이스**: MongoDB
- **ML 프레임워크**: scikit-learn, LightGBM, XGBoost, CatBoost
- **실험 추적**: MLflow
- **하이퍼파라미터 튜닝**: Ray Tune
- **컨테이너**: Docker, Docker Compose
