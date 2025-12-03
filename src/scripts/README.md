# Scripts 사용 가이드

ML 파이프라인 실행을 위한 Shell 스크립트 모음

## 스크립트 목록

### 1. insert_data.sh

sklearn Toy 데이터셋을 MongoDB에 저장

**파라미터**

| 파라미터 | 필수 | 설명 | 가능한 값 |
|---------|------|------|-----------|
| `--dataset` | O | 데이터셋 종류 | `iris`, `digits`, `wine`, `breast_cancer` |
| `--database` | O | 저장할 데이터베이스 명 | 문자열 |
| `--collection` | O | 저장할 컬렉션 명 | 문자열 |
| `--date` | O | 데이터 저장 날짜 | YYYYMM 형식 (예: `202501`) |

**사용 예시**

```bash
cd /workflow/src
bash scripts/insert_data.sh \
  --dataset iris \
  --database test \
  --collection iris.data \
  --date 202501
```

### 2. insert_table.sh

데이터셋의 컬럼(feature) 정보를 MongoDB에 저장

**파라미터**

| 파라미터 | 필수 | 형식 | 설명 |
|---------|------|------|------|
| `--data` | O | JSON | 데이터가 저장된 위치 정보 |
| `--table` | O | JSON | 컬럼 정보를 저장할 위치 |

**data JSON 구조**
- `database`: 데이터가 저장된 데이터베이스 명
- `collection`: 데이터가 저장된 컬렉션 명
- `start_date`: 시작 날짜 (YYYYMM)
- `end_date`: 종료 날짜 (YYYYMM)

**table JSON 구조**
- `database`: 컬럼 정보를 저장할 데이터베이스 명
- `collection`: 컬럼 정보를 저장할 컬렉션 명

**사용 예시**

```bash
cd /workflow/src
bash scripts/insert_table.sh \
  --data='{"database":"test","collection":"iris.data","start_date":"202501","end_date":"202501"}' \
  --table='{"database":"test","collection":"table"}'
```

### 3. train.sh / predict.sh

모델 학습 및 예측을 실행함. 두 스크립트는 동일한 파라미터를 사용

**파라미터**

#### 필수 파라미터

| 파라미터 | 형식 | 설명 |
|---------|------|------|
| `--data` | JSON | 학습/예측에 사용할 데이터 정보 |
| `--table` | JSON | 컬럼 정보가 저장된 위치 |
| `--classification_file_name` | 문자열 | 모델 설정 YAML 파일명 (예: `iris.yml`) |

**data/table JSON 구조**
- `database`: 데이터베이스 명
- `collection`: 컬렉션 명
- `start_date`: 시작 날짜 (YYYYMM)
- `end_date`: 종료 날짜 (YYYYMM)

#### MLflow 파라미터

| 파라미터 | 형식 | 설명 |
|---------|------|------|
| `--experiment` | 문자열 | MLflow Experiment 명 |
| `--run_name` | 문자열 | MLflow Run 이름 |

#### 데이터 설정

| 파라미터 | 형식 | 설명 |
|---------|------|------|
| `--seed` | 정수 | Random Seed 값 (재현성 보장) |
| `--split_ratio` | JSON | 데이터 분할 비율 (train/validation/test) |
| `--show_data` | 플래그 | 데이터 분할 정보 출력 |

#### 학습 관련

| 파라미터 | 형식 | 설명 |
|---------|------|------|
| `--train` | 플래그 | 모델 학습 실행 |
| `--train_with_tuning` | 플래그 | Ray Tune을 이용한 하이퍼파라미터 튜닝 후 학습<br>※ YAML 파일의 `hyper_parameter` 섹션이 최적값으로 갱신됨 |

#### 모델 저장/로드

| 파라미터 | 형식 | 설명 |
|---------|------|------|
| `--save_model` | JSON | 학습된 모델 저장 위치 (`database`, `collection`) |
| `--load_model` | JSON | 기존 모델 로드 위치 (`database`, `collection`, `start_date`, `end_date`) |

#### 평가 관련

| 파라미터 | 형식 | 설명 |
|---------|------|------|
| `--evaluate` | 플래그 | 예측 및 평가 실행 |
| `--show_metric_by_thresholds` | 문자열 | 지정한 임계값에서 평가 메트릭 출력<br>쉼표로 구분 (예: `0.3,0.5,0.7`) |
| `--show_optimal_metric` | 플래그 | 각 모델의 최적 F1-Score 임계값 출력 |
| `--find_best_model` | 플래그 | 전체 모델 중 F1-Score가 가장 높은 모델 출력 |

**평가 메트릭**
- Accuracy (정확도)
- Precision (정밀도)
- Recall (재현율)
- F1-Score
- ROC-AUC
- Miss Rate (미탐률)
- Fall Out (과탐률)

**사용 예시**

#### 기본 학습 및 평가

```bash
cd /workflow/src
bash scripts/train.sh \
  --experiment=iris_exp \
  --run_name=baseline \
  --data='{"database":"test","collection":"iris.data","start_date":"202501","end_date":"202501"}' \
  --table='{"database":"test","collection":"table","start_date":"202501","end_date":"202501"}' \
  --classification_file_name=iris.yml \
  --seed=42 \
  --train \
  --evaluate \
  --show_optimal_metric \
  --find_best_model
```

#### 하이퍼파라미터 튜닝 후 학습

```bash
cd /workflow/src
bash scripts/train.sh \
  --experiment=iris_tuning \
  --run_name=tuned_model \
  --data='{"database":"test","collection":"iris.data","start_date":"202501","end_date":"202501"}' \
  --table='{"database":"test","collection":"table","start_date":"202501","end_date":"202501"}' \
  --classification_file_name=iris.yml \
  --seed=42 \
  --train_with_tuning \
  --evaluate \
  --show_metric_by_thresholds=0.3,0.5,0.7 \
  --show_optimal_metric \
  --find_best_model \
  --save_model='{"database":"test","collection":"model"}'
```

#### 저장된 모델로 예측

```bash
cd /workflow/src
bash scripts/predict.sh \
  --experiment=iris_predict \
  --run_name=prediction \
  --data='{"database":"test","collection":"iris.data","start_date":"202501","end_date":"202501"}' \
  --table='{"database":"test","collection":"table","start_date":"202501","end_date":"202501"}' \
  --classification_file_name=iris.yml \
  --load_model='{"database":"test","collection":"model","start_date":"202501","end_date":"202501"}' \
  --evaluate \
  --show_optimal_metric
```
