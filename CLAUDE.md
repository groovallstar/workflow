# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Machine learning pipeline system for training and evaluating classification models using scikit-learn toy datasets (iris, digits, wine, breast_cancer). The system provides a web-based management console for workflow orchestration, experiment tracking with MLflow, and hyperparameter tuning using Ray Tune.

## Architecture

### Core Components

**Web Management Console** ([src/web/](src/web/))
- FastAPI-based REST API and SSE server ([main.py](src/web/main.py:39))
- Jinja2 templates for UI rendering
- Server-sent events (SSE) for real-time task status updates via Redis pub/sub
- Routes: [controller.py](src/web/controller.py) (REST API), [view.py](src/web/view.py) (HTML views)

**Async Task Processing** ([src/celery/](src/celery/))
- Celery workers execute long-running ML tasks asynchronously ([tasks.py](src/celery/tasks.py))
- Three main task types: `insertdata`, `inserttable`, `pipeline`
- Tasks publish completion events to Redis channels

**ML Pipeline** ([src/learning/](src/learning/))
- [pipeline.py](src/learning/pipeline.py:181): Core `PipeLine` class orchestrates train/eval workflow
- [model.py](src/learning/model.py): `Model` class handles data splitting, training, evaluation
- [classifier.py](src/learning/classifier.py): Algorithm configuration from YAML files
- [tune.py](src/learning/tune.py): Ray Tune integration for Bayesian hyperparameter optimization
- Supports: RandomForest, LightGBM, XGBoost, CatBoost

**Data Management** ([src/common/container/](src/common/container/))
- [database_container.py](src/common/container/database_container.py:9): Loads data/columns from MongoDB
- [mongo.py](src/common/container/mongo.py): MongoDB connection and query building
- Data stored with `date`, `label`, and feature columns

**Infrastructure**
- MongoDB: Data and model storage
- MLflow: Experiment tracking and model registry
- Redis: Message broker for Celery and SSE pub/sub
- Docker: Separate client (Celery workers) and server (FastAPI) containers

### Data Flow

1. User configures task via web UI
2. FastAPI controller saves settings and dispatches Celery task
3. Celery worker executes Python script (insert_data.py, pipeline.py, etc.)
4. Worker publishes task completion ID to Redis channel
5. Web client receives SSE notification of task completion

### Model Configuration

Models configured via YAML files in [src/learning/classification/](src/learning/classification/):
- `hyper_parameter`: Parameters for direct training
- `tuning_parameter`: Ray Tune search space (min/max ranges)
  - `base_parameter`: Fixed params during tuning
  - `search_parameter`: Ranges for Bayesian optimization (must be 2-element lists)

## Common Commands

### Docker Environment

Start all services (MongoDB, Redis, MLflow, client, server):
```bash
cd docker/test
./start.sh
```

Individual services:
```bash
cd docker/mongo && ./start.sh     # MongoDB on port 37017
cd docker/redis && ./start.sh     # Redis on port 6379
cd docker/mlflow && ./start.sh    # MLflow on port 8088
```

Initialize Docker environment:
```bash
cd docker/test
./init.sh
```

### Development

Environment variables must be configured in [docker/test/.env](docker/test/.env:3):
- `HOST_IP`: Host machine IP address
- `MONGODB_URL`, `MLFLOW_URL`, `CELERY_BROKER_URL`, etc.

Run web server (inside server container):
```bash
cd /workflow/src/web
./run-fastapi.sh  # Starts uvicorn on port 8008
```

Run Celery worker (inside client container):
```bash
cd /workflow/src/celery
./run-celery.sh
```

### Data Management Scripts

Insert sklearn toy dataset into MongoDB:
```bash
cd /workflow/src
bash scripts/insert_data.sh --dataset iris --database test --collection iris.data --date 202501
```

Insert column metadata:
```bash
cd /workflow/src
bash scripts/insert_table.sh \
  --data='{"database":"test","collection":"iris.data","start_date":"202501","end_date":"202501"}' \
  --table='{"database":"test","collection":"table"}'
```

### Training and Prediction

Train models:
```bash
cd /workflow/src
bash scripts/train.sh \
  --experiment=test \
  --run_name=iris_experiment \
  --data='{"database":"test","collection":"iris.data","start_date":"202501","end_date":"202501"}' \
  --table='{"database":"test","collection":"table","start_date":"202501","end_date":"202501"}' \
  --classification_file_name=iris.yml \
  --seed=123 \
  --train \
  --evaluate \
  --show_metric_by_thresholds=0.5 \
  --show_optimal_metric \
  --find_best_model \
  --save_model='{"database":"test","collection":"model"}'
```

Train with hyperparameter tuning (Ray Tune):
```bash
cd /workflow/src
bash scripts/train.sh \
  [same params as above] \
  --train_with_tuning  # Updates YAML file with best params
```

Load and evaluate existing model:
```bash
cd /workflow/src
bash scripts/predict.sh \
  --load_model='{"database":"test","collection":"model","start_date":"202501","end_date":"202501"}' \
  [other params...] \
  --evaluate \
  --show_optimal_metric
```

### Testing

Run unit tests:
```bash
cd /workflow/src
python -m pytest tests/test_evaluation.py
```

## Key Implementation Notes

### Pipeline Parameters

All pipeline operations accept these parameter formats:

**--data/--table**: Dictionary with `database`, `collection`, `start_date`, `end_date`

**--split_ratio**: Dictionary with `train`, `validation`, `test` ratios (floats summing to 1.0)

**--show_metric_by_thresholds**: Comma-separated threshold values (e.g., `0.1,0.5,0.9`)

### MLflow Integration

- Parent runs track overall pipeline execution
- Nested child runs for each algorithm
- Artifacts: trained models, trace logs, YAML configs
- Models logged with sklearn format including pip dependencies
- Best model tagged in parent run with F1-score and threshold

### MongoDB Schema

**Data Collections**:
- Fields: `_id`, `date` (YYYYMM), `label`, feature columns
- Query by date range: `start_date` to `end_date`

**Table Collections**:
- Stores column names/metadata for feature selection

**Model Collections**:
- Stores model file paths and metadata for loading

**Web Collections**:
- `web.last_setting`: User's last UI configuration per page
- `web.select_list`: Database/collection filter rules for UI

### Hyperparameter Tuning

Uses Ray Tune with Bayesian optimization:
- Objective metric: F1-score (configurable)
- Search space defined in YAML `tuning_parameter.search_parameter`
- Updates YAML file in-place with best parameters
- Stratified train/validation split (90/10) for tuning
- Early stopping enabled for gradient boosting algorithms (LightGBM, XGBoost, CatBoost)

### Evaluation Metrics

Models evaluated using:
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Miss Rate (False Negative Rate), Fall Out (False Positive Rate)
- Confusion matrices
- Threshold-based metrics (finds optimal threshold per model)

### Code Organization Patterns

1. **Dependency Injection**: Settings and database connections initialized at app startup ([main.py](src/web/main.py:18))
2. **Container Pattern**: `DataContainer` abstraction for different data sources ([database_container.py](src/common/container/database_container.py:9))
3. **Query Builder**: MongoDB queries constructed through [QueryBuilder](src/common/container/mongo.py) class
4. **Logging**: `TraceLog` singleton for consistent logging across modules

## Recent Changes

Based on git history:
- Replaced manual hyperparameter tuning with Ray Tune Bayesian optimization
- Removed Sacred experiment tracking (replaced with MLflow)
- Simplified Docker setup (removed MongoDB replication)
- Restructured `data_type` dataclasses and MongoDB query API
