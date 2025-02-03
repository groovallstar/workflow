# !bin/bash

# --experiment=test \
# --run_name=ExperimentName \
# --data='{"database":"test", "collection":"data", "start_date":"202501", "end_date":"202501"}' \
# --table='{"database":"test", "collection":"table", "start_date":"202501", "end_date":"202501"}' \
# --split_ratio='{"train": "0.8", "validation": "0.1", "test": "0.1"}' \
# --show_data \
# --seed=123 \
# --classification_file_name=dataset.yml \
# --train \
# --load_model='{"database":"test", "collection":"model", "start_date":"202501", "end_date":"202501"}' \
# --evaluate \
# --show_metric_by_thresholds="0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9" \
# --show_optimal_metric \
# --find_best_model \
# --save_model='{"database":"test", "collection":"model"}' \

python $PYTHONPATH/learning/pipeline.py --run_name="predict iris data" --experiment="iris" \
--data='{"database":"test", "collection": "iris.data", "start_date": "202501", "end_date": "202501"}' \
--table='{"database":"test", "collection": "iris.table", "start_date": "202501", "end_date": "202501"}' \
--split_ratio='{"test": "1.0"}' \
--classification_file_name=iris.yml \
--show_data \
--load_model='{"database":"test", "collection":"iris.model", "start_date":"202501", "end_date":"202501"}' \
--evaluate \
--show_metric_by_thresholds="0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9" \
--show_optimal_metric \
--find_best_model

python $PYTHONPATH/learning/pipeline.py --run_name="predict breast_cancer data" --experiment="breast_cancer" \
--data='{"database":"test", "collection": "breast_cancer.data", "start_date": "202501", "end_date": "202501"}' \
--table='{"database":"test", "collection": "breast_cancer.table", "start_date": "202501", "end_date": "202501"}' \
--split_ratio='{"test": "1.0"}' \
--classification_file_name=breast_cancer.yml \
--show_data \
--load_model='{"database":"test", "collection":"breast_cancer.model", "start_date":"202501", "end_date":"202501"}' \
--evaluate \
--show_metric_by_thresholds="0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9" \
--show_optimal_metric \
--find_best_model

python $PYTHONPATH/learning/pipeline.py --run_name="predict digits data" --experiment="digit" \
--data='{"database":"test", "collection": "digits.data", "start_date": "202501", "end_date": "202501"}' \
--table='{"database":"test", "collection": "digits.table", "start_date": "202501", "end_date": "202501"}' \
--split_ratio='{"test": "1.0"}' \
--classification_file_name=digits.yml \
--show_data \
--load_model='{"database":"test", "collection":"digits.model", "start_date":"202501", "end_date":"202501"}' \
--evaluate \
--show_metric_by_thresholds="0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9" \
--show_optimal_metric \
--find_best_model

python $PYTHONPATH/learning/pipeline.py --run_name="predict wine data" --experiment="wine" \
--data='{"database":"test", "collection": "wine.data", "start_date": "202501", "end_date": "202501"}' \
--table='{"database":"test", "collection": "wine.table", "start_date": "202501", "end_date": "202501"}' \
--split_ratio='{"test": "1.0"}' \
--classification_file_name=wine.yml \
--show_data \
--load_model='{"database":"test", "collection":"wine.model", "start_date":"202501", "end_date":"202501"}' \
--evaluate \
--show_metric_by_thresholds="0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9" \
--show_optimal_metric \
--find_best_model
