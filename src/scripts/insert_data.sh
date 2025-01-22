# !bin/bash

SCRIPT=$PYTHONPATH/make_data/insert_data.py

python $SCRIPT --dataset='iris' --database='test' --collection='iris.data' --date='202501'
python $SCRIPT --dataset='digits' --database='test' --collection='digits.data' --date='202501'
python $SCRIPT --dataset='wine' --database='test' --collection='wine.data' --date='202501'
python $SCRIPT --dataset='breast_cancer' --database='test' --collection='breast_cancer.data' --date='202501'

python $SCRIPT --dataset='iris' --database='test' --collection='iris.data' --date='202502'
python $SCRIPT --dataset='digits' --database='test' --collection='digits.data' --date='202502'
python $SCRIPT --dataset='wine' --database='test' --collection='wine.data' --date='202502'
python $SCRIPT --dataset='breast_cancer' --database='test' --collection='breast_cancer.data' --date='202502'
