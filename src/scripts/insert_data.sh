# !bin/bash

SCRIPT=/work/git/test/workflow/src/make_data/insert_data.py

# python $SCRIPT --dataset='iris' --database='test' --collection='iris.data' --date='202201'
# python $SCRIPT --dataset='digits' --database='test' --collection='digits.data' --date='202201'
# python $SCRIPT --dataset='wine' --database='test' --collection='wine.data' --date='202201'
# python $SCRIPT --dataset='breast_cancer' --database='test' --collection='breast_cancer.data' --date='202201'

python $SCRIPT --dataset='iris' --database='test' --collection='iris.data' --date='202202'
python $SCRIPT --dataset='digits' --database='test' --collection='digits.data' --date='202202'
python $SCRIPT --dataset='wine' --database='test' --collection='wine.data' --date='202202'
python $SCRIPT --dataset='breast_cancer' --database='test' --collection='breast_cancer.data' --date='202202'
