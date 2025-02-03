import os
import tempfile

# If set to 0, this disables the experimental new console output.
os.environ["RAY_AIR_NEW_OUTPUT"] = "0"

from ray import train
from sklearn.model_selection import train_test_split
from learning.classifier import Classifier
from common.data_type import MetricData

def objective_function(
    config, pipeline, name, x_data, y_data, x_test, y_test,
    base_param, objective_metric):
    """ray tune's objective function

    Args:
        config (ray.tune.config): search space parameters
        pipeline (Pipeline): sklearn.pipeline
        name (str): algorithm Name
        x_data (pd.DataFrame): train x_data
        y_data (pd.Series): train y_data
        x_test (pd.DataFrame): test x_data
        y_test (pd.Series): test y_data
        base_param (dict): 기본 설정할 parameter
        objective_metric (str): 달성할 metric. default 'f1-score'.
    """
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_data, y_data, test_size=0.1, stratify=y_data)
    
    # 랜덤포레스트는 early stopping 기능을 제공하지 않으므로 사용불가.
    if name != Classifier.RANDOMFOREST:
        config.update({'clf__early_stopping_rounds': 5})

    config.update(base_param)
    pipeline.set_params(**config)

    if name == Classifier.RANDOMFOREST:
        pipeline.fit(x_train, y_train)
    else:
        pipeline.fit(x_train, y_train,
                     clf__eval_set=[(x_valid, y_valid)])
    predict_result = pipeline.predict(x_test)
    result = MetricData.get_f1_score(y_test, predict_result)

    train.report({objective_metric: result})
