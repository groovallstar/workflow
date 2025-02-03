import numpy as np
import pytest

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, roc_auc_score

from common.data_type import MetricData, ThresholdMetricData

from learning.evaluation import get_evaluation_metric
from learning.evaluation import get_probability_by_binarizer
from learning.evaluation import get_metric
from learning.evaluation import get_metrics_by_threshold
from learning.evaluation import find_best_threshold
from learning.evaluation import get_metric_with_best_score
    
#@pytest.mark.skip()
def test_metric_score():
    """실제 메트릭 계산"""

    y_test  = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    predict = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    probability = np.array([
        [0.1, 0.9],[0.2, 0.8],[0.3, 0.7],[0.4, 0.6],[0.45, 0.55],
        [0.55, 0.45],[0.6, 0.4],[0.7, 0.3],[0.8, 0.2],[0.9, 0.1],
        [0.95, 0.05]])
    threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # 임계값에 따라 미리 계산한 예측값.
    calculated_predict = [
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]), # 0.1
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]), # 0.2
        np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]), # 0.3
        np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]), # 0.4
        np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]), # 0.5
        np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]), # 0.6
        np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), # 0.7
        np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), # 0.8
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # 0.9
    ]

    # 임계값에 따라 predict 값이 변하는지 체크.
    for th, correct_value in zip(threshold, calculated_predict):
        modify_predict = get_probability_by_binarizer(
            threshold=th, probability=probability)
        assert isinstance(modify_predict, np.ndarray)

        assert np.all(modify_predict == correct_value)

    # native 호출 결과와 method를 통한 메트릭 점수 체크.
    md = get_evaluation_metric(y_test=y_test, predict=predict)
    assert isinstance(md, MetricData)

    assert np.all(md.confusion == confusion_matrix(y_test, predict))
    assert md.accuracy == round(accuracy_score(y_test, predict), 8)
    assert md.precision == round(precision_score(y_test, predict), 8)
    assert md.recall == round(recall_score(y_test, predict), 8)
    assert md.f1 == round(f1_score(y_test, predict, average='weighted'), 8)

    # 임계값 증가에 따른 메트릭 점수와 native 호출 결과가 같은지 체크.
    for th, correct_value in zip(threshold, calculated_predict):
        md = get_metric(
            y_test=y_test, predict=predict,
            probability=probability, threshold=th)
        assert isinstance(md, MetricData)

        assert np.all(
            md.confusion == confusion_matrix(y_test, correct_value))
        assert md.accuracy == accuracy_score(y_test, correct_value)
        assert md.precision == precision_score(y_test, correct_value)
        assert md.recall == recall_score(y_test, correct_value)
        assert md.f1 == f1_score(y_test, correct_value, average='weighted')
        assert md.roc_auc == roc_auc_score(y_test, correct_value)

    # 임계값 리스트의 제너레이터를 통해 메트릭 점수 체크.
    gen_object = get_metrics_by_threshold(
        y_test=y_test, predict=predict,
        probability=probability, thresholds=threshold)
    for (tm, th, correct_value) in zip(
        gen_object, threshold, calculated_predict):
        assert isinstance(tm, ThresholdMetricData)

        assert tm.threshold == th
        assert np.all(
            tm.metric.confusion == confusion_matrix(y_test, correct_value))
        assert md.accuracy == round(accuracy_score(y_test, correct_value), 8)
        assert md.precision == round(precision_score(y_test, correct_value), 8)
        assert md.recall == round(recall_score(y_test, correct_value), 8)
        assert md.f1 == round(
            f1_score(y_test, correct_value, average='weighted'), 8)

    # 가장 높은 임계치와 메트릭 점수 체크.
    th = find_best_threshold(f1_score, y_test, probability)
    metric = get_metric(y_test=y_test, predict=predict,
               probability=probability, threshold=th)
    tm = get_metric_with_best_score(f1_score, y_test, predict, probability)
    assert isinstance(metric, MetricData)
    assert isinstance(tm, ThresholdMetricData)

    assert np.all(tm.metric.confusion == metric.confusion)
    assert tm.metric.accuracy == round(
            accuracy_score(y_test, correct_value), 8)
    assert tm.metric.precision == round(
        precision_score(y_test, correct_value), 8)
    assert tm.metric.recall == round(
        recall_score(y_test, correct_value), 8)
    assert tm.metric.f1 == round(
        f1_score(y_test, correct_value, average='weighted'), 8)
        
