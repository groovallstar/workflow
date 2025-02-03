"""evaluation.py"""
import inspect
from typing import Iterator
import numpy as np

from common.data_type import MetricData, ThresholdMetricData
from common.function import get_code_line

def get_probability_by_binarizer(threshold:float, probability) -> np.ndarray:
    """Threshold 값과 예측 확률을 이용해 예측 결과를 변경

    Args:
        threshold (float): 예측 결과를 결정할 임계값
        probability (numpy.ndarray): 예측 확률 값

    Raises:
        TypeError: 임계값 타입이 float 가 아닐 경우
        TypeError: probability 차원이 2차원이 아닐 경우

    Returns:
        numpy.ndarray: 임계값에 따라 변경된 예측 결과 리턴
    """
    from sklearn.preprocessing import Binarizer
    if isinstance(threshold, float) is False:
        raise TypeError('threshold must be float.',
                         get_code_line(inspect.currentframe()))

    if probability.ndim != 2:
        raise TypeError('probability must be two ndim.',
                         get_code_line(inspect.currentframe()))

    return Binarizer(threshold=threshold).fit_transform(
        probability[:, 1].reshape(-1, 1)).astype('int32').reshape(-1)

def get_evaluation_metric(y_test, predict) -> MetricData:
    """평가 지표 값을 dictionary 형태로 리턴하는 함수

    Args:
        y_test (pd.Series): 타겟 레이블 값.
        predict (pd.Series): 예측 값.

    Raises:
        BufferError: y_test, predict 가 없을 경우

    Returns:
        MetricData: confusion matrix, accuracy, precision, recall,\
                    f1 score, weighted f1 score, fbeta score(2, 0.5),\
                    roc-auc, miss rate, fall out 값
    """
    if (y_test is None) or (predict is None):
        raise BufferError('y_test or predict Value Empty.',
                         get_code_line(inspect.currentframe()))

    metric = MetricData(
            confusion=MetricData.get_confusion_matrix(y_test, predict),
            accuracy=MetricData.get_accuracy_score(y_test, predict),
            precision=MetricData.get_precision_score(y_test, predict),
            recall=MetricData.get_recall_score(y_test, predict),
            f1=MetricData.get_f1_score(y_test, predict),
            roc_auc=MetricData.get_roc_auc_score(y_test, predict),
            miss_rate=MetricData.get_miss_rate_score(y_test, predict),
            fall_out=MetricData.get_fall_out_score(y_test, predict),
            specificity=MetricData.get_specificity_score(y_test, predict),)
    return metric

def get_metric(
    y_test, predict, probability, threshold:float=0.0) -> MetricData:
    """Threshold 값에 따라 평가 지표 리턴

    Args:
        y_test (np.ndarray): 타겟 레이블 값
        probability (np.ndarray): 예측 확률 값
        threshold (float) : 임계값. None 이면 기본 0.5 값으로 평가

    Raises:
        TypeError: threshold가 float 타입이 아닐 경우

    Returns:
        MetricData: confusion matrix, accuracy, precision, recall,\
                    f1 score, roc-auc 값
    """
    # 임계 값이 없으면 기본 평가 메트릭 호출
    if threshold is None:
        return get_evaluation_metric(y_test=y_test, predict=predict)

    if isinstance(threshold, float) is False:
        raise TypeError('threshold Type Must Float.',
                        get_code_line(inspect.currentframe()))

    predict_by_threshold = get_probability_by_binarizer(
        threshold=threshold, probability=probability)
    return get_evaluation_metric(y_test, predict_by_threshold)

def get_metrics_by_threshold(
    y_test, predict, probability, thresholds: list
    ) -> Iterator[ThresholdMetricData]:
    """Threshold 값에 따라 평가 지표 제너레이터

    Args:
        y_test: Target 값
        predict: 예측 값
        probability: 예측 확률 값
        thresholds (list): 0-1 사이의 값으로 이루어진 list 값 e.g. [0.1,0.8]

    Raises:
        TypeError: thresholds 가 list 타입이 아닐 경우

    Returns:
        ThresholdMetricData : threshold, MetricData
    """
    if (thresholds is None) or (isinstance(thresholds, list) is False):
        raise TypeError('thresholds is None or Not List Type.')

    for th in thresholds:
        metric = get_metric(y_test, predict, probability, th)
        yield ThresholdMetricData(threshold=th, metric=metric)

def find_best_threshold(
    metric_function, y_test, probability, step=1000) -> float:
    """metric_function 의 평가 함수 결과를 통해 가장 높은 점수가 나오는 임계값 리턴.

    Args:
        metric_function (Callable Method): sklearn.metrics 의 평가 함수
        y_test: 타겟 레이블 값
        probability: 예측 확률 값.
        step (int, optional): 0에서 1.0 사이에서 계산할 간격. Defaults to 1000.
                              e.g. step=10 => 0~1.0 사이에서 10개를 계산함

    Raises:
        BufferError: 호출할 메트릭 함수가 없을 경우
        BufferError: y_test, probability 가 없을 경우
        BufferError: step 이 0 이하거나 1000 을 초과할 경우

    Returns:
        float: 평가 함수에서 가장 높은 점수의 임계값
    """
    threshold = -1
    if metric_function is None:
        raise BufferError('metric_function is None.',
                         get_code_line(inspect.currentframe()))

    if (y_test is None) or (probability is None):
        raise BufferError('y_test or probability is None.',
                         get_code_line(inspect.currentframe()))

    # step 범위: 0 < step < 1000
    if ((step <= 0) or (step > 1000)):
        raise BufferError('num_step value 0 < num_step < 1.0',
                         get_code_line(inspect.currentframe()))

    score_list = np.array([])
    floating_point = step / (step * step) # e.g. 10 -> 0.1, 100 -> 0.01
    for th in np.linspace(0, 1.0, num=step, endpoint=False):
        # 리스트 형식으로 사용 시 numpy float imprecision 문제가 있어서
        # 문자열로 변환 후 재사용 예) 0.30000000000000004
        th = float(f"{th:g}")
        predict_by_threshold = get_probability_by_binarizer(th, probability)

        # 메트릭 함수 호출
        metrics = metric_function(y_test, predict_by_threshold)
        score_list = np.append(score_list, metrics)

    if score_list.size:
        max_score_index = np.argmax(score_list)
        # max_score_index 가 0일 경우는 floating_point 에 곱할수 없으므로
        # 첫번째 인덱스로 대체함
        threshold = max_score_index * floating_point\
            if (max_score_index) else (1 * floating_point)
        threshold = float(f"{threshold:g}")

    if threshold == -1:
        raise BufferError('Get Best Threshold Failed.',
                         get_code_line(inspect.currentframe()))

    return threshold

def get_metric_with_best_score(
    metric_function, y_test, predict, probability) -> ThresholdMetricData:
    """metric_function 변수를 통해 전달받은 평가 함수와 가장 높은 점수의 임계값을
       이용하여 평가 함수 호출.

    Args:
        metric_function (Callable Method): sklearn.metrics 의 평가 함수
        y_test: 타겟 레이블 값
        predict: 예측 값
        probability: 예측 확률 값

    Raises:
        BufferError: 호출할 메트릭 함수가 없을 경우

    Returns:
        ThresholdMetricData: threshold, MetricData
    """
    if metric_function is None:
        raise BufferError('Callable Metric Function is None.',
                         get_code_line(inspect.currentframe()))

    best_threshold = find_best_threshold(
        metric_function, y_test, probability, step=1000)

    metric = get_metric(y_test, predict, probability, best_threshold)

    threshold_metric = ThresholdMetricData(
        threshold=best_threshold, metric=metric)
    return threshold_metric
