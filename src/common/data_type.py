from dataclasses import dataclass, field
from typing import List

import numpy as np

@dataclass
class MetricData:
    """평가 메트릭 데이터 객체."""
    confusion: np.ndarray = np.ndarray([])  # Confusion-Matrix
    accuracy: np.float64 = np.empty         # Accuracy
    precision: np.float64 = np.empty        # Precision
    recall: np.float64 = np.empty           # Recall
    f1: np.float64 = np.empty               # F1-Score
    roc_auc: np.float64 = np.empty          # ROC-AUC

    def verify(self) -> bool:
        """유효성 체크

        Returns:
            bool: 모든 값이 있으면 True, 없으면 False.
        """
        if ((not self.confusion.size) or
            (not self.accuracy) or
            (not self.precision) or
            (not self.recall) or
            (not self.f1) or
            (not self.roc_auc)):
            return False
        return True

    @staticmethod
    def str_confusion() -> str:
        """Return 'confusion'"""
        return 'confusion'

    @staticmethod
    def str_accuracy() -> str:
        """Return 'accuracy'"""
        return 'accuracy'

    @staticmethod
    def str_precision() -> str:
        """Return 'precision'"""
        return 'precision'

    @staticmethod
    def str_recall() -> str:
        """Return 'recall'"""
        return 'recall'

    @staticmethod
    def str_f1() -> str:
        """"Return 'f1'"""
        return 'f1'

    @staticmethod
    def str_roc_auc() -> str:
        """Return 'roc_auc'"""
        return 'roc_auc'

@dataclass
class ThresholdMetricData:
    """임계값에 대한 메트릭 정보"""
    threshold: float = None     # Threshold
    metric: MetricData = field(default_factory=MetricData)

    def verify(self) -> bool:
        """유효성 체크

        Returns:
            bool: 모든 값이 있으면 True, 없으면 False.
        """
        if ((not self.threshold) or
            (self.metric.verify() is False)):
            return False
        return True

@dataclass
class ScoreInfo:
    """Score 정보"""
    name: str = None        # Classifier Name
    threshold: float = None # Threshold
    metric: MetricData = field(default_factory=MetricData)

    def verify(self) -> bool:
        """유효성 체크

        Returns:
            bool: 모든 값이 있으면 True, 없으면 False.
        """
        if ((not self.name) or
            (not self.threshold) or
            (self.metric.verify() is False)):
            return False
        return True

    @staticmethod
    def str_name() -> str:
        """"Return 'name'"""
        return 'name'

    @staticmethod
    def str_threshold() -> str:
        """"Return 'threshold'"""
        return 'threshold'

    @staticmethod
    def str_metric() -> str:
        """"Return 'metric'"""
        return 'metric'

@dataclass
class MetricScoreInfo:
    """가장 높은 메트릭 이름과 Score 정보"""
    metric_name: str = None
    score_info: ScoreInfo = field(default_factory=ScoreInfo)

    def verify(self) -> bool:
        """유효성 체크

        Returns:
            bool: 모든 값이 있으면 True, 없으면 False.
        """
        if ((not self.metric_name) or
            (self.score_info.verify() is False)):
            return False
        return True

@dataclass
class BestModelScoreInfo:
    """모델의 메트릭 정보"""

    each_best_f1_score: List[ScoreInfo] = field(default_factory=list)
    """각 모델마다 가장 높은 F1-Score 정보"""

    each_best_roc_auc_score: List[ScoreInfo] = field(default_factory=list)
    """각 모델마다 가장 높은 ROC-AUC 정보"""

    highest_f1: ScoreInfo = field(default_factory=ScoreInfo)
    """모델 중에서 가장 높은 F1-Score 를 가진 모델 정보"""

    highest_roc_auc: ScoreInfo = field(default_factory=ScoreInfo)
    """모델 중에서 가장 높은 ROC-AUC 를 가진 모델 정보"""
