from dataclasses import dataclass, field

import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.metrics import f1_score, fbeta_score

@dataclass
class MetricData:
    """평가 메트릭 데이터 객체."""
    confusion: np.ndarray = field(
        default_factory=\
            lambda: np.zeros(0))        # Confusion-Matrix
    accuracy: np.float32 = 0.0          # Accuracy
    precision: np.float32 = 0.0         # Precision
    recall: np.float32 = 0.0            # Recall
    f1: np.float32 = 0.0                # F1-Score
    f1_weighted: np.float32 = 0.0       # F1-Score (Weighted)
    f2: np.float32 = 0.0                # F2-Score (F-beta score)
    f05: np.float32 = 0.0               # F0.5-Score (F-beta score)
    roc_auc: np.float32 = 0.0           # ROC-AUC
    miss_rate: np.float32 = 0.0         # Miss Rate (FNR:False Negative Rate)
    fall_out: np.float32 = 0.0          # Fall-out (FPR:False Positive Rate)
    specificity: np.float32 = 0.0       # Specificity (TNR:True Negative Rate)

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
    def str_f1_weighted() -> str:
        """"Return 'f1_weighted'"""
        return 'f1_weighted'

    @staticmethod
    def str_f2() -> str:
        """Return 'f2'"""
        return 'f2'

    @staticmethod
    def str_f05() -> str:
        """Return 'f05'"""
        return 'f05'

    @staticmethod
    def str_roc_auc() -> str:
        """Return 'roc_auc'"""
        return 'roc_auc'

    @staticmethod
    def str_miss_rate() -> str:
        """Return 'roc_auc'"""
        return 'miss_rate'

    @staticmethod
    def str_fall_out() -> str:
        """Return 'fall_out'"""
        return 'fall_out'

    @staticmethod
    def str_specificity() -> str:
        """REturn 'specificity'"""
        return 'specificity'

    @staticmethod
    def get_confusion_matrix(y_test, predict) -> np.ndarray:
        """Get Confusion Matrix"""
        return confusion_matrix(y_test, predict)

    @staticmethod
    def get_accuracy_score(y_test, predict) -> np.float32:
        """Get Accuracy Score"""
        return round(accuracy_score(y_test, predict), 8)

    @staticmethod
    def get_precision_score(y_test, predict) -> np.float32:
        """Get Precision Score"""
        return round(precision_score(y_test, predict, zero_division=0), 8)

    @staticmethod
    def get_recall_score(y_test, predict) -> np.float32:
        """Get Recall Score"""
        return round(recall_score(y_test, predict), 8)

    @staticmethod
    def get_f1_score(y_test, predict) -> np.float32:
        """Get F1 Score"""
        return round(f1_score(y_test, predict), 8)

    @staticmethod
    def get_f1_weighted_score(y_test, predict) -> np.float32:
        """Get Weighted F1 Score"""
        return round(f1_score(y_test, predict, average='weighted'), 8)

    @staticmethod
    def get_f2_score(y_test, predict) -> np.float32:
        """Get FBeta Score(2)"""
        return round(fbeta_score(y_test, predict, beta=2), 8)

    @staticmethod
    def get_f05_score(y_test, predict) -> np.float32:
        """Get FBeta Score(0.5)"""
        return round(fbeta_score(y_test, predict, beta=0.5), 8)

    @staticmethod
    def get_roc_auc_score(y_test, predict) -> np.float32:
        """Get Roc Auc Score"""
        
        if len(y_test.unique()) > 2:
            def roc_auc_score_multiclass(
                actual_class, pred_class, average = "macro") -> dict:
                """get roc_auc_score for multiclass"""
                unique_class = set(actual_class)
                roc_auc_dict = {}
                for per_class in unique_class:
                    other_class = [x for x in unique_class if x != per_class]
                    new_actual_class = [0 if x in other_class \
                                        else 1 for x in actual_class]
                    new_pred_class = [0 if x in other_class \
                                    else 1 for x in pred_class]
                    roc_auc = roc_auc_score(
                        new_actual_class, new_pred_class, average=average)
                    roc_auc_dict[per_class] = roc_auc
                return roc_auc_dict

            roc_auc = roc_auc_score_multiclass(y_test, predict)
        else:
            roc_auc = roc_auc_score(y_test, predict)
        
        return round(roc_auc, 8)

    @staticmethod
    def get_miss_rate_score(y_test, predict) -> np.float32:
        """Get Miss Rate (미탐률)"""
        matrix = MetricData.get_confusion_matrix(y_test, predict)
        if ((isinstance(matrix, np.ndarray) is True) and
            (matrix.size == 4)):
            tp = matrix[1][1]
            fn = matrix[1][0]
            return round((fn / (fn + tp)), 8)
        else:
            return -1.0

    @staticmethod
    def get_fall_out_score(y_test, predict) -> np.float32:
        """Get Fall Out (과탐률)"""
        matrix = MetricData.get_confusion_matrix(y_test, predict)
        if ((isinstance(matrix, np.ndarray) is True) and
            (matrix.size == 4)):
            tn = matrix[0][0]
            fp = matrix[0][1]
            return round((fp / (fp + tn)), 8)
        else:
            return -1.0

    @staticmethod
    def get_specificity_score(y_test, predict) -> np.float32:
        """Get Specificity (정상파일탐지율)"""
        matrix = MetricData.get_confusion_matrix(y_test, predict)
        if ((isinstance(matrix, np.ndarray) is True) and
            (matrix.size == 4)):
            tn = matrix[0][0]
            fp = matrix[0][1]
            return round((tn / (fp + tn)), 8)
        else:
            return -1.0

@dataclass
class ThresholdMetricData:
    """임계값에 대한 메트릭 정보"""
    threshold: float = 0.0 # Threshold
    metric: MetricData = field(default_factory=MetricData)

@dataclass
class ScoreInfo:
    """Score 정보"""
    name: str = '' # Classifier Name
    threshold: float = 0.0 # Threshold
    metric: MetricData = field(default_factory=MetricData)

@dataclass
class MetricScoreInfo:
    """가장 높은 메트릭 이름과 Score 정보"""
    metric_name: str = ''
    score_info: ScoreInfo = field(default_factory=ScoreInfo)

@dataclass
class BestModelScoreInfo:
    """모델의 메트릭 정보
       여러 모델 중 가장 높은 Score를 가진 모델 정보 저장
    """
    highest_f1: ScoreInfo = field(default_factory=ScoreInfo)
    highest_f1_weighted: ScoreInfo = field(default_factory=ScoreInfo)
    highest_f2: ScoreInfo = field(default_factory=ScoreInfo)
    highest_f05: ScoreInfo = field(default_factory=ScoreInfo)
