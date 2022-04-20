from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

class Classifier:
    """Classifier Class"""

    # define classifier list index
    from enum import IntEnum
    class Index(IntEnum):
        """Classifier List Index Enum Class"""
        CLASSIFIER = 0              # classifier
        HYPER_PARAMETER = 1         # hyper parameter
        BAYESIAN_PARAMETER = 2      # bayesian parameter
        GRID_SEARCH_PARAMETER = 3   # grid search parameter

    # define classifier prefix name
    PREFIX_NAME = 'clf'
    
    # define algorithm name
    RANDOMFOREST = 'RandomForest'
    LIGHTGBM = 'LightGBM'
    XGBOOST = 'XGBoost'
    CATBOOST = 'CatBoost'

    _classifier_dict = {}

    @staticmethod
    def get_classification_path_from_file_name(file_name: str) -> str:
        """파일명을 입력 받아 paramters 폴더 경로 return"""
        import os
        current_path = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_path, 'classification', file_name)

    @classmethod
    def get_classifier_dict(cls) -> dict:
        """get classifier dict"""
        return cls._classifier_dict

    @classmethod
    def initialize(
        cls, classification_file_name: str, seed: int = None) -> None:
        """Classification 설정 파일에 의해 Classifier 설정

        Args:
            classification_file_name (yaml file path): yaml 파일명.
            seed (int, optional): 각 알고리즘의 난수 고정 값. Defaults to None.

        Raises:
            BufferError: classification_file_name 이 없을 경우
            BufferError: yaml 파일에서 데이터를 읽어오지 못할 경우
            BufferError: classifier dict 초기화 실패할 경우
        """
        import inspect
        from common.function import get_code_line

        if not classification_file_name:
            raise BufferError("Classification File Name Empty.",
                            get_code_line(inspect.currentframe()))

        clf = None
        yaml_file = cls.get_classification_path_from_file_name(
            classification_file_name)
        with open(yaml_file) as f:
            import yaml
            clf = yaml.load(f, Loader=yaml.FullLoader)

        if not clf:
            raise BufferError("Classification File Open Failed.",
                            get_code_line(inspect.currentframe()))
        cls._classifier_dict.clear()

        for name in clf.keys():
            if name == cls.RANDOMFOREST:
                # Set RandomForest.
                randomforest = [{} for _ in cls.Index]

                randomforest[cls.Index.CLASSIFIER] = RandomForestClassifier()
                randomforest[cls.Index.HYPER_PARAMETER]=\
                    clf[cls.RANDOMFOREST]['hyper_parameter']
                randomforest[cls.Index.HYPER_PARAMETER]['clf__random_state']=\
                    seed

                randomforest[cls.Index.BAYESIAN_PARAMETER]=\
                    clf[cls.RANDOMFOREST]['bayesian_parameter']

                randomforest[cls.Index.GRID_SEARCH_PARAMETER]=\
                    clf[cls.RANDOMFOREST]['grid_search_parameter']

                cls._classifier_dict[name] = randomforest

            elif name == cls.LIGHTGBM:
                # Set LightGBM.
                lightgbm = [{} for _ in cls.Index]

                lightgbm[cls.Index.CLASSIFIER] = LGBMClassifier()
                lightgbm[cls.Index.HYPER_PARAMETER]=\
                    clf[cls.LIGHTGBM]['hyper_parameter']
                lightgbm[cls.Index.HYPER_PARAMETER]['clf__random_state']=\
                    seed

                lightgbm[cls.Index.BAYESIAN_PARAMETER]=\
                    clf[cls.LIGHTGBM]['bayesian_parameter']

                lightgbm[cls.Index.GRID_SEARCH_PARAMETER]=\
                    clf[cls.LIGHTGBM]['grid_search_parameter']

                cls._classifier_dict[name] = lightgbm

            elif name == cls.XGBOOST:
                # Set XGBoost.
                xgboost = [{} for _ in cls.Index]

                xgboost[cls.Index.CLASSIFIER] = XGBClassifier()
                xgboost[cls.Index.HYPER_PARAMETER]=\
                    clf[cls.XGBOOST]['hyper_parameter']
                # XGBoost 는 random state default가 0값임.
                if seed is None:
                    xgboost[cls.Index.HYPER_PARAMETER]['clf__random_state'] = 0
                else:
                    xgboost[cls.Index.HYPER_PARAMETER]['clf__random_state'] = \
                        seed

                xgboost[cls.Index.BAYESIAN_PARAMETER]=\
                    clf[cls.XGBOOST]['bayesian_parameter']

                xgboost[cls.Index.GRID_SEARCH_PARAMETER]=\
                    clf[cls.XGBOOST]['grid_search_parameter']

                cls._classifier_dict[name] = xgboost

            elif name == cls.CATBOOST:
                # Set CatBoost.
                catboost = [{} for _ in cls.Index]

                catboost[cls.Index.CLASSIFIER] = CatBoostClassifier()
                catboost[cls.Index.HYPER_PARAMETER]=\
                    clf[cls.CATBOOST]['hyper_parameter']
                # CatBoost 는 random state default가 0값임.
                if seed is None:
                    catboost\
                        [cls.Index.HYPER_PARAMETER]['clf__random_state'] = 0
                else:
                    catboost\
                        [cls.Index.HYPER_PARAMETER]['clf__random_state'] = seed

                catboost[cls.Index.BAYESIAN_PARAMETER]=\
                    clf[cls.CATBOOST]['bayesian_parameter']

                catboost[cls.Index.GRID_SEARCH_PARAMETER]=\
                    clf[cls.CATBOOST]['grid_search_parameter']

                cls._classifier_dict[name] = catboost

        if not cls._classifier_dict:
            raise BufferError("Classifier List Initialize Failed.",
                            get_code_line(inspect.currentframe()))
