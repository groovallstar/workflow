import os
from typing import Iterator, Any
import yaml

class Classifier:
    """Classifier Class"""
    # define classifier prefix name
    PREFIX_NAME = 'clf'

    # define algorithm name
    RANDOMFOREST = 'RandomForest'
    LIGHTGBM = 'LightGBM'
    XGBOOST = 'XGBoost'
    CATBOOST = 'CatBoost'

    # define dictionary key name
    CLASSIFIER_OBJECT = 'classifier_object'
    HYPER_PARAMETER = 'hyper_parameter'
    TUNING_PARAMETER = 'tuning_parameter'
    BASE_PARAMETER = 'base_parameter'
    SEARCH_PARAMETER = 'search_parameter'

    def __init__(self, file_name: str, seed: int = 0):
        """Classification 설정 파일에 의해 Classifier 설정

        Args:
            file_name (YAML file name): YAML 파일명.
            seed (int, optional): 각 알고리즘의 난수 고정 값. Defaults to Zero.

        Raises:
            BufferError: file_name 이 없을 경우
            BufferError: YAML 파일에서 데이터를 읽어오지 못할 경우
            BufferError: classifier dict 초기화 실패할 경우
        """
        import inspect
        from common.function import get_code_line

        if not file_name:
            raise BufferError("file_name Empty.",
                              get_code_line(inspect.currentframe()))

        self._classifier_dict = {}

        yaml_file = Classifier.get_yml_file_path(file_name=file_name)
        with open(yaml_file, 'r', encoding='utf8') as f:
            informations = yaml.load(f, Loader=yaml.FullLoader)
        if not informations:
            raise BufferError("Parameters Empty.",
                              get_code_line(inspect.currentframe()))

        # yaml 포멧에 각 알고리즘 객체만 추가함
        if self.RANDOMFOREST in informations: # Set RandomForest.
            from sklearn.ensemble import RandomForestClassifier
            informations[self.RANDOMFOREST]\
                [self.CLASSIFIER_OBJECT] = RandomForestClassifier()
            # random_state default 값이 None임.
            if not seed:
                informations[self.RANDOMFOREST]\
                    [self.HYPER_PARAMETER]['clf__random_state'] = None
            else:
                informations[self.RANDOMFOREST]\
                    [self.HYPER_PARAMETER]['clf__random_state'] = seed

        if self.LIGHTGBM in informations: # Set LightGBM.
            from lightgbm import LGBMClassifier
            informations[self.LIGHTGBM]\
                [self.CLASSIFIER_OBJECT] = LGBMClassifier()
            # random_state default 값이 None임.
            if not seed:
                informations[self.LIGHTGBM]\
                    [self.HYPER_PARAMETER]['clf__random_state'] = None
            else:
                informations[self.LIGHTGBM]\
                    [self.HYPER_PARAMETER]['clf__random_state'] = seed

        if self.XGBOOST in informations: # Set XGBoost.
            from xgboost import XGBClassifier
            informations[self.XGBOOST]\
                [self.CLASSIFIER_OBJECT] = XGBClassifier()
            informations[self.XGBOOST]\
                [self.HYPER_PARAMETER]['clf__random_state'] = seed

        if self.CATBOOST in informations: # Set CatBoost.
            from catboost import CatBoostClassifier
            informations[self.CATBOOST]\
                [self.CLASSIFIER_OBJECT] = CatBoostClassifier()
            informations[self.CATBOOST]\
                [self.HYPER_PARAMETER]['clf__random_state'] = seed

        self._classifier_dict = informations
        # 유효성 체크.
        for value in self._classifier_dict.values():
            if not value:
                raise BufferError("Classifier List Initialize Failed.",
                                  get_code_line(inspect.currentframe()))
            # search parameter는 최소값, 최대값으로만 설정해야함.
            param = value[self.TUNING_PARAMETER][self.SEARCH_PARAMETER]
            for value in param.values():
                if len(value) != 2:
                    raise BufferError(
                        "Search Parameter Must Two Value. (min, max)",
                        get_code_line(inspect.currentframe()))

    @property
    def classifier_dict(self) -> dict:
        """getter"""
        return self._classifier_dict

    @classifier_dict.setter
    def classifier_dict(self, data):
        """setter"""
        self._classifier_dict = data

    def get_classifier_object(self, name: str) -> Any:
        """Return Classifier Object

        Args:
            name (str): 리턴할 알고리즘 객체

        Raises:
            ValueError: classifier dictionary가 초기화 되지 않을 경우

        Returns:
            object: 알고리즘 생성자 객체
        """
        if name not in self._classifier_dict:
            raise ValueError(f'{name} Not in Classifier Dict.')
        return self._classifier_dict[name][self.CLASSIFIER_OBJECT]

    def get_hyper_parameter(self, name: str) -> dict:
        """하이퍼 파라미터 섹션 리턴

        Args:
            name (str): 리턴할 파라미터의 알고리즘명

        Raises:
            ValueError: classifier dictionary가 초기화 되지 않을 경우

        Returns:
            dict: 하이퍼 파라미터 목록
        """
        if name not in self._classifier_dict:
            raise ValueError(f'{name} Not in Classifier Dict.')
        return self._classifier_dict[name][self.HYPER_PARAMETER]

    def get_base_parameter(self, name: str) -> dict:
        """기본 파라미터 리턴

        Args:
            name (str): 리턴할 파라미터의 알고리즘명

        Raises:
            ValueError: classifier dictionary가 초기화 되지 않을 경우

        Returns:
            dict: 기본 파라미터 목록
        """
        if name not in self._classifier_dict:
            raise ValueError(f'{name} Not in Classifier Dict.')
        return self._classifier_dict[name]\
            [self.TUNING_PARAMETER][self.BASE_PARAMETER]

    def get_search_parameter(self, name: str) -> dict:
        """Searching할 파라미터 리턴

        Args:
            name (str): 리턴할 파라미터의 알고리즘명

        Raises:
            ValueError: classifier dictionary가 초기화 되지 않을 경우

        Returns:
            dict: Search 파라미터 목록
        """
        if name not in self._classifier_dict:
            raise ValueError(f'{name} Not in Classifier Dict.')
        return self._classifier_dict[name]\
            [self.TUNING_PARAMETER][self.SEARCH_PARAMETER]

    def generator_classifier_keys(self) -> Iterator[str]:
        """generator classifier key list"""
        for key in self._classifier_dict:
            yield key

    def update_hyper_parameters(self, name: str, params: dict) -> None:
        """update hyper_parameter value in dictionary.

        Args:
            name (str): 업데이트할 알고리즘명
            params (dict): 업데이트할 파라미터 이름

        Raises:
            ValueError: 입력받은 알고리즘명이 없을 경우
        """
        if name not in self._classifier_dict:
            raise ValueError(f'{name} Not in Classifier Dict.')
        self._classifier_dict[name][self.HYPER_PARAMETER].update(params)

    def save_file(self, file_name: str) -> None:
        """yaml 파일 쓰기.
           초기화된 classifier 객체는 yaml 파일에 기록하지 않음

        Args:
            file_name (str): 저장할 파일명
        """
        import copy
        classifier_dict = copy.deepcopy(self._classifier_dict)
        # classifier object 정보는 제거하고 저장함
        for _, value in classifier_dict.items():
            value.pop(self.CLASSIFIER_OBJECT, None)
        yml_file = self.get_yml_file_path(file_name)
        with open(yml_file, 'w', encoding='utf8') as f:
            yaml.dump(classifier_dict, f, sort_keys=False)

    @staticmethod
    def get_yml_file_path(file_name: str = '') -> str:
        """yaml 파일 경로 리턴

        Args:
            file_name (str, optional): yaml 파일명.

        Returns:
            str: 파일 경로. file_name이 없을 경우 폴더명.
        """
        current_path = os.path.dirname(os.path.abspath(__file__))
        # 파일명이 없을 경우 폴더명 리턴.
        if not file_name:
            return os.path.join(current_path, 'classification')
        return os.path.join(current_path, 'classification', file_name)

    @staticmethod
    def get_classifier_version(classifier_name: str) -> str:
        """알고리즘 버전 리턴

        Args:
            classifier_name (str): 알고리즘 이름

        Returns:
            str: 버전 문자열
        """
        match(classifier_name):
            case Classifier.RANDOMFOREST:
                import sklearn
                return sklearn.__version__
            case Classifier.LIGHTGBM:
                import lightgbm
                return lightgbm.__version__
            case Classifier.XGBOOST:
                import xgboost
                return xgboost.__version__
            case Classifier.CATBOOST:
                import catboost
                return catboost.__version__
            case _:
                return ""

    @staticmethod
    def get_pip_package_string(classifier_name: str) -> str:
        """알고리즘 버전 리턴

        Args:
            classifier_name (str): 알고리즘 이름

        Returns:
            str: pip package spec 스타일 e.g. 'scikit-learn==0.0.0'
        """
        version = Classifier.get_classifier_version(classifier_name)
        match(classifier_name):
            case Classifier.RANDOMFOREST:
                return f'scikit-learn=={version}'
            case Classifier.LIGHTGBM:
                return f'lightgbm=={version}'
            case Classifier.XGBOOST:
                return f'xgboost=={version}'
            case Classifier.CATBOOST:
                return f'catboost=={version}'
            case _:
                return ""
