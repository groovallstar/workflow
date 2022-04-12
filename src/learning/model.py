import os
import pickle
from typing import Iterator, Tuple, Union, List, Any, Callable
import inspect
from enum import IntEnum

from common.data_type import MetricData, ThresholdMetricData, MetricScoreInfo
from common.data_type import ScoreInfo, BestModelScoreInfo
from common.function import timeit, get_code_line

from learning.data_loader import DataLoader
from learning.classifier import Classifier

class Model(DataLoader):
    """모델 클래스

    Args:
        DataLoader (Class): 데이터 로드 Class
    """
    # define model list index
    class Index(IntEnum):
        """Model List Index Enum Class"""
        PIPE_LINE = 0       # pipeline
        MODEL_PATH = 1      # local model path
        PREDICT = 2         # predict result
        PREDICT_PROBA = 3   # predict probability result

    def __init__(
        self, x_train, y_train, x_validation, y_validation, x_test, y_test,
        seed):
        """생성자

        Args:
            x_train (pd.DataFrame): 학습용 x 데이터
            y_train (pd.Series): 학습용 y 데이터
            x_validation (pd.DataFrame): 검증용 x 데이터
            y_validation (pd.Series): 검증용 y 데이터
            x_test (pd.DataFrame): 예측용 x 데이터
            y_test (pd.Series): 예측용 y 데이터
            seed (int): 각 함수에 사용할 Random State 값
        """
        super().__init__(
            x_train, y_train, x_validation, y_validation, x_test, y_test)
        self._seed = seed

        self._model_dict = {} # 모델 클래스에서 가지고 있는 정보
        self._classifier_dict = {} # 모델 클래스에서 초기화 할 Classifier 정보

    @timeit
    def init_model(self, classification_file_name: str) -> None:
        """Classifier 및 모델 초기화. 명시적으로 호출해야함

        Args:
            classification_file_name (yaml Filename): Classifier 파일명

        Raises:
            BufferError: classification_file_name 이 없을 경우
            BufferError: Classifier가 초기화 되지 않을 경우
            BufferError: Model Dict이 초기화 되지 않을 경우
        """
        # Classifier 초기화.
        if classification_file_name is None:
            raise BufferError('classification_file_name Parameter Empty.',
                             get_code_line(inspect.currentframe()))

        Classifier.initialize(
            classification_file_name=classification_file_name,
            seed=self._seed)

        self._classifier_dict = Classifier.get_classifier_dict()
        if not self._classifier_dict:
            raise BufferError('Classifier Class Must Call initialize() first.',
                get_code_line(inspect.currentframe()))

        # Model Dictionary init.
        # {{'RandomForest': [pipeline, model path, predict, probablity]},
        #  {'LightGBM': [pipeline, model path, predict, probablity]},
        #  {'XGBoost': [pipeline, model path, predict, probablity]},
        #  {'CatBoost': [pipeline, model path, predict, probablity]}}

        for name in self._classifier_dict.keys():
            self._model_dict[name] = [None for _ in Model.Index]

        if not self._model_dict:
            raise BufferError('Model Dict Empty.',
                             get_code_line(inspect.currentframe()))

    @property
    def classifier_dict(self) -> dict:
        """classifier_dict getter"""
        return self._classifier_dict

    @classifier_dict.setter
    def classifier_dict(self, data) -> None:
        """classifier_dict setter"""
        self._classifier_dict = data

    def __make_pipeline_object(self, name: str) -> Any:
        """Sklearn Pipeline 객체 생성.

        Args:
            name (str): Classifier 이름

        Raises:
            NameError: Classifier Dictionary 에서 name을 못 찾을 경우
            BufferError: Sklearn Pipeline 초기화가 실패할 경우

        Description:
            catboost 는 이미 fit 된 객체에서 다시 set_param 을 할 경우
            "You can't change params of fitted model." 에러 발생.
            이 method 를 여러번 호출 할 경우 해당 에러 발생하게 되어
            객체를 그대로 쓰지 않고 deepcopy 하여 새로운 객체가 복사되도록 추가.
        """
        if ((isinstance(name, str) is False) and
            (name not in self._classifier_dict)):
            raise NameError(f"'{name}' not in classifier dictionary.")

        # sklearn pipline 객체 생성
        import copy
        from sklearn.pipeline import Pipeline
        pipeline_object = Pipeline([(
            Classifier.PREFIX_NAME,
            copy.deepcopy(
                self._classifier_dict[name][Classifier.Index.CLASSIFIER]))])

        if not pipeline_object:
            raise BufferError('sklearn Pipeline Object Not Initialized.',
                             get_code_line(inspect.currentframe()))

        hyper_parameter=\
            self._classifier_dict[name][Classifier.Index.HYPER_PARAMETER]
        if hyper_parameter:
            pipeline_object.set_params(**hyper_parameter)

        return pipeline_object

    @timeit
    def train(self, find_type: Union[str, int] = None) -> None:
        """학습.

        Args:
            find_type (Union[str, int], optional):
                str=지정된 Classifier 이름으로 학습.
                int=지정된 Classifier index로 학습. Defaults to None.

        Raises:
            NameError: Classifier를 못찾을 경우
        """
        if isinstance(find_type, str):
            return self.train_by_name(find_type)

        elif isinstance(find_type, int):
            for index, name in enumerate(self._model_dict):
                if index == find_type:
                    return self.train_by_name(name)
            else:
                raise NameError('Not Found Classifier Index.',
                                get_code_line(inspect.currentframe()))

        for name in self._model_dict.keys():
            self.train_by_name(name)

    @timeit
    def train_by_name(self, name: str) -> None:
        """Classifier의 이름을 통해 학습

        Args:
            name (str): Classifier 이름

        Raises:
            BufferError: Classifier or Model Dictionary 가 비어 있을 경우
                         학습 데이터가 비어 있을 경우
            NameError: dict에서 Classifier명이 없을 경우
        """
        if ((not self._classifier_dict) or (not self._model_dict)):
            raise BufferError('Classifier or Model Dict Empty.',
                              get_code_line(inspect.currentframe()))

        if ((name not in self._classifier_dict) or
            (name not in self._model_dict)):
            raise NameError(
                f"'{name}' not in classifier or model dictionary.")

        if (self._x_train.empty) or (self._y_train.empty):
            raise BufferError('Train Data is Empty.',
                              get_code_line(inspect.currentframe()))

        # sklearn pipline 객체 생성
        pipeline_object = self.__make_pipeline_object(name)
        pipeline_object.fit(self._x_train, self._y_train)

        self._model_dict[name][Model.Index.PIPE_LINE] = pipeline_object

    @timeit
    def evaluate(self, find_type: Union[str, int] = None) -> None:
        """평가.
        model dict 에 pipeline 객체가 있으면 예측 값과 예측 확률 값을
        각 model dict 에 추가함.

        Args:
            find_type (Union[str, int], optional):
                str=지정된 Classifier 이름으로 평가.
                int=지정된 Classifier index로 평가. Defaults to None.

        Raises:
            BufferError: Model Dictionary 가 비어 있을 경우
        """
        if not self._model_dict:
            raise BufferError("Model Dict Empty.",
                             get_code_line(inspect.currentframe()))

        if isinstance(find_type, str):
            # classifier 이름으로 예측 결과 생성
            return self.evaluate_by_name(find_type)

        elif isinstance(find_type, int):
            # 인덱스로 예측 결과 생성
            for index, name in enumerate(self._model_dict):
                if index == find_type:
                    return self.evaluate_by_name(name)
            else:
                return

        for name in self._model_dict.keys():
            self.evaluate_by_name(name)

    @timeit
    def evaluate_by_name(self, name: str) -> None:
        """Classifier 이름으로 평가.

        Args:
            name (str): Classifier 이름.

        Raises:
            BufferError: Model Dictionary 가 비어 있을 경우
            NameError: dict에서 Classifier 명이 없을 경우
            BufferError: 테스트 데이터가 없을 경우
            BufferError: 파이프라인 객체 초기화 실패할 경우
        """
        if not self._model_dict:
            raise BufferError(f"'{name}' Model Dict empty.",
                             get_code_line(inspect.currentframe()))

        if name not in self._model_dict:
            raise NameError(f"'{name}' not in Model Dict.",
                             get_code_line(inspect.currentframe()))

        if self._x_test.empty:
            raise BufferError(f"'{name}' x_test empty.",
                            get_code_line(inspect.currentframe()))

        pipeline = self._model_dict[name][Model.Index.PIPE_LINE]
        if not pipeline:
            raise BufferError(f"'{name}' pipeline Empty.",
                             get_code_line(inspect.currentframe()))

        self._model_dict[name][Model.Index.PREDICT]=\
            pipeline.predict(self._x_test)
        self._model_dict[name][Model.Index.PREDICT_PROBA]=\
            pipeline.predict_proba(self._x_test)
        return

    @timeit
    def grid_search(self, name: str) -> dict:
        """하이퍼 파라미터 찾기 (GridSearchCV 방식)

        Args:
            name (str): Classifier 명
            n_splits (int, optional): 데이터 분할 개수. Defaults to 5.

        Raises:
            BufferError: Classifier or Model Dictionary 가 비어 있을 경우
            NameError: dict에서 Classifier 명이 없을 경우
            BufferError: 학습 데이터가 비어 있을 경우
        """
        if ((not self._classifier_dict) or (not self._model_dict)):
            raise BufferError('Classifier or Model Dict Empty.',
                              get_code_line(inspect.currentframe()))

        if ((name not in self._classifier_dict) or
            (name not in self._model_dict)):
            raise NameError(
                f"'{name}' not in classifier or model dictionary.")

        if (self._x_train.empty) or (self._y_train.empty):
            raise BufferError('Train Data is Empty.',
                              get_code_line(inspect.currentframe()))

        pipeline_object = self.__make_pipeline_object(name)
        grid_search_parameter=\
            self._classifier_dict[name][Classifier.Index.GRID_SEARCH_PARAMETER]
        from ray.tune.sklearn import TuneGridSearchCV
        tune_search = TuneGridSearchCV(
                        pipeline_object,
                        grid_search_parameter,
                        scoring='f1',
                        n_jobs=-1)
        tune_search.fit(self._x_train, self._y_train)
        return tune_search.best_params_

    @timeit
    def get_hyper_parameters_by_bo(self, name: str) -> dict:
        """Bayesian Optimizer 방식으로 하이퍼 파라미터 찾기

        Args:
            name (str): Classifier 명

        Raises:
            BufferError: Classifier or Model Dict 가 비어 있을 경우
            NameError: dictionary 에 key 가 없을 경우
            BufferError: 학습 데이터가 없을 경우
            BufferError: 베이지안 최적화 파라미터가 없을 경우

        Returns:
            dict: 하이퍼 파라미터 정보
        """
        if ((not self._classifier_dict) or (not self._model_dict)):
            raise BufferError('Classifier or Model Dict Empty.',
                              get_code_line(inspect.currentframe()))

        if ((name not in self._classifier_dict) or
            (name not in self._model_dict)):
            raise NameError(
                f"'{name}' not in classifier or model dictionary.")

        if (self._x_train.empty) or (self._y_train.empty):
            raise BufferError('Train Data is Empty.',
                              get_code_line(inspect.currentframe()))

        search_params=\
            self._classifier_dict[name][Classifier.Index.BAYESIAN_PARAMETER]
        if not search_params:
            raise BufferError(f"'{name}' search_params Empty.")

        pipeline_object = self.__make_pipeline_object(name)

        from ray.tune.sklearn import TuneSearchCV
        tune_search = TuneSearchCV(
                        pipeline_object,
                        search_params,
                        search_optimization="bayesian",
                        scoring='f1')
        tune_search.fit(self._x_train, self._y_train)
        return tune_search.best_params_

    @timeit
    def cross_validation(
        self, name: str, n_splits: int=5,
        scoring: str='f1', shuffle=True) -> float:
        """교차 검증.

        Args:
            name (str): Classifier 명
            n_split (int, optional): 교차 검증 수행 횟수. Defaults to 5.
            scoring (str, optional): 교차 검증 시 평가 메트릭.
                                     Defaults to 'f1'.
            shuffle (bool, optional): 교차 검증 시 데이터 셔플링 여부.
                                      Defaults to True.

        Raises:
            BufferError: 검증 데이터가 없을 경우
            BufferError: Classifier or Model Dict 가 비어 있을 경우
            BufferError: 파이프라인 객체가 없을 경우

        Returns:
            float: 지정한 평가 메트릭의 평균값
        """
        if (self._x_validation.empty) or (self._y_validation.empty):
            import inspect
            raise BufferError('Validation Data Empty.',
                              get_code_line(inspect.currentframe()))

        if ((not self._classifier_dict) or (not self._model_dict)):
            raise BufferError('Classifier or Model Dict Empty.',
                              get_code_line(inspect.currentframe()))
        
        pipeline_object = self._model_dict[name][Model.Index.PIPE_LINE]
        if not pipeline_object:
            raise BufferError('pipeline Empty.',
                              get_code_line(inspect.currentframe()))
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle)
        result = cross_val_score(
            estimator=pipeline_object,
            X=self._x_validation,
            y=self._y_validation,
            cv=stratified_kfold,
            scoring=scoring)

        return result.mean()

    @timeit
    def load_model(self, model: dict, table: dict) -> None:
        """이미 생성된 모델 파일을 로드하여 pipeline 객체 저장

        Args:
            model (dict): 'database', 'collection', 'start_date', 'end_date'
            table (dict): 'database', 'collection', 'start_date', 'end_date'

        Raises:
            BufferError: 저장된 모델 경로 리스트에 값이 없을 경우
            FileNotFoundError: 실제 경로에 파일이 없을 경우
            BufferError: 모델 로드에 실패할 경우
            TypeError: 로드한 모델 파일이 tuple이 아닐 경우
            BufferError: 파이프라인 객체 로드 실패할 경우
        """
        if (not self._classifier_dict) or (not self._model_dict):
            raise ValueError('Classifier or Model Dict Empty.',
                             get_code_line(inspect.currentframe()))

        from common.container.mongo import Collection
        load_model_list = Collection.get_model_list_from_dict(
            model=model, table=table)

        if not load_model_list:
            raise BufferError('Load Model List Empty.',
                              get_code_line(inspect.currentframe()))

        for file in load_model_list:
            if not os.path.exists(file):
                raise FileNotFoundError(f'{file} not found',
                                        get_code_line(inspect.currentframe()))

        for file in load_model_list:
            model_object = None
            with open(file, "rb") as f:
                model_object = pickle.load(f)

            if model_object is None:
                raise BufferError(f'{file} Load Failed',
                                  get_code_line(inspect.currentframe()))

            if ((isinstance(model_object, tuple) is False) or
                (len(model_object) != 2)):
                raise TypeError(f'{file} Load Failed',
                                get_code_line(inspect.currentframe()))

            (name, pipeline_object) = model_object
            if (not name) or (not pipeline_object):
                raise BufferError(f'{name} Pipeline Object Empty.',
                                  get_code_line(inspect.currentframe()))

            self._model_dict[name][Model.Index.MODEL_PATH] = file
            self._model_dict[name][Model.Index.PIPE_LINE] = pipeline_object

    @timeit
    def set_save_model_file(self, name: str, model_file: str) -> None:
        """저장된 학습 모델 경로를 model 객체에 저장.

        Args:
            name (str): Classifier 명
            model_file (str): 저장된 모델 파일 경로

        Raises:
            BufferError: model dict 가 비어 있을 경우
            FileNotFoundError: 모델 파일이 해당 경로에 없을 경우
            BufferError: 모델 pickle 로드 실패할 경우
            TypeError: 로드한 pickle이 tuple 타입이 아닐 경우
            TypeError: 로드한 pickle 0번째 index의 Classifier 명이 다를 경우
        """
        if not self._model_dict:
            raise BufferError(f"'{name}' Model Dict Empty.",
                             get_code_line(inspect.currentframe()))

        # 실제 모델이 존재하는지 체크.
        if os.path.exists(model_file) is False:
            raise FileNotFoundError(f'{model_file} Not Found.',
                                    get_code_line(inspect.currentframe()))

        model_object = None
        with open(model_file, "rb") as f:
            model_object = pickle.load(f)

        if model_object is None:
            raise BufferError(f'{model_file} Pickle Load Failed.',
                              get_code_line(inspect.currentframe()))

        if ((isinstance(model_object, tuple) is False) or
            (len(model_object) != 2)):
            raise TypeError(f'{model_file} Wrong File Type.',
                            get_code_line(inspect.currentframe()))

        if name != model_object[0]:
            raise TypeError(f'{name} != {model_object[0]}.',
                            get_code_line(inspect.currentframe()))

        self._model_dict[name][Model.Index.MODEL_PATH] = model_file

    def dump_model(self, model: dict) -> None:
        """model list의 전체 pipeline을 파일로 저장.
        Args:
            model (dict): 'database', 'collection', 'path'
        Returns:
            list: 모델 저장 경로 list
        """
        if not self._model_dict:
            raise BufferError('Model Dict Empty.',
                              get_code_line(inspect.currentframe()))

        if ('path' not in model):
            raise BufferError("model dict not in 'path'",
                              get_code_line(inspect.currentframe()))

        save_path = model['path']
        if (not save_path) or (isinstance(save_path, str) is False):
            raise BufferError("save_path invalid.",
                              get_code_line(inspect.currentframe()))

        if os.path.isdir(save_path) is False:
            os.mkdir(save_path)

        for name, pipe_line in self.get_pipelines_with_name():
            save_model_file = os.path.join(save_path, name + '.pkl')
            dump_file_tuple = (name, pipe_line)
            with open(save_model_file, "wb") as f:
                pickle.dump(dump_file_tuple, f)
            self.set_save_model_file(name=name, model_file=save_model_file)

        return

    @timeit
    def save_model_information_to_database(
        self, model: dict, data: dict, table: dict,
        save_path_list: list) -> None:
        """학습한 모델 정보를 Database에 저장.

        Args:
            model (dict): 'database', 'collection'
            data (dict): 'database', 'collection', 'start_date', 'end_date'
            table (dict): 'database', 'collection', 'start_date', 'end_date'
            save_path_list (list): 저장된 모델 경로.

        Description:
            data 파라미터를 통해 모델 컬렉션의 데이터 시작/종료 날짜 쿼리
            table 파라미터를 통해 모델 컬렉션의 table object id 값 쿼리
            save_path_list를 통해 로컬 경로 리스트를 database에 저장

        Raises:
            BufferError: 모델 저장 경로 리스트가 비어 있을 경우
            FileNotFoundError: 모델 파일이 해당 경로에 없을 경우
            BufferError: 모델 pickle 로드 실패할 경우
            TypeError: 로드한 pickle이 tuple 타입이 아닐 경우
        """

        if not save_path_list:
            raise BufferError('File Path List Empty.',
                              get_code_line(inspect.currentframe()))

        # 실제 모델이 존재하는지 체크.
        for file in save_path_list:
            if os.path.exists(file) is False:
                raise FileNotFoundError(f'{file} Not Found.',
                                        get_code_line(inspect.currentframe()))

            model_object = None
            with open(file, "rb") as f:
                model_object = pickle.load(f)

            if model_object is None:
                raise BufferError(f'{file} Pickle Load Failed.',
                                  get_code_line(inspect.currentframe()))

            if ((isinstance(model_object, tuple) is False) or
                (len(model_object) != 2)):
                raise TypeError(f'{file} Wrong File Type.',
                                get_code_line(inspect.currentframe()))

        # MongoDB에 모델 정보 저장.
        from common.container.mongo import Collection
        Collection.insert_model_information_from_dict(
            data=data, table=table, model=model,
            model_save_path_list=save_path_list)

    @timeit
    def get_evaluate_metrics_by_name(
        self, name:str, thresholds:list) -> Iterator[ThresholdMetricData]:
        """평가 지표 출력 함수

        Args:
            find_type (Union[str, int]): str=Classifier 명으로 평가 지표 출력.
                                         int=인덱스로 평가 지표 출력.
            thresholds (list): threshold list값을 이용한 평가 지표 출력.\
                              e.g. [0.3, 0.4, 0.6, 0.7]

        Raises:
            BufferError: classifier or model dict 가 비어 있을 경우
            TypeError: 파라미터가 잘못될 경우
            NameError: dictionary 에 key 가 없을 경우

        Yields:
             Iterator[ThresholdMetricData]: threshold, MetricData
        """
        if (not self._classifier_dict) or (not self._model_dict):
            raise BufferError('Classifier or Model Dict Empty.')

        if ((isinstance(name, str) is False) or
            (isinstance(thresholds, list) is False)):
            raise TypeError('name or thresholds Wrong Type.')

        if name not in self._model_dict:
            raise NameError(f'Model Dict not in {name}.')

        from learning.evaluation import get_metrics_by_threshold
        metric_with_threshold = get_metrics_by_threshold(
            y_test=self._y_test,
            predict=self._model_dict[name][Model.Index.PREDICT],
            probability=self._model_dict[name][Model.Index.PREDICT_PROBA],
            thresholds=thresholds)
        return metric_with_threshold

    @timeit
    def get_optimal_metrics(
        self, name: str, metric_name_list:list) -> Iterator[MetricScoreInfo]:
        """임계값을 증가시키면서 평가 메트릭이 가장 높은 값을 리턴함

        Args:
            name (str): Classifier 명
            metric_name_list (list): 평가할 메트릭 이름

        Raises:
            BufferError: Classifier 이름이 없을 경우
            BufferError: 평가할 함수를 못 찾을 경우
            TypeError: 하나의 Classifier에서 가장 높은 메트릭 정보가 아닐 경우

        Yields:
            Iterator[MetricScoreInfo]: MetricScoreInfo 데이터 클래스
        """

        if not name:
            raise BufferError('Classifier Name is None.',
                         get_code_line(inspect.currentframe()))

        import sklearn.metrics
        # 각각의 모델에서 f1-score 가 높은 threshold와 metric을 구함.
        for metric_name in metric_name_list:
            metric_function = None
            if metric_name == MetricData.str_f1():
                metric_function = sklearn.metrics.f1_score

            if metric_function is None:
                raise BufferError('Metric Function Not Found.',
                                  get_code_line(inspect.currentframe()))

            best_score_for_metric = self.get_best_score_for_model(
                metric_function, find_type=name)

            # 이름으로 찾을 경우는 결과값이 하나여야 함.
            if len(best_score_for_metric) != 1:
                raise TypeError('Best Score Result Must Be One.',
                                get_code_line(inspect.currentframe()))
            metric_data_info = MetricScoreInfo(
                metric_name=metric_name,
                score_info=best_score_for_metric[0])

            yield metric_data_info

    @timeit
    def get_highest_score_for_model(self) -> BestModelScoreInfo:
        """평가지표 중 가장 높은 score 를 가진 모델과 임계값을 찾는 함수
           F1-Score 가 가장 높은 모델과 임계값 출력

        Returns:
            BestModelScoreInfo: 각 모델의 f1-score 최고 메트릭 및 임계치,
                                모델 중 가장 높은 f1-score 모델 메트릭 정보
        """
        if (not self._classifier_dict) or (not self._model_dict):
            raise ValueError('Classifier or Model Dict Empty.',
                             get_code_line(inspect.currentframe()))

        def get_highest_score_model(
            metric_name: str, each_best_score_list: list) -> ScoreInfo:
            """각 모델에서 평가했던 metric 중 score list 에서 
               가장 높은 score 의 임계값을 통해 전체 metric 출력.

            Args:
                metric_name (str): 메트릭 이름
                each_best_score_list (list): 각각 가장 높은 점수가 기록된 리스트

            Returns:
                ScoreInfo: (Model name, threshold, MetricData)
            """
            import numpy as np
            # 리스트에서 score 가 가장 높은 인덱스를 얻음
            index = np.argmax(
                [getattr(score.metric, metric_name)\
                            for score in each_best_score_list])

            name = each_best_score_list[index].name
            best_threshold = each_best_score_list[index].threshold

            # 해당 threshold 로 예측값 조절
            from learning.evaluation import get_metric
            best_metric = get_metric(
                y_test=self._y_test,
                predict=self._model_dict[name][Model.Index.PREDICT],
                probability=self._model_dict[name][Model.Index.PREDICT_PROBA],
                threshold=best_threshold)
            score_info = ScoreInfo(
                name=name, threshold=best_threshold, metric=best_metric)
            return score_info

        import sklearn.metrics

        # 각각의 모델에서 f1-score가 높은 threshold와 metric을 구함.
        each_best_f1_score = self.get_best_score_for_model(
            sklearn.metrics.f1_score)

        # 모델 중에서도 높은 metric을 가진 모델 하나를 구함.
        highest_f1 = get_highest_score_model(
            MetricData.str_f1(), each_best_f1_score)

        best_model_info = BestModelScoreInfo(
            each_best_f1_score=each_best_f1_score,
            highest_f1=highest_f1)
        return best_model_info

    @timeit
    def get_best_score_for_model(
        self, metric_function:Callable,
        find_type: Union[str, int] = None) -> List[ScoreInfo]:
        """각 Classifier 별 가장 높은 점수 리스트 전달

        Args:
            metric_function (Callable Method): 점수를 평가할 메트릭 함수 포인터
            find_type (Union[str, int], optional):
                str=지정된 Classifier 이름.
                int=지정된 Classifier index. Defaults to None.
        Raises:
            BufferError: 호출할 메트릭 함수가 없을 경우
            BufferError: model dict 가 비어 있을 경우
            BufferError: ThresholdMetricData 데이터의 유효성 체크가 실패할 경우

        Returns:
            List: ScoreInfo (Model name, threshold, MetricData)
        """
        best_score_list = []
        if metric_function is None:
            raise BufferError('Callable Metric Function is None.',
                         get_code_line(inspect.currentframe()))

        if not self._model_dict:
            raise BufferError('Model Dict Empty.',
                             get_code_line(inspect.currentframe()))

        target_model = {}
        if isinstance(find_type, str):
            target_model[find_type] = self._model_dict[find_type]
        elif isinstance(find_type, int):
            # classifier 이름으로 찾기
            for index, name in enumerate(self._model_dict):
                if index == find_type:
                    target_model[name] = self._model_dict[name]
        else:
            target_model = self._model_dict

        # 모델별로 임계값, 점수에 대한 리스트 생성
        from learning.evaluation import get_metric_with_best_score
        for name, model in target_model.items():
            metric_data = get_metric_with_best_score(
                metric_function=metric_function,
                y_test=self._y_test,
                predict=model[Model.Index.PREDICT],
                probability=model[Model.Index.PREDICT_PROBA])
            if metric_data.verify() is False:
                raise BufferError(f"model=({name}) Get Best Score Failed.")

            score_info = ScoreInfo(
                name=name,
                threshold=metric_data.threshold,
                metric=metric_data.metric)
            best_score_list.append(score_info)

        return best_score_list

    def get_pipeline_with_name(self, name) -> Tuple[str, Any]:
        """Classifier 이름으로 파이프라인 객체 리턴.

        Raises:
            BufferError: Model Dict 가 비어 있을 경우

        Return:
            Tuple: (Classifier 이름, sklearn.Pipeline 객체)
        """
        if not self._model_dict:
            raise BufferError('Model Dict Not Found.',
                           get_code_line(inspect.currentframe()))
        return (name, self._model_dict[name][Model.Index.PIPE_LINE])

    def get_classifier_names(self) -> Iterator[str]:
        """Classifier 이름 제너레이터

        Raises:
            BufferError: Classifier Dict 가 비어 있을 경우

        Yields:
            str: Classifier 이름
        """
        if not self._classifier_dict:
            raise BufferError('Classifier list Not Found.',
                           get_code_line(inspect.currentframe()))

        for name in self._classifier_dict:
            if name:
                yield name

    def get_pipelines_with_name(self) -> Iterator[Tuple[str, Any]]:
        """Classifier 이름, pipeline 객체 제너레이터

        Raises:
            BufferError: Model Dict이 비어 있을 경우 예외 발생

        Yields:
            tuple: Classifier 이름, pipeline 객체
        """
        if not self._model_dict:
            raise BufferError('Model Dict Not Found.',
                           get_code_line(inspect.currentframe()))

        for name, model in self._model_dict.items():
            if model[Model.Index.PIPE_LINE]:
                yield (name, model[Model.Index.PIPE_LINE])

    def get_save_model_path_list(self) -> List[str]:
        """모델 파일 경로 리스트 전달

        Raises:
            BufferError: Model Dict 가 비어 있을 경우
            TypeError: 로드한 모델 리스트와 model dict 의 크기가 다를 경우

        Returns:
            list[str]: 모델 파일 경로 리스트
        """
        if not self._model_dict:
            raise BufferError('Model Dict Not Found.',
                              get_code_line(inspect.currentframe()))

        path_list = []
        for model in self._model_dict.values():
            if model[Model.Index.MODEL_PATH]:
                path_list.append(model[Model.Index.MODEL_PATH])

        if len(path_list) != len(self._model_dict):
            raise TypeError('Get Save Model Path List Length Invalid.',
                            get_code_line(inspect.currentframe()))
        return path_list

    def get_local_model_path_with_name(self) -> Iterator[Tuple[str, str]]:
        """Classifier 이름, 모델 파일 경로 제너레이터

        Raises:
            BufferError: Model Dict 가 비어 있을 경우

        Yields:
            Tuple: Classifier 이름, 모델 파일 경로
        """
        if not self._model_dict:
            raise BufferError('Model Dict Not Found.',
                           get_code_line(inspect.currentframe()))

        for name, model in self._model_dict.items():
            if model[Model.Index.MODEL_PATH]:
                yield (name, model[Model.Index.MODEL_PATH])
