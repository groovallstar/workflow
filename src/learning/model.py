import os
import pickle
from typing import Iterator, Tuple, List, Any, Callable
import inspect
import copy

import numpy as np
from sklearn.pipeline import Pipeline

from common.data_type import MetricData, ThresholdMetricData, MetricScoreInfo
from common.data_type import ScoreInfo, BestModelScoreInfo
from common.container.mongo import QueryBuilder
from common.function import timeit, conditional_decorator, get_code_line

from learning.data_loader import DataLoader
from learning.classifier import Classifier

class Model(DataLoader):
    """모델 클래스

    Args:
        DataLoader (Class): 데이터 로드 Class
    """
    # define model dict items
    PIPELINE = 'pipeline'           # pipeline
    MODEL_PATH = 'model_path'       # local model path (mlflow)
    PREDICT = 'predict'             # predict
    PREDICT_PROBA = 'predict_proba' # predict probability

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

        # 모델 클래스 정보
        self._model_dict = {}
        self._classifier = None

    @conditional_decorator(timeit, True)
    def init_model(self, file_name: str) -> None:
        """Classifier 및 모델 초기화. 명시적으로 호출해야 함.

        Args:
            file_name (YAML Filename): Classifier 파일명

        Raises:
            BufferError: classification_file_name이 없을 경우
            BufferError: Model Dictionary가 초기화 되지 않을 경우
        """
        if not file_name:
            raise BufferError('classification_file_name Parameter Empty.',
                              get_code_line(inspect.currentframe()))
        # Classifier 초기화.
        self._classifier = Classifier(file_name=file_name, seed=self._seed)

        # Model Dictionary init.
        # {'CLASSIFIER NAME': {'pipeline': sklearn.Pipeline,
        #                      'model_path': str,
        #                      'predict': np.ndarray,
        #                      'probablity': np.ndarray}, ...}
        for name in self._classifier.generator_classifier_keys():
            self._model_dict[name] = {}
            self._model_dict[name][Model.PIPELINE] = None
            self._model_dict[name][Model.MODEL_PATH] = ''
            self._model_dict[name][Model.PREDICT] = None
            self._model_dict[name][Model.PREDICT_PROBA] = None

        if not self._model_dict:
            raise BufferError('Model Dict Empty.',
                              get_code_line(inspect.currentframe()))

    @conditional_decorator(timeit, False)
    def make_pipeline_object(self, name: str, parameters = None) -> Pipeline:
        """Sklearn Pipeline 객체 생성.

        Args:
            name (str): Classifier 이름
            parameters (dict): Hyper Parameter 목록

        Raises:
            ValueError: Classifier 클래스 객체가 초기화 되지 않을 경우
            BufferError: Sklearn Pipeline 초기화가 실패할 경우

        Returns:
            Sklearn Pipeline : 파이프라인 객체

        Description:
            catboost 는 이미 fit 된 객체에서 다시 set_param 을 할 경우
            "You can't change params of fitted model." 에러 발생.
            이 method 를 여러번 호출 할 경우 해당 에러 발생하게 되어
            객체를 그대로 쓰지 않고 deepcopy 하여 새로운 객체가 복사되도록 추가.
        """
        if not self._classifier:
            raise ValueError(
                "Classifier Object Not Initialize. Call 'init_model'",
                get_code_line(inspect.currentframe()))
        # sklearn pipline 객체 생성
        pipeline_object = Pipeline([(
            Classifier.PREFIX_NAME,
            copy.deepcopy(
                self._classifier.get_classifier_object(name=name)))])
        if not pipeline_object:
            raise BufferError('sklearn Pipeline Object Not Initialized.',
                              get_code_line(inspect.currentframe()))
        if parameters:
            pipeline_object.set_params(**parameters)
        return pipeline_object

    @conditional_decorator(timeit, False)
    def train(self, name: str = '') -> None:
        """학습.

        Args:
            name (str, optional): str = 지정된 Classifier 이름으로 학습.\
                                        없을 경우 전체 Classifier 학습.
        """
        if name:
            return self.train_by_name(name=name)

        for name in self._model_dict:
            self.train_by_name(name=name)

    @conditional_decorator(timeit, True)
    def train_by_name(self, name: str) -> None:
        """Classifier의 이름을 통해 학습

        Args:
            name (str): Classifier 이름

        Raises:
            ValueError: Model Dictionary가 비어있을 경우
                       Model Dictionary에 Classifier 이름이 없을 경우
            ValueError: Classifier 클래스 객체가 초기화 되지 않을 경우
            BufferError: 학습 데이터가 비어 있을 경우
        """
        if (not self._model_dict) or (name not in self._model_dict):
            raise ValueError(f"'{name}' Not Found In Model Dict.",
                             get_code_line(inspect.currentframe()))
        if not self._classifier:
            raise ValueError(
                "Classifier Object Not Initialize. Call 'init_model'",
                get_code_line(inspect.currentframe()))
        if (self._x_train.empty) or (self._y_train.empty):
            raise BufferError('Train Data is Empty.',
                              get_code_line(inspect.currentframe()))

        hyper_parameter = self._classifier.get_hyper_parameter(name=name)
        # sklearn pipline 객체 생성
        pipeline_object = self.make_pipeline_object(
            name=name, parameters=hyper_parameter)
        pipeline_object.fit(self._x_train, self._y_train)

        self._model_dict[name][Model.PIPELINE] = pipeline_object

    @conditional_decorator(timeit, False)
    def tuning(self, name: str = '', update_file: str = '') -> dict | None:
        """nested cross-validation을 통한 파라미터 튜닝 및 학습.

        Args:
            name (str, optional): str = 지정된 Classifier 이름으로 튜닝 및 학습.
                                        없을 경우 전체 Classifier 학습.
            update_file (str, optional): 파라미터 관리 yaml 파일

        Returns:
            dict | None : Searching Best Parameters (name이 있을 경우, dict)
        """
        if name:
            return self.tuning_by_name(name=name, update_file=update_file)

        for name in self._model_dict:
            self.tuning_by_name(name=name, update_file=update_file)

    @conditional_decorator(timeit, True)
    def tuning_by_name(self, name: str, update_file: str = '') -> dict:
        """Classifier의 이름으로 파라미터 튜닝 및 학습.

        Args:
            name (str): Classifier 이름
            update_file (str, optional): 파라미터 관리 yaml 파일

        Raises:
            ValueError: Classifier 클래스 객체가 초기화 되지 않을 경우

        Returns:
            dict : Searching Best Parameters
        """
        if not self._classifier:
            raise ValueError(
                "Classifier Object Not Initialize. Call 'init_model'",
                get_code_line(inspect.currentframe()))

        parameters = self.get_hyper_parameters(name)
        if not parameters:
            raise ValueError('Get Hyper Parameter Value Empty.')

        # early stopping 파라미터는 pipeline에서 세팅될 경우
        # 이후 fit할 때 validation set가 없으면 학습이 되지 않아
        # 실제 학습할 때는 해당 파라미터를 제거해야함.
        if 'clf__early_stopping_rounds' in parameters:
            parameters.pop('clf__early_stopping_rounds', None)

        # 찾은 파라미터로 하이퍼 파라미터 갱신.
        self._classifier.update_hyper_parameters(name=name, params=parameters)
        # 파일명이 있을 경우 yaml 파일 갱신.
        if update_file:
            self._classifier.save_file(file_name=update_file)
        return parameters

    @conditional_decorator(timeit, True)
    def get_hyper_parameters(
        self, name: str,
        objective_metric: str = 'f1-score',
        objective_mode: str = 'max') -> dict:
        """Ray Tune + Optuna로 하이퍼 파라미터 찾기

        Args:
            name (str): Classifier 명
            objective_metric (str): objective function 내 비교할 메트릭
            objective_mode (str): 목표 최적화를 위한 메트릭 값의 모드(최소/최대)

        Raises:
            ValueError: Model Dict 가 비어 있을 경우
            ValueError: Classifier 클래스 객체가 초기화 되지 않을 경우
            BufferError: 학습 데이터가 없을 경우
        """
        if (not self._model_dict) or (name not in self._model_dict):
            raise ValueError(f"'{name}' Not Found In Model Dict.")
        if not self._classifier:
            raise ValueError(
                "Classifier Object Not Initialize. Call 'init_model'",
                get_code_line(inspect.currentframe()))
        if ((self._x_train.empty) or (self._y_train.empty) or
            (self._x_test.empty) or (self._y_test.empty)):
            raise BufferError('Tuning data must set Train/Test data.',
                              get_code_line(inspect.currentframe()))

        from ray import air
        from ray import tune
        from ray.tune.search import ConcurrencyLimiter
        from ray.tune.schedulers import AsyncHyperBandScheduler
        from ray.tune.search.optuna import OptunaSearch
        from optuna import distributions
        from optuna.samplers import TPESampler
        import tempfile

        # Optuna Space 지정.
        search_space = {}
        hyper_params = self._classifier.get_search_parameter(name=name)
        for k, _ in hyper_params.items():
            low = hyper_params[k][0]
            high = hyper_params[k][1]
            # float type or int type.
            if ((isinstance(low, float) is True) or
                (isinstance(high, float) is True)):
                search_space[k] = distributions.FloatDistribution(
                    low=low, high=high, log=True)
            else:
                search_space[k] = distributions.IntDistribution(
                    low=low, high=high, log=True)

        search_algorithm = OptunaSearch(
            space=search_space,
            metric=objective_metric,
            mode=objective_mode,
            sampler=TPESampler())

        # A wrapper algorithm for limiting the number of concurrent trials.
        algorithm = ConcurrencyLimiter(search_algorithm, max_concurrent=10)

        # Trial Scheduler. Async Successive Halving.
        scheduler = AsyncHyperBandScheduler(
            metric=objective_metric,
            mode=objective_mode,
            grace_period=3,
            reduction_factor=2)

        # 변경하지 않을 기본 파라미터.
        # 주로 binary classification에 사용되는 파라미터들.
        base_param = self._classifier.get_base_parameter(name=name)

        # 파이프라인 객체를 미리 만들고 objective function 내부에서
        # base parameter를 세팅함.
        pipeline = self.make_pipeline_object(name=name)

        from learning.tune import objective_function
        tuner = tune.Tuner(
            tune.with_parameters(
                objective_function,
                name=name,
                pipeline=pipeline,
                x_data=self._x_train,
                y_data=self._y_train,
                x_test=self._x_test,
                y_test=self._y_test,
                base_param=base_param,
                objective_metric=objective_metric),
            tune_config=tune.TuneConfig(
                search_alg=algorithm,
                scheduler=scheduler,
                num_samples=10),
            run_config=air.RunConfig(
                verbose=0,
                storage_path=os.path.join(
                    tempfile.gettempdir(), 'ray_results')))

        results = tuner.fit()
        best_params = results.get_best_result(
            metric=objective_metric, mode=objective_mode).config
        return best_params

    @conditional_decorator(timeit, True)
    def evaluate(self, name: str = '') -> None:
        """평가.
        model dict 에 pipeline 객체가 있으면 예측 값과 예측 확률 값을
        각 model dict 에 추가함.

        Args:
            find_type (str, optional): 지정된 Classifier 이름으로 평가.\
                                       없을 경우 전체 Classifier 평가.
        """
        if name:
            return self.evaluate_by_classifier_name(name=name)

        for name in self._model_dict:
            self.evaluate_by_classifier_name(name)

    @conditional_decorator(timeit, True)
    def evaluate_by_classifier_name(self, name: str) -> None:
        """Classifier 이름으로 평가.

        Args:
            name (str): Classifier 이름.

        Raises:
            NameError: dict에서 Classifier 명이 없을 경우
            BufferError: 테스트 데이터가 없을 경우
            BufferError: 파이프라인 객체 초기화 실패할 경우
        """
        if (not self._model_dict) or (name not in self._model_dict):
            raise NameError(f"'{name}' Not Found In Model Dict.",
                            get_code_line(inspect.currentframe()))

        if self._x_test.empty:
            raise BufferError(f"'{name}' x_test empty.",
                              get_code_line(inspect.currentframe()))

        pipeline = self._model_dict[name][Model.PIPELINE]
        if not pipeline:
            raise BufferError(f"'{name}' pipeline Empty.",
                              get_code_line(inspect.currentframe()))

        self._model_dict[name][Model.PREDICT] = \
            pipeline.predict(self._x_test)
        self._model_dict[name][Model.PREDICT_PROBA] = \
            pipeline.predict_proba(self._x_test)

    @conditional_decorator(timeit, True)
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
        if (not self._model_dict) or (name not in self._model_dict):
            raise NameError(f"'{name}' Not Found In Model Dict.",
                            get_code_line(inspect.currentframe()))

        if self._x_validation.empty:
            raise BufferError(f"'{name}' x_validation empty.",
                              get_code_line(inspect.currentframe()))

        pipeline_object = self._model_dict[name][Model.PIPELINE]
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

    @conditional_decorator(timeit, True)
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
        if not self._model_dict:
            raise ValueError('Model Dict Empty.',
                             get_code_line(inspect.currentframe()))

        load_model_list = []
        if ('file_list' in model) and (model['file_list']):
            load_model_list = model['file_list']
        else:
            load_model_list = QueryBuilder.get_model_list(
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

            self._model_dict[name][Model.MODEL_PATH] = file
            self._model_dict[name][Model.PIPELINE] = pipeline_object

    @conditional_decorator(timeit, True)
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
        if not os.path.exists(model_file):
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

        self._model_dict[name][Model.MODEL_PATH] = model_file

    @conditional_decorator(timeit, True)
    def save_model_information_to_database(
        self, model: dict, data: dict, table: dict,
        save_path_list: list) -> None:
        """학습한 모델 정보를 Database에 저장.

        Args:
            model (dict): 'database', 'collection'
            data (dict): 'database', 'collection',\
                               'start_date', 'end_date'
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
            if not os.path.exists(file):
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
        QueryBuilder.insert_model_information(
            train=data,
            table=table,
            model=model,
            model_save_path_list=save_path_list)

    @conditional_decorator(timeit, False)
    def get_evaluate_metrics(
        self, name:str, thresholds:list) -> Iterator[ThresholdMetricData]:
        """평가 지표 출력 함수

        Args:
            find_type (str): str=Classifier 명으로 평가 지표 출력.
            thresholds (list): threshold list값을 이용한 평가 지표 출력.\
                              e.g. [0.3, 0.4, 0.6, 0.7]

        Raises:
            BufferError: classifier or model dict 가 비어 있을 경우
            TypeError: 파라미터가 잘못될 경우
            NameError: dictionary 에 key 가 없을 경우

        Yields:
             Iterator[ThresholdMetricData] : threshold, MetricData
        """
        if ((isinstance(name, str) is False) or
            (isinstance(thresholds, list) is False)):
            raise TypeError('name or thresholds Wrong Type.')

        if (not self._model_dict) or (name not in self._model_dict):
            raise NameError(f"'{name}' Not Found In Model Dict.")

        from learning.evaluation import get_metrics_by_threshold
        metric_with_threshold = get_metrics_by_threshold(
            y_test=self._y_test,
            predict=self._model_dict[name][Model.PREDICT],
            probability=self._model_dict[name][Model.PREDICT_PROBA],
            thresholds=thresholds)
        return metric_with_threshold

    @conditional_decorator(timeit, False)
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

        # 모델에서 지정한 Score가 가장 높은 threshold와 metric을 구함.
        for metric_name in metric_name_list:
            metric_function = None
            if metric_name == MetricData.str_f1():
                metric_function = MetricData.get_f1_score

            if not metric_function:
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

    @conditional_decorator(timeit, True)
    def get_highest_score_for_model(self) -> BestModelScoreInfo:
        """평가지표 중 가장 높은 score 를 가진 모델과 임계값을 찾는 함수.
           현재는 F1-Score, ROC-AUC 가 가장 높은 모델과 임계값 출력.

        Returns:
            BestModelScoreInfo: 각 모델의 f1-score 최고 메트릭 및 임계치,
                                모델 중 가장 높은 f1-score 모델 메트릭 정보,
        """
        if not self._model_dict:
            raise ValueError('Model Dict Empty.',
                             get_code_line(inspect.currentframe()))

        def get_highest_score_in_model_list(
            metric_name: str, each_best_score_list: list) -> ScoreInfo:
            """각 모델에서 평가했던 metric 중 score list 에서
               가장 높은 score 의 임계값을 통해 전체 metric 출력.

            Args:
                metric_name (str): 메트릭 이름
                each_best_score_list (list): 각각 가장 높은 점수가 기록된 리스트

            Returns:
                ScoreInfo: (Model name, threshold, MetricData)
            """
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
                predict=self._model_dict[name][Model.PREDICT],
                probability=self._model_dict[name][Model.PREDICT_PROBA],
                threshold=best_threshold)
            score_info = ScoreInfo(
                name=name, threshold=best_threshold, metric=best_metric)
            return score_info

        # 각각의 모델에서 score가 높은 threshold와 metric을 구함.
        each_best_f1_score = self.get_best_score_for_model(
            MetricData.get_f1_score)

        # 모델 중에서도 높은 metric을 가진 모델 하나를 구함.
        highest_f1 = get_highest_score_in_model_list(
            MetricData.str_f1(), each_best_f1_score)

        best_model_info = BestModelScoreInfo(
            highest_f1=highest_f1)
        return best_model_info

    @conditional_decorator(timeit, False)
    def get_best_score_for_model(
        self, metric_function:Callable,
        find_type: str | int | None = None) -> List[ScoreInfo]:
        """각 Classifier 별 가장 높은 점수 리스트 전달

        Args:
            metric_function (Callable Method): 점수를 평가할 메트릭 함수 포인터
            find_type (str | int, optional):
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
                predict=model[Model.PREDICT],
                probability=model[Model.PREDICT_PROBA])
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
            Tuple : (Classifier 이름, sklearn.Pipeline 객체)
        """
        if not self._model_dict:
            raise BufferError('Model Dict Not Found.',
                           get_code_line(inspect.currentframe()))
        return (name, self._model_dict[name][Model.PIPELINE])

    def get_classifier_names(self) -> Iterator[str]:
        """Classifier 이름 제너레이터

        Yields:
            str: Classifier 이름
        """
        for name in self._model_dict:
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

        for name, model_info in self._model_dict.items():
            if Model.PIPELINE in model_info:
                yield (name, model_info[Model.PIPELINE])

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
        for model_info in self._model_dict.values():
            if Model.MODEL_PATH in model_info:
                path_list.append(model_info[Model.MODEL_PATH])

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

        for name, model_info in self._model_dict.items():
            if Model.MODEL_PATH in model_info:
                yield (name, model_info[Model.MODEL_PATH])
