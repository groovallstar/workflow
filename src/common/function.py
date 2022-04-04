"""function.py"""

def singleton(class_):
    """singleton pattern."""
    class class_w(class_):
        _instance = None

        def __new__(class_, *args, **kwargs):
            if class_w._instance is None:
                class_w._instance = super(class_w, class_).__new__(class_)
                class_w._instance._sealed = False
            return class_w._instance

        def __init__(self, *args, **kwargs):
            if self._sealed:
                return
            super(class_w, self).__init__(*args, **kwargs)
            self._sealed = True
    class_w.__name__ = class_.__name__
    return class_w

def timeit(method):
    """함수 종료 시간 측정"""
    import functools
    @functools.wraps(method)
    def timed(*args, **kwargs):
        import time
        start_time = time.time()
        result = method(*args, **kwargs)
        end_time = time.time()

        from common.trace_log import TraceLog
        TraceLog().info(f"'{method.__name__}' "
                        f"Time : {(end_time - start_time):.4f} seconds")
        return result

    return timed

def get_code_line(frame) -> tuple:
    """해당 frame의 파일명, 코드라인 얻기

    Args:
        frame (inspect.currentframe): inspect 객체의 currentframe() 호출 결과

    Returns:
        string: 코드 위치, 코드 라인
    """
    # __FILE__
    file_name = frame.f_code.co_filename
    # __LINE__
    file_line = frame.f_lineno

    return file_name, file_line

from enum import Enum
class StrEnum(str, Enum):
    """string enum class"""
    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def __next__(self):
        return self.name

class TempDir:
    """임시폴더 클래스"""
    def __init__(
        self, prefix: str=None, chdr: bool=False, remove_on_exit: bool=True):
        """초기화.

        Args:
            prefix (str, optional): 임시폴더 prefix.
            chdr (bool, optional): SetCurrentDirectory 지정.
            remove_on_exit (bool, optional): with scope 빠져나올 때 폴더 삭제.
        """
        self._dir = None
        self._path = None
        self._prefix = prefix
        self._chdr = chdr
        self._remove = remove_on_exit

    def __enter__(self):
        """with scope 진입 시"""
        import tempfile
        import os
        self._path = os.path.abspath(tempfile.mkdtemp(prefix=self._prefix))
        assert os.path.exists(self._path)
        if self._chdr:
            self._dir = os.path.abspath(os.getcwd())
            os.chdir(self._path)
        return self

    def __exit__(self, tp, val, traceback):
        """with scope 빠져나올 때"""
        import os
        if self._chdr and self._dir:
            os.chdir(self._dir)
            self._dir = None
        if self._remove and os.path.exists(self._path):
            import shutil
            shutil.rmtree(self._path)

        assert not self._remove or not os.path.exists(self._path)
        assert os.path.exists(os.getcwd())

    def path(self, *path):
        """경로"""
        import os
        return os.path.join("./", *path)\
            if self._chdr else os.path.join(self._path, *path)

def apply_standard_scaler_to_train_data(
    x_train, x_validation, x_test, save_scaler_path: str = None) -> tuple:
    """학습/검증/테스트 데이터에 StandardScaler 적용 후 객체 저장.
        학습 할때만 사용 할 것.
    """
    def apply_scaler_to_fit_data(
        scaler, x_train, x_validation=None, x_test=None,
        log_function=True) -> tuple:
        """
        학습 데이터 set 에 Scaler 적용

        - parameters
        scaler : 데이터에 적용할 scaler object
        x_train, x_validation, x_test : log 함수 및 scaler 적용할 DataFrame
        log_function (bool) : True=numpy.log1p 적용, False=미적용

        - description
        검증/테스트 데이터에는 transform() API만 적용 해야함.
        학습 데이터에 해당 scaler를 적용했으면, 예측 때 동일한
        scaler를 이용해 transfrom 해야함.
        """
        import pandas as pd
        import numpy as np

        if (scaler is None) or (x_train is None):
            return

        # log1p 함수 적용.
        if log_function:
            log1p_train = pd.DataFrame(None, columns=x_train.columns)

            if x_validation is not None:
                log1p_validation = pd.DataFrame(None, columns=x_train.columns)
            else:
                log1p_validation = None

            if x_test is not None:
                log1p_test = pd.DataFrame(None, columns=x_train.columns)
            else:
                log1p_test = None

            for i in x_train.columns:
                log1p_train[i] = x_train.loc[:, i].map(
                    lambda x: np.log1p(x) if x > 0 else 0)
                if x_validation is not None:
                    log1p_validation[i] = x_validation.loc[:, i].map(
                        lambda x: np.log1p(x) if x > 0 else 0)
                if x_test is not None:
                    log1p_test[i] = x_test.loc[:, i].map(
                        lambda x: np.log1p(x) if x > 0 else 0)

            x_train = log1p_train.copy()
            if x_validation is not None:
                x_validation = log1p_validation.copy()
            if x_test is not None:
                x_test = log1p_test.copy()

        # scaler 적용.
        scaler.fit(x_train)
        x_train = pd.DataFrame(
            scaler.transform(x_train), columns=x_train.columns)
        if x_validation is not None:
            x_validation = pd.DataFrame(scaler.transform(
                x_validation), columns=x_train.columns)
        if x_test is not None:
            x_test = pd.DataFrame(scaler.transform(x_test),
                                columns=x_train.columns)

        return (x_train, x_validation, x_test)

    import os
    import pickle
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    apply_data = apply_scaler_to_fit_data(
        scaler,
        x_train,
        x_validation if x_validation.empty is False else None,
        x_test if x_test.empty is False else None)

    if os.path.isdir(save_scaler_path) is False:
        os.mkdir(save_scaler_path)
    scaler_file_name = os.path.join(save_scaler_path, 'Scaler.pickle')
    with open(scaler_file_name, "wb") as f:
        pickle.dump(scaler, f)

    return apply_data

def apply_standard_scaler_to_test_data(x_test, scaler_file_path: str):
    """테스트 데이터에 StandardScaler 적용 함수. 예측할 때만 사용 할 것."""
    import os
    import pickle
    if os.path.exists(scaler_file_path) is False:
        return

    scaler = None
    with open(scaler_file_path, "rb") as f:
        scaler = pickle.load(f)

    def apply_scaler_to_predict_data(scaler, x_data, log_function=True):
        """
        예측 데이터 set 에 Scaler 적용

        - parameters
        scaler : 데이터에 적용할 scaler object
        x_data : log 함수 및 scaler 적용할 DataFrame
        log_function (bool) : True=numpy.log1p 적용, False=미적용

        - description
        학습에 사용했던 scaler object를 사용해야함
        예측 데이터에는 fit() API는 호출하지 않고, transform() API 만 호출함
        """
        import pandas as pd
        import numpy as np

        if (scaler is None) or (x_data is None):
            return

        # log1p 함수 적용.
        if log_function:
            # pandas DataFrame 일 경우
            if isinstance(x_data, pd.DataFrame):
                log1p_data = pd.DataFrame(
                    None,
                    columns=x_data.columns if x_data.columns is not None else None)
                for i in x_data.columns:
                    log1p_data[i] = x_data.loc[:, i].map(
                        lambda x: np.log1p(x) if x > 0 else 0)
                x_data = log1p_data.copy()
            else:
                # list 일 경우
                for i in range(len(x_data)):
                    if x_data[i] > 0:
                        x_data[i] = np.log1p(x_data[i])
                    else:
                        x_data[i] = 0

        # scaler 적용.
        if isinstance(x_data, pd.DataFrame):
            x_data = pd.DataFrame(
                scaler.transform(x_data), 
                columns=x_data.columns if x_data.columns is not None else None)
        else:
            # 리스트 형태는 2차원으로 변경해야함
            x_data = np.reshape(x_data, (-1, 1))  
            x_data = scaler.transform(x_data)

        return x_data

    return apply_scaler_to_predict_data(scaler, x_test)

def find_bayesian_optimizer(
    pipeline_object, search_parameters:dict, x_train, y_train) -> dict:
    """Bayesian Optimizer 방식으로 하이퍼 파라미터 찾기 (Deprecated)

    Args:
        pipeline_object (sklearn.pipeline): pipeline 객체
        search_parameters : bayesian parameter
        x_train : 학습할 x 데이터
        y_train : 학습할 y 데이터

    Raises:
        BufferError: Classifier or Model Dict 가 비어 있을 경우
        NameError: dictionary 에 key 가 없을 경우
        BufferError: 학습 데이터가 없을 경우
        BufferError: 베이지안 최적화 파라미터가 없을 경우

    Returns:
        Dict : 하이퍼 파라미터 정보
    """
    def evaluate_procedure(**bayesian_params) -> float:
        """최적화 계산할 함수
            파라미터로 전달되는 값은 float로 변환되서 전달되므로
            sklearn에서 사용하기 위해서는 int 로 변환해야함
            (float 형 파라미터도 있으므로 무조건 변환하면 안됨)
            변환 e.g.
            'max_depth': int(round(max_depth))
            'subsample': max(min(subsample, 1), 0)
            'reg_lambda': max(reg_lambda, 0)

        Returns:
            mean: f1-score 평균값
        """

        if 'clf__n_estimators' in bayesian_params:
            bayesian_params['clf__n_estimators']=\
                int(round(bayesian_params['clf__n_estimators']))

        if 'clf__iterations' in bayesian_params:
            bayesian_params['clf__iterations']=\
                int(round(bayesian_params['clf__iterations']))

        if 'clf__max_depth' in bayesian_params:
            bayesian_params['clf__max_depth']=\
                int(round(bayesian_params['clf__max_depth']))

        if 'clf__num_leaves' in bayesian_params:
            bayesian_params['clf__num_leaves']=\
                int(round(bayesian_params['clf__num_leaves']))

        if 'clf__depth' in bayesian_params:
            bayesian_params['clf__depth']=\
                int(round(bayesian_params['clf__depth']))

        if 'clf__min_data_in_leaf' in bayesian_params:
            bayesian_params['clf__min_data_in_leaf']=\
                int(round(bayesian_params['clf__min_data_in_leaf']))

        pipeline_object.set_params(**bayesian_params)
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True)
        result = cross_val_score(
            estimator=pipeline_object, 
            X=x_train, 
            y=y_train, 
            cv=stratified_kfold, 
            scoring='f1')
        return result.mean()

    from bayes_opt import BayesianOptimization
    bayesian_opt = BayesianOptimization(
        f=evaluate_procedure,
        pbounds=search_parameters,
        random_state=None,
        verbose=0)

    bayesian_opt.maximize(init_points=0, n_iter=10)
    return bayesian_opt.max
