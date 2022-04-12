from typing import Type, Any

import pandas as pd

from common.function import timeit
from common.trace_log import TraceLog

class DataLoader:
    """데이터 로드 클래스"""
    def __init__(self,
                 x_train: pd.DataFrame, y_train: pd.Series,
                 x_validation: pd.DataFrame, y_validation: pd.Series,
                 x_test: pd.DataFrame, y_test: pd.Series):
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
        self._x_train = x_train
        self._y_train = y_train
        self._x_validation = x_validation
        self._y_validation = y_validation
        self._x_test = x_test
        self._y_test = y_test

    @classmethod
    @timeit
    def prepare_data(cls,
                     data: dict, table: dict, split_ratio: dict = None,
                     sampling: float = None, seed: int = None) -> Type[Any]:
        """데이터 로드 및 가공 작업
        data 값
        Format: {'database': str, 'collection': str,
                  'start_date': str, 'end_date': str}

        table 값
        Format: {'database': str, 'collection': str,
                  'start_date': str, 'end_date': str}

        split_ratio 값
        Format: {'train': float, 'validation': float, 'test': float }
        학습/검증/예측 비율 총 합이 1.0 이어야함.
        split_ratio값이 None일 경우 전체 데이터가 학습 데이터로 사용.
        validation 값이 None이거나 0일 경우 학습/예측 데이터로 분리.
        test 값이 1.0일 경우 전체 데이터가 예측 데이터로 사용.

        Args:
            data (dict): 데이터 dictionary
            table (dict): 테이블 dictionary
            split_ratio (dict, optional): 데이터 분할 할 비율 Defaults to None.
            sampling (float, optional): 전체 데이터에서 축소할 비율(0 ~ 1.0 사이)
                                        Defaults to None.
            seed (int, optional): RandomState값. Defaults to None.
        """
        x_train, x_validation, x_test = (pd.DataFrame(),
                                         pd.DataFrame(),
                                         pd.DataFrame())
        y_train, y_validation, y_test = (pd.Series(dtype='uint32'),
                                         pd.Series(dtype='uint32'),
                                         pd.Series(dtype='uint32'))

        # load data
        from common.container.database_container import DataBaseContainer
        container_data = DataBaseContainer.load_data(data=data, table=table)

        # data sampling.
        if sampling:
            (x_data, y_data) = container_data.sampling(frac=sampling)

            # vector data instance variable re-assign.
            container_data.x_data = x_data
            container_data.y_data = y_data

        # data split
        if split_ratio:
            train_ratio, validation_ratio, test_ratio = 0.0, 0.0, 0.0
            if (('train' in split_ratio) and (split_ratio['train'])):
                train_ratio = split_ratio['train']
            if (('validation' in split_ratio) and (split_ratio['validation'])):
                validation_ratio = split_ratio['validation']
            if (('test' in split_ratio) and (split_ratio['test'])):
                test_ratio = split_ratio['test']

            import math
            if math.isclose(a=train_ratio, b=1.0):
                x_train = container_data.x_data
                y_train = container_data.y_data
            elif math.isclose(a=test_ratio, b=1.0):
                x_test = container_data.x_data
                y_test = container_data.y_data
            elif ((validation_ratio is None) or
                  (math.isclose(a=validation_ratio, b=0.0))):
                (x_train, x_test, y_train, y_test) = container_data.split(
                    train_size=train_ratio, test_size=test_ratio,
                    random_state=seed)
            else:
                (x_train, x_validation, x_test,
                 y_train, y_validation, y_test) = container_data.split(
                    train_size=train_ratio,
                    validation_size=validation_ratio,
                    test_size=test_ratio,
                    random_state=seed)
        else:
            x_train = container_data.x_data
            y_train = container_data.y_data

        return cls(
            x_train, y_train,
            x_validation, y_validation,
            x_test, y_test,
            seed)

    def show_data(self) -> None:
        """show data"""
        if self._y_train.empty is False:
            TraceLog().info(f"Train Data Shape: x_train={self._x_train.shape}, "
                            f"y_train={self._y_train.shape}")
            message = "Train Data Ratio: "
            for index in range(self._y_train.value_counts().size):
                message += "Class_" + str(index)
                ratio = self._y_train.value_counts(normalize=True)[index]
                message += f"=({ratio:.4f}) "
            TraceLog().info(message)

        if self._y_validation.empty is False:
            TraceLog().info(f"Validation Data Shape: "
                            f"x_validation={self._x_validation.shape}, "
                            f"y_validation={self._y_validation.shape}")
            message = "Validation Data Ratio: "
            for index in range(self._y_validation.value_counts().size):
                message += "Class_" + str(index)
                ratio = self._y_validation.value_counts(normalize=True)[index]
                message += f"=({ratio:.4f}) "
            TraceLog().info(message)

        if self._y_test.empty is False:
            TraceLog().info(f"Test Data Shape: x_test={self._x_test.shape}, "
                            f"y_test={self._y_test.shape}")
            message = "Test Data Ratio: "
            for index in range(self._y_test.value_counts().size):
                message += "Class_" + str(index)
                ratio = self._y_test.value_counts(normalize=True)[index]
                message += f"=({ratio:.4f}) "
            TraceLog().info(message)

    @property
    def x_train(self):
        """get x_train data"""
        return self._x_train

    @x_train.setter
    def x_train(self, data):
        self._x_train = data

    @property
    def y_train(self):
        """get y_train data"""
        return self._y_train

    @y_train.setter
    def y_train(self, data):
        self._y_train = data

    @property
    def x_validation(self):
        """get x_validation data"""
        return self._x_validation

    @x_validation.setter
    def x_validation(self, data):
        self._x_validation = data

    @property
    def y_validation(self):
        """get y_validation data"""
        return self._y_validation

    @y_validation.setter
    def y_validation(self, data):
        self._y_validation = data

    @property
    def x_test(self):
        """get x_test data"""
        return self._x_test

    @x_test.setter
    def x_test(self, data):
        self._x_test = data

    @property
    def y_test(self):
        """get y_test data"""
        return self._y_test

    @y_test.setter
    def y_test(self, data):
        self._y_test = data
