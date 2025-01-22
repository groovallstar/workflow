import inspect
import pandas as pd

from common.function import get_code_line

class DataContainer:
    """Base Container"""
    def __init__(self, x_data, y_data):
        """initialize"""
        self._x_data = x_data
        self._y_data = y_data

        import re
        import pandas as pd
        if isinstance(x_data, pd.DataFrame) is True:
            # int64 -> int32. Reduce Memory.
            self._x_data = x_data.astype('int32')
            self._y_data = y_data.astype('int32')

            # LightGBM categorical feature 관련 에러
            # (Do not support special JSON characters in feature name.)
            # 컬럼명에 특정 특수문자 포함될 경우 회피하기 위해 제거.
            df = self._x_data.rename(
                columns=lambda x: re.sub('[^A-Za-z0-9_/]+', '', x))

            # 각 컬럼의 특수문자가 제거된 이후 중복 컬럼이 생길 경우 처리.
            # 'test!!!~!_!@@@', 'test!!!_!' -> 'test_' 로 치환되면서 중복됨.
            # 중복된 컬럼의 row 들의 합을 구하기 위해서는 지정된 index 없이
            # level을 0 지정. (groupby를 통해 중복 컬럼이 병합되고
            # 행에 따라 열 방향으로 sum이 적용됨)
            self._x_data = df.T.groupby(level=0).sum().T

    @property
    def x_data(self):
        return self._x_data

    @x_data.setter
    def x_data(self, data):
        self._x_data = data

    @property
    def y_data(self):
        return self._y_data

    @y_data.setter
    def y_data(self, data):
        self._y_data = data

    def split(
        self, train_size: float = 0.75, validation_size: float = None,
        test_size: float = None, **options) -> tuple:
        """
        train_test_split.

        - parameters
        train_size: split train data size
        validation_size: split validation data size
        test_size: split test data size
        options: sklearn train_test_split parameter

        - return value
        train_data_list, validation_data_list, test_data_list

        - description
        validation_size 사용 할 때: train_size + validation_size
                                    + test_size = 1.0 이 되어야 함
        validation_size 사용 안할 때: train_size + test_size = 1.0 이 되어야 함
        """
        import math
        # 총 비율 체크.
        if validation_size is None:
            if math.isclose(a=(train_size + test_size), b=1.0) is False:
                raise Exception('Total proportion must be 1.0.',
                                get_code_line(inspect.currentframe()))
        else:
            if math.isclose(
                a=(train_size + validation_size + test_size), b=1.0) is False:
                raise Exception('Total proportion must be 1.0.',
                                get_code_line(inspect.currentframe()))

        from sklearn.model_selection import train_test_split

        if validation_size is None:
            x_train, x_test, y_train, y_test = train_test_split(
                self._x_data, self._y_data,
                train_size=train_size,
                test_size=test_size, stratify=self._y_data, **options)

            x_train.reset_index(drop=True, inplace=True)
            x_test.reset_index(drop=True, inplace=True)

            y_train.reset_index(drop=True, inplace=True)
            y_test.reset_index(drop=True, inplace=True)
            return (x_train, x_test, y_train, y_test)

        else:
            x_train, x_split, y_train, y_split = train_test_split(
                self._x_data, self._y_data, train_size=train_size, **options)
            # train : validation : test 비율 계산
            proportion = validation_size / (1.0 - train_size)
            x_validation, x_test, y_validation, y_test = train_test_split(
                x_split, y_split, train_size=proportion, stratify=y_split,
                **options)

            x_train.reset_index(drop=True, inplace=True)
            x_validation.reset_index(drop=True, inplace=True)
            x_test.reset_index(drop=True, inplace=True)

            y_train.reset_index(drop=True, inplace=True)
            y_validation.reset_index(drop=True, inplace=True)
            y_test.reset_index(drop=True, inplace=True)
            return (x_train, x_validation, x_test,
                    y_train, y_validation, y_test)

    def sampling(
        self, frac: float, random_state: int=None) -> tuple:
        """Data Sampling. (y_data 비율에 따라 균등 추출)

        - parameters
        frac: 추출 비율

        Args:
            frac (float): 추출 비율
            random_state (int, optional): 추출 랜덤 시드 값

        Raises:
            ValueError: y_data가 없을 경우
            ValueError: 추출 비율이 1을 초과할 경우
            ValueError: x_data, y_data 사이즈가 0인 경우
            ValueError: y_data 비율대로 분할이 잘못된 경우

        Returns:
            tuple: x_data, y_data
        """
        x_data, y_data = (pd.DataFrame(), pd.Series(dtype='uint32'))

        # y_data를 이용해 StratifiedKFold 해야하므로 empty 체크.
        if self._y_data.empty:
            raise ValueError('y_data empty.')

        if frac > 1.0:
            raise ValueError('Frac ratio is overflowed.')

        if (self._x_data is not None) and (self._x_data.size):
            x_data = self._x_data.sample(frac=frac)
        if (self._y_data is not None) and (self._y_data.size):
            y_data = self._y_data.sample(frac=frac)

        if (not x_data.size) or (not y_data.size):
            raise ValueError('x_data or y_data size empty.')

        from sklearn.model_selection import StratifiedKFold
        fold = StratifiedKFold(n_splits=10, shuffle=True,
                               random_state=random_state)
        train_index = None
        for train_index, _ in fold.split(x_data, y_data):
            break

        if not train_index.size:
            raise ValueError('Train Data fold split empty.')

        if (x_data is not None) and (x_data.size):
            x_data = x_data.iloc[train_index]
        if (y_data is not None) and (y_data.size):
            y_data = y_data.iloc[train_index]

        return (x_data, y_data)

    def to_csv(self, csv_file_path: str) -> None:
        """pandas DataFrame -> csv file로 저장

        Args:
            csv_file_path (str): 저장할 csv 파일명

        Raises:
            ValueError: 파라미터가 잘못될 경우
            FileExistsError: 파일이 이미 있을 경우
            ValueError: x_data or y_data 가 없을 경우
        """
        try:
            import os
            if not csv_file_path:
                raise ValueError('csv_file_path size 0.')

            if os.path.isfile(csv_file_path):
                raise FileExistsError(f'{csv_file_path} already exist.')

            if (self._x_data is None) or (self._y_data is None):
                raise ValueError('x_data or y_data is None.')

            df = self._x_data.copy()
            df['label'] = self._y_data

            df.to_csv(csv_file_path, index=False)

        except Exception as ex:
            raise Exception(ex)
