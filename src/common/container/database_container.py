from typing import Union
import inspect
import pandas as pd

from common.function import get_code_line
from common.container.base_container import DataContainer
from common.container.mongo import Collection

class DataBaseContainer(DataContainer):
    """Data Gather From Database"""

    @staticmethod
    def verify(data: dict, table: dict = None) -> bool:
        """data 및 table 파라미터 검증

        Args:
            data (dict): 데이터 로드할 파라미터
            table (dict, optional): 컬럼 로드할 파라미터

        Returns:
            bool: True: 로드 가능한 파라미터가 있음
                  False: 해당 파라미터로 로드 불가능
        """
        # database명이 있을 경우는 컬렉션명도 같이 체크.
        if (('database' in data) and ('collection' in data)):
            pass
        else:
            return False

        if table is None:  # 테이블 값이 없으면 x_data의 컬럼으로 대체됨
            pass
        elif (('database' in table) and ('collection' in table)):
            pass
        else:
            return False

        return True

    @classmethod
    def get_data(cls, data: dict) -> Union[pd.DataFrame, pd.Series]:
        """database로부터 데이터 로드

        Args:
            data (dict): 'database', 'collection', 'start_date', 'end_date'

        Returns:
            x_data, y_data: pandas DataFrame, Series
        """
        x_data = None
        y_data = None

        if (('database' not in data) or ('collection' not in data)):
            return x_data, y_data

        # database 정보를 통해 데이터 로드.
        result_list = Collection.get_data_list_from_data_dict(data=data)
        df = pd.DataFrame(result_list)

        x_data = df.drop(['_id', 'label', 'date'], axis=1)
        y_data = df['label']

        return x_data, y_data

    @classmethod
    def get_columns(cls, table: dict = None) -> list:
        """컬럼 정보 로드 함수

        Args:
            table (dict): 'database', 'collection', 'start_date', 'end_date'

        Returns:
            feature_table: 컬럼 리스트

        Description:
            table 이 None 일 경우는 x_data의 컬럼으로 데이터가 만들어짐
        """
        # 데이터가 없으면 x_data의 컬럼으로 대체됨.
        feature_table = None
        if table is None:
            return feature_table

        if (('database' not in table) or ('collection' not in table)):
            return feature_table

        # database 정보를 통해 데이터 로드.
        feature_table = Collection.get_columns_from_table_dict(table=table)

        return feature_table

    @classmethod
    def load_data(cls, data: dict, table: dict = None) -> DataContainer:
        """파라미터를 통해 데이터 로드하여 클래스 초기화

        - parameters
        data 값
        Key: 'database', 'collection', 'start_date', 'end_date'
              database: database 명
              collection: collection 명
              start_date: 쿼리할 시작 날짜 문자열
              end_date: 쿼리할 종료 날짜 문자열

        table 값
        Key: 'database', 'collection', 'start_date', 'end_date'
              database: database 명
              collection: collection 명
              start_date: 쿼리할 시작 날짜 문자열
              end_date: 쿼리할 종료 날짜 문자열
        """
        # 파라미터 내 dict의 필수 key값이 있는지 검증.
        if cls.verify(data, table) is False:
            raise Exception('Data Parameter Verify Failed.',
                            get_code_line(inspect.currentframe()))

        # 사용할 데이터 로드.
        x_data, y_data = (None, None)
        x_data, y_data = cls.get_data(data)
        if (x_data is None) or (y_data is None):
            raise Exception('Initialize Data Failed.',
                            get_code_line(inspect.currentframe()))

        # 컬럼 정보 로드.
        feature_table = cls.get_columns(table)
        if feature_table is None:
            feature_table = x_data.columns

        x_data = pd.DataFrame(x_data, columns=feature_table)
        x_data.fillna(0, inplace=True)  # 결측치 대체.
        y_data = pd.Series(y_data)

        return cls(x_data, y_data)
