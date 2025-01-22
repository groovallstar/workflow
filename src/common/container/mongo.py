import os
from datetime import datetime
from typing import Any
import inspect

import pymongo

from common.function import get_code_line, singleton

def _check_key_in_dict(value: dict) -> bool:
    """dictionary에 database, collection key 가 있는지 체크"""
    if not isinstance(value, dict):
        return False
    if ('database' in value) and ('collection' in value):
        return True
    else:
        return False

@singleton
class MongoDBConnection:
    """pymongo connection object singleton."""
    _connection = None # 스크립트 내 한번만 생성함.

    @classmethod
    def initialize(cls, url: str = ''):
        """init."""
        if url:
            mongodb_url = url
        else:
            mongodb_url = os.environ.get('MONGODB_URL')
        if not mongodb_url:
            raise ValueError("'MONGODB_URL' environment variable not found.")
        if cls._connection is None:
            cls._connection = pymongo.MongoClient(
                host=mongodb_url, directConnection=True)
        return cls._connection

    @classmethod
    def close(cls):
        """pymongo client close
        
        Description:
            pymongo로 close 통지를 여러번 호출하면 디버거의 break point 및 \
            call stack 위치에 따라 "ReferenceError: weakly-referenced \
            object no longer exists" 에러가 발생할 수 있음.
        """
        if cls._connection is not None:
            cls._connection.close()

class MongoDB:
    """MongoDB Class"""

    def __init__(self, database_name: str):
        """Database 연결.

        Args:
            database_name (Database 명): Database 명
        """
        self._connection = MongoDBConnection().initialize()
        self._database = None
        if database_name:
            self._database = self._connection.get_database(database_name)

    def close(self) -> None:
        """ Database 객체 소멸

        Description:
            pymongo로 close 통지를 여러번 호출하면 디버거의 break point 및 \
            call stack 위치에 따라 "ReferenceError: weakly-referenced \
            object no longer exists" 에러가 발생할 수 있음.
        """
        MongoDBConnection().close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self.close()

    @property
    def database(self):
        """get database object"""
        if self._database is not None:
            return self._database
        else:
            raise ValueError("MongoClient not initialized.")

    def get_database_list(self) -> list:
        """Get Database List."""
        if self._connection is not None:
            return self._connection.list_database_names()
        else:
            raise ValueError("MongoClient not initialized.")

    def get_collection_list(self) -> list:
        """Get collection List.

        Raises:
            Exception: connection object empty.

        Returns:
            list: collection name list.
        """
        if self._database is not None:
            return self._database.list_collection_names()
        else:
            raise ValueError("MongoClient not initialized.")

    @staticmethod
    def convert_datetime(datetime_string: str) -> datetime | None:
        """문자열을 날짜값으로 변환.

        Args:
            datetime_string (str): YYYYMM, YYYYMMDD 형식의 문자열

        Raises:
            BaseException: 날짜 변환이 실패할 경우

        Returns:
            datetime.datetime: 변환된 날짜 데이터
            None: 파라미터가 잘못될 경우, 날짜 변환이 실패할 경우
        """
        try:
            if len(datetime_string) == 6: # YYYYMM format.
                date = datetime.strptime(datetime_string, '%Y%m').date()
            elif len(datetime_string) == 8: # YYYYMMDD format
                date = datetime.strptime(
                    datetime_string, '%Y%m%d').date()
            else:
                return None
        except:
            return None
        return datetime(date.year, date.month, date.day)

    @staticmethod
    def make_datetime_query(start: str = '', end: str = '') -> dict:
        """시작,종료 기간 값을 쿼리하기 위한 쿼리문 생성

        Args:
            start (str, optional): YYYYMMDD 문자열.
            end (str, optional): YYYYMMDD 문자열.

        Returns:
            dict: {'start_date': datetime(), 'end_date': datetime()} 값
        """
        start_date = MongoDB.convert_datetime(datetime_string=start)
        end_date = MongoDB.convert_datetime(datetime_string=end)

        query = {}
        if start_date:
            query['start_date'] = start_date
        if end_date:
            query['end_date'] = end_date

        return query

    @staticmethod
    def make_date_range_query_with_equal(
        start_date: str = '', end_date: str = '') -> dict:
        """데이터 컬렉션의 날짜 데이터를 쿼리하기 위한 조건문 생성.

        Args:
            start_date (str, optional): YYYYMM 형식의 문자열
            end_date (str, optional): YYYYMM 형식의 문자열

        Returns:
            dict: 시작날짜 <= 데이터 <= 종료날짜를 쿼리할 수 있는 조건문
        """
        query = {}

        if start_date:
            start = MongoDB.convert_datetime(datetime_string=start_date)
            if start:
                query['$gte'] = start

        if end_date:
            end = MongoDB.convert_datetime(datetime_string=end_date)
            if end:
                query['$lte'] = end

        return query

class Collection(MongoDB):
    """Collection Class

    Args:
        MongoDB (MongoDB Class Object): MongoDB 클래스
    """
    def __init__(self, database_name: str, collection_name: str):
        """컬렉션 연결 Object 초기화

        Args:
            database_name (str): 연결할 Database 명
            collection_name (str): 연결할 Collection 명
        """
        super().__init__(database_name=database_name)
        self._collection = self.database[collection_name]

    @property
    def object(self):
        """get collection object"""
        if self._collection is not None:
            return self._collection
        else:
            ValueError("Collection object is not initialized.",
                       get_code_line(inspect.currentframe()))

    def show_all_documents(self) -> None:
        """컬렉션 내 모든 document 출력."""
        cursor = self.object.find({})
        for item in cursor:
            print(item)

    def exists(self, key_name: str) -> bool:
        """키가 있는지 체크"""
        cursor = self.object.find_one({key_name: {"$exists": True}})
        if cursor:
            return True
        else:
            return False

    def rename_field_name(
        self, origin: str, rename: str, filter_query = None) -> Any:
        """필드명 변경

        Args:
            origin (str): 기존 필드명
            rename (str): 변경할 필드명
            filter_query (dict, optional): 필터링. Defaults to "".
        """
        if not filter_query:
            filter_query = {}
        return self.object.update_many(
            filter_query, {"$rename": {origin: rename}})

    def insert_document(self, document: dict) -> Any:
        """컬렉션에 document 추가.

        Args:
            document (dict): 추가할 document
        """
        return self.object.insert_one(document)

    def insert_document_many(self, documents: list) -> Any:
        """컬렉션에 document list 추가.

        Args:
            documents (list): document list
        """
        return self.object.insert_many(documents)

    def delete_document(self, filter_query: dict) -> Any:
        """document 삭제.

        Args:
            filter_query (dict): filter query.
        """
        return self.object.delete_one(filter_query)

    def find_one_and_replace(
        self, filter_query: dict, document: dict,
        upsert: bool = True) -> Any:
        """하나의 document를 찾고 전체 값을 변경.

        Args:
            filter_query (dict): 변경할 document에 대한 filter_query
            document (dict): 변경 document
            upsert (bool, optional): 없으면 insert. Defaults to False.
        """
        return self.object.find_one_and_replace(
            filter_query, document, upsert=upsert)

    def find_one_and_update(
        self, filter_query: dict, set_value: dict,
        upsert: bool = True) -> Any:
        """하나의 document를 찾고 특정 dict 값을 변경.

        Args:
            filter_query (dict): 변경할 document에 대한 filter_query
            set_value (dict): 변경할 key, value의 dict
            upsert (bool, optional): 없으면 insert. Defaults to False.
        """
        return self.object.find_one_and_update(
            filter_query, {'$set': set_value}, upsert=upsert)

    def query_early_or_last_datetime(self, sort: int) -> datetime | None:
        """'date' 값 중 가장 빠르거나 오래된 datetime 값 쿼리

        Args:
            sort (int): 1: 가장 오래된 날짜(오름차순).
                       -1: 가장 빠른 날짜(내림차순).

        Raises:
            ValueError: sort가 1, -1이 아닐 경우

        Returns:
            datetime | None: 변환된 날짜 데이터
        """
        match(sort):
            case pymongo.ASCENDING: # 오름차순
                pass
            case pymongo.DESCENDING: # 내림차순
                pass
            case _:
                raise ValueError(
                    "'sort' value 'pymongo.ASCENDING' or"
                    " 'pymongo.DESCENDING'.")

        if not self.exists('date'):
            return None
        # 전체 값 중 'date'만 쿼리하고 오름/내림 정렬 후 하나만 리턴.
        result = list(
            self.object.find(
                {}, {'date': 1, '_id': 0}).sort([('date', sort)]).limit(1))[0]
        if not result or 'date' not in result:
            return None

        return result['date']

    def get_collection_datetime_list(
            self, start_date: str = '', end_date: str = '') -> list:
        """컬렉션에서 날짜 데이터 리스트를 쿼리함.
           data Collection 일 경우: 시작날짜 <= 날짜 쿼리 <= 종료날짜
           data 외 Collection 일 경우: 시작날짜 이후의 종료 날짜 쿼리,
                                      종료날짜 이전의 시작 날짜 쿼리

        Args:
            start_date (str, optional): YYYYMM 형식의 문자열
            end_date (str, optional): YYYYMM 형식의 문자열

        Returns:
            list: 날짜 list
        """
        result = []

        # 시작, 종료 날짜가 둘다 있거나 둘다 없을 경우는 예외발생
        if (((start_date) and (end_date)) or
            ((not start_date) and (not end_date))):
            raise ValueError('Invalid Parameters.')

        # date 컬럼은 data database에만 존재함
        if self.exists(key_name='date'):
            query = Collection.make_date_range_query_with_equal(
                start_date=start_date, end_date=end_date)
            if query:
                result = self.object.find(
                   {'date': query},
                   {'date': True}).distinct('date')
        else:
            start_datetime = MongoDB.convert_datetime(
                datetime_string=start_date)
            end_datetime = MongoDB.convert_datetime(
                datetime_string=end_date)

            if start_datetime:
                result = self.object.distinct(
                    'end_date', {'start_date': start_datetime})
            elif end_datetime:
                result = self.object.distinct(
                    'start_date', {'end_date': end_datetime})

        return result

    def get_count_from_datetime(
            self, start_date: str = '', end_date: str = '') -> int:
        """컬렉션에서 해당 날짜의 데이터 개수를 쿼리함.
           'data' Collection 일 경우 : (시작날짜 <= 데이터 <= 종료날짜)
           'data' 외 Collection 일 경우 : 쿼리에 시작, 종료날짜 포함

        Args:
            start_date (str, optional): YYYYMM 형식의 문자열
            end_date (str, optional): YYYYMM 형식의 문자열

        Returns:
            int: 쿼리 결과 개수
        """
        query = {}

        # date 컬럼은 data database에만 존재함
        if self.exists(key_name='date'):
            result = Collection.make_date_range_query_with_equal(
                start_date=start_date, end_date=end_date)
            if result:
                query['date'] = result
        else:
            query = Collection.make_datetime_query(
                start=start_date, end=end_date)

        # 시작, 종료 날짜가 None이면 0 리턴
        if not query:
            return 0

        return self.object.count_documents(query)

class QueryBuilder:
    """dict을 파라미터로 받는 method."""
    @staticmethod
    def get_data_list(data: dict) -> list:
        """data dictionary 파라미터로 데이터를 쿼리함.
           (시작날짜 <= 데이터 <= 종료날짜)

        Args:
            data (dict): 'database', 'collection', 'start_date', 'end_date'

        Raises:
            ValueError: parameter에 필수 key가 없을 경우
            ValueError: 쿼리 결과가 없을 경우

        Returns:
            list: 쿼리 결과를 list로 변환
        """
        result = []
        if _check_key_in_dict(data) is False:
            raise ValueError("data not in 'database' or 'collection' key.",
                             get_code_line(inspect.currentframe()))

        start_date = data.get('start_date', '')
        end_date = data.get('end_date', '')

        col = Collection(data['database'], data['collection'])

        # 시작, 종료 날짜가 없으면 전체 데이터 쿼리.
        if (not start_date) and (not end_date):
            result = list(col.object.find({}))
        else:
            query = Collection.make_date_range_query_with_equal(
                start_date=start_date, end_date=end_date)
            if not query:
                result = list(col.object.find({}))
            else:
                result = list(col.object.find({'date': query}))

        if not result:
            raise Exception(
                'Cursor Object None.', get_code_line(inspect.currentframe()))

        return result

    @classmethod
    def query_table_data(cls, table: dict) -> dict:
        """table dictionary 파라미터로 데이터를 쿼리함.

        Args:
            table (dict): 'database', 'collection', 'start_date', 'end_date'

        Raises:
            ValueError: parameter에 필수 key가 없을 경우
            ValueError: 쿼리 결과가 dictionary type이 아닐 경우

        Returns:
            dict: 쿼리 결과
        """
        if not _check_key_in_dict(table):
            raise ValueError(
                "table not in 'database' or 'collection' key.",
                get_code_line(inspect.currentframe()))

        if ('start_date' not in table) or ('end_date' not in table):
            raise ValueError(
                "table not in 'start_date' or 'end_date' key.",
                get_code_line(inspect.currentframe()))
        
        col = Collection(table['database'], table['collection'])
        query = MongoDB.make_datetime_query(
            start=table.get('start_date', ''), end=table.get('end_date', ''))
        if not query:
            return {}
        result = col.object.find_one(query)
        if (not result) or (not isinstance(result, dict)):
            raise ValueError("Table Data Query Result Empty.",
                             get_code_line(inspect.currentframe()))
        return result

    @staticmethod
    def get_column_list(table: dict) -> list:
        """컬럼 정보 쿼리.

        Args:
            table (dict): 'database', 'collection', 'start_date', 'end_date'

        Raises:
            ValueError: 쿼리 결과에 columns 키가 없을 경우

        Returns:
            list: 컬럼 리스트
        """
        result = QueryBuilder.query_table_data(table=table)
        if 'columns' not in result:
            raise ValueError("Query Result Not in 'column' key.",
                             get_code_line(inspect.currentframe()))

        return result['columns']

    @staticmethod
    def get_object_id(table: dict) -> Any:
        """table collection에서 쿼리 결과의 id를 리턴.

        Args:
            table (dict): 'database', 'collection', 'start_date', 'end_date'

        Raises:
            ValueError: 파라미터가 잘못된 경우
            ValueError: 쿼리 결과에 _id 키가 없을 경우

        Returns:
            BSON: 해당 document의 ObjectID
        """
        result = QueryBuilder.query_table_data(table=table)
        if '_id' not in result:
            raise ValueError("Result not found '_id' key.",
                             get_code_line(inspect.currentframe()))
        return result['_id']

    @staticmethod
    def get_field_list(data: dict) -> list:
        """특정 월별 데이터의 컬럼명을 aggregate하는 기능

        Args:
            data (dict): 'database', 'collection', 'start_date', 'end_date'

        Raises:
            ValueError: 파라미터에 필수 key가 없을 경우

        Returns:
            list: 컬럼리스트
        """
        if _check_key_in_dict(data) is False:
            raise ValueError(
                "Parameters not in 'database' or 'collection' key.",
                get_code_line(inspect.currentframe()))

        date_query = MongoDB.make_date_range_query_with_equal(
            start_date=data.get('start_date', ''),
            end_date=data.get('end_date', ''))

        query = []
        if date_query:
            query.append({"$match": {"date": date_query}})
        # 필드값 -> array 로 변경
        query.append({"$project": {"o": {"$objectToArray": "$$ROOT"}}})
        # list 타입 값이 있으면 각각 document 로 변경
        query.append({"$unwind": "$o"})
        # key값(필드명)만 그룹화.
        query.append({"$group": {"_id": 0, "keys": {"$addToSet": "$o.k"}}})

        columns = []
        col = Collection(data['database'], data['collection'])
        result = list(col.object.aggregate(query))
        if not result:
            return columns

        columns = result[0]['keys']

        if 'label' in columns:
            columns.remove('label')

        if '_id' in columns:
            columns.remove('_id')

        if 'date' in columns:
            columns.remove('date')

        return columns

    @staticmethod
    def insert_model_information(
        train: dict, table: dict, model: dict,
        model_save_path_list: list) -> Any:
        """모델 컬렉션에 정보 저장.

        Args:
            train (dict): 'database', 'collection', 'start_date', 'end_date'
            table (dict): 'database', 'collection', 'start_date', 'end_date'
            model (dict): 'database', 'collection', 'path'
            model_save_path_list (str): 저장된 모델 경로 리스트

        Raises:
            ValueError: 파라미터가 잘못된 경우
            ValueError: 파라미터에 필수 key 가 없을 경우
        """
        if ((train is None) or (table is None) or (model is None) or
            (not model_save_path_list)):
            raise ValueError("Parameters is None.",
                             get_code_line(inspect.currentframe()))

        if ((not _check_key_in_dict(train)) or
            (not _check_key_in_dict(table)) or
            (not _check_key_in_dict(model))):
            raise ValueError(
                "Parameters not in 'database', 'collection', 'model' key.",
                get_code_line(inspect.currentframe()))

        col = Collection(model['database'], model['collection'])
        query = Collection.make_datetime_query(
            start=train.get('start_date', ''), end=train.get('end_date', ''))
        query['table'] = QueryBuilder.get_object_id(table=table)

        return col.find_one_and_update(
            filter_query=query,
            set_value={'path': model_save_path_list})

    @staticmethod
    def get_model_list(model: dict, table: dict) -> list:
        """모델 컬렉션에서 정보 쿼리.

        Args:
            model (dict): 'database', 'collection', 'start_date', 'end_date'
            table (dict): 'database', 'collection', 'start_date', 'end_date'

        Raises:
            ValueError: 파라미터가 잘못된 경우
            ValueError: 파라미터에 필수 key가 없을 경우
            ValueError: 쿼리한 모델 데이터가 없는 경우
            ValueError: 쿼리 결과에 path가 없는 경우

        Return:
            list: 모델 파일 경로 list
        """
        if (not model) or (not table):
            raise ValueError(
                "Parameter Empty.", get_code_line(inspect.currentframe()))

        if ((not _check_key_in_dict(table)) or
            (not _check_key_in_dict(model))):
            raise ValueError(
                "Parameters not in 'database' or 'collection' key.",
                get_code_line(inspect.currentframe()))

        col = Collection(model['database'], model['collection'])
        query = Collection.make_datetime_query(
            start=model.get('start_date', ''), end=model.get('end_date', ''))
        query['table'] = QueryBuilder.get_object_id(table=table)
        result = col.object.find_one(query, {'path': 1})
        if (not result) or ('path' not in result):
            raise ValueError("Query result not in 'path' key.",
                             get_code_line(inspect.currentframe()))
        return result['path']
