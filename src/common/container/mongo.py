import pymongo
from datetime import datetime
from typing import Union, Callable, Any
import inspect
from common.function import get_code_line

def verify_collection_object(func) -> Callable:
    """Collection 객체 유효성 체크 데코레이터"""
    def wrapper(self, *args, **kwargs):
        if self._collection is None:
            raise ValueError(
                "Collection object is not initialized.",
                get_code_line(inspect.currentframe()))
        else:
            return func(self, *args, **kwargs)
    return wrapper

def _check_key_in_dict(value: dict) -> bool:
    """dictionary에 database, collection key 가 있는지 체크"""
    if 'database' in value and 'collection' in value:
        return True
    else:
        return False

class MongoDB:
    """MongoDB Class"""

    def __init__(self, database_name: str = ""):
        """Database 연결.

        Args:
            database_name (Database 명): Database 명
        """
        import os
        mongodb_url = os.environ.get('MONGODB_URL')
        if not mongodb_url:
            raise Exception('MONGODB_URL environment variable not found.')
        self._connection = pymongo.MongoClient(
            mongodb_url, directConnection=True)
        self._database = None
        if database_name:
            self._database = self._connection.get_database(database_name)

    def __del__(self) -> None:
        """소멸자.
        """
        self.close()

    def close(self) -> None:
        """ Database 객체 소멸"""
        if self._connection:
            self._connection.close()

    @property
    def database(self):
        """Get database object"""
        return self._database

    def get_collection_list(self) -> list:
        """Get collection List.

        Raises:
            Exception: connection object empty.

        Returns:
            list: collection name list.
        """
        if self._connection is None:
            raise Exception('Database connection is not initialized.')

        if self._database is not None:
            return self._database.list_collection_names()

    def get_database_list(self) -> list:
        """Get Database List."""
        if self._connection:
            return self._connection.list_database_names()

class Collection(MongoDB):
    """Collection Class

    Args:
        MongoDB (MongoDB Class Object): MongoDB 클래스
    """
    def __init__(self, database_name: str, collection_name: str):
        """컬렉션 연결 Object 초기화

        Args:
            database_name (Database 명): 연결할 Database 명
            collection_name (Collection 명): 연결할 Collection 명
        """
        super().__init__(database_name=database_name)
        self._collection = self._database[collection_name]

    @property
    def object(self):
        """Get collection object"""
        return self._collection

    @verify_collection_object
    def show_all_documents(self) -> None:
        """컬렉션 내 모든 document 출력"""
        cursor = self._collection.find({})
        for item in cursor:
            print(item)

    @verify_collection_object
    def exists(self, key_name: str) -> bool:
        """키가 있는지 체크

        Args:
            key_name (str): key 이름

        Returns:
            bool: 있으면 True, 없으면 False
        """
        cursor = self._collection.find_one({key_name: {"$exists": True}})
        if cursor:
            return True
        else:
            return False

    @verify_collection_object
    def rename_field_name(
        self, origin: str, rename: str, filter_query: dict = None) -> Any:
        """필드명 변경

        Args:
            origin (str): 기존 필드명
            rename (str): 변경할 필드명
            filter_query (dict, optional): 필터링. Defaults to "".
        """
        if not filter_query:
            filter_query = {}

        return self._collection.update_many(
            filter_query, {"$rename": {origin: rename}})

    @verify_collection_object
    def insert_document(self, document: dict) -> None:
        """컬렉션에 document 추가.

        Args:
            document (dict): 추가할 document
        """
        return self._collection.insert_one(document)

    @verify_collection_object
    def insert_document_many(self, documents: list) -> None:
        """컬렉션에 document list 추가.

        Args:
            documents (list): document list
        """
        return self._collection.insert_many(documents)

    @verify_collection_object
    def delete_document(self, filter_query: dict = None) -> None:
        """document 삭제.

        Args:
            filter_query (dict, optional): filter query. Defaults to None.
        """
        self._collection.delete_one(filter_query)

    @verify_collection_object
    def find_one_and_replace(
        self, filter_query: dict, query: dict, upsert: bool = False) -> Any:
        """하나의 document를 찾고 변경.

        Args:
            filter_query (dict): 변경할 document에 대한 filter_query
            query (dict): 변경 document
            upsert (bool, optional): 없으면 insert. Defaults to False.
        """
        return self._collection.find_one_and_replace(
            filter_query, query, upsert=upsert)

    @classmethod
    def convert_string_to_datetime(
        cls, date_time_string: str) -> Union[datetime, None]:
        """문자열을 날짜값으로 변환.

        Args:
            date_time_string (str): YYYYMM 형식의 문자열

        Raises:
            BaseException: 날짜 변환이 실패할 경우

        Returns:
            datetime.datetime: 변환된 날짜 데이터
            None: 파라미터가 잘못될 경우, 날짜 변환이 실패할 경우
        """
        if not date_time_string:
            return None
        try:
            date = datetime.strptime(date_time_string, '%Y%m').date()
        except BaseException:
            return None
        return datetime(date.year, date.month, date.day)

    @verify_collection_object
    def query_start_date(
        self, start_date_string: str = None) -> Union[datetime, None]:
        """문자열(YYYYMM)로 컬렉션 날짜 데이터 쿼리

        Args:
            start_date_string (str): YYYYMM 형식의 문자열 \
                                    (None일 경우 컬렉션에서 \
                                    가장 오래된 날짜 값 쿼리)

        Raises:
            ValueError: 쿼리 결과가 1개가 아닐경우

        Returns:
            datetime.datetime: 변환된 날짜 데이터

        Description:
            특정 컬렉션의 시작 날짜부터 쿼리하고 싶을 경우\
            start_date_string=None 값으로 호출 가능
        """
        if start_date_string is None:
            result = []
            if self.exists('date'):
                result = list(
                    self._collection.find({}, {'date': 1, '_id': 0}).sort(
                        [('date', pymongo.ASCENDING)]).limit(1))
            if len(result) != 1:
                raise ValueError('Query Result Must Be One.',
                                 get_code_line(inspect.currentframe()))
            return result[0]['date']
        else:
            return Collection.convert_string_to_datetime(
                date_time_string=start_date_string)

    @verify_collection_object
    def query_end_date(
        self, end_date_string: str = None) -> Union[datetime, None]:
        """문자열(YYYYMM)로 컬렉션 날짜 데이터 쿼리

        Args:
            start_date_string (str): YYYYMM 형식의 문자열
                                    (None일 경우 컬렉션에서 가장 최근 날짜 값 쿼리)

        Raises:
            ValueError: 쿼리 결과가 1개가 아닐경우

        Returns:
            datetime.datetime: 변환된 날짜 데이터

        Description:
            특정 컬렉션의 최근 날짜부터 쿼리하고 싶을 경우
            start_date_string=None 값으로 호출 가능
        """
        if end_date_string is None:
            result = []
            if self.exists('date'):
                result = list(
                    self._collection.find({}, {'date': 1, '_id': 0}).sort(
                        [('date', pymongo.DESCENDING)]).limit(1))
            if len(result) != 1:
                raise ValueError('Query Result Must Be One.',
                                 get_code_line(inspect.currentframe()))
            return result[0]['date']
        else:
            return Collection.convert_string_to_datetime(
                date_time_string=end_date_string)

    @classmethod
    def make_date_query_from_data_collection(
        cls, start_date: str = None, end_date: str = None,
        ignore_convert_error = True) -> dict:
        """데이터 컬렉션의 날짜 데이터를 쿼리하기 위한 조건문 생성.

        Args:
            start_date (str, optional): YYYYMM 형식의 문자열
            end_date (str, optional): YYYYMM 형식의 문자열
            ignore_convert_error(bool, optional): 문자열 -> 날짜로
                변환 실패할 경우 쿼리문 생성을 실패 처리하는 Flag. default True.

        Returns:
            dictionary: 시작날짜 <= 데이터 <= 종료날짜를 쿼리할 수 있는 조건문
        """
        query = {}
        if (start_date is None) and (end_date is None):
            return query

        if start_date:
            start = cls.convert_string_to_datetime(
                date_time_string=start_date)
            if (ignore_convert_error is False) and (start is None):
                return query
            if start:
                query['$gte'] = start
        if end_date:
            end = cls.convert_string_to_datetime(
                date_time_string=end_date)
            if (ignore_convert_error is False) and (end is None):
                return query
            if end:
                query['$lte'] = end
        return query

    @verify_collection_object
    def get_datetime_list_from_data_collection(
            self, start_date: str = None, end_date: str = None) -> list:
        """데이터 컬렉션으로부터 날짜 데이터의 개수를 쿼리함.
           (시작날짜 <= 날짜값 <= 종료날짜)

        Args:
            start_date (str, optional): YYYYMM 형식의 문자열
            end_date (str, optional): YYYYMM 형식의 문자열

        Returns:
            list: 쿼리 결과
        """
        query = Collection.make_date_query_from_data_collection(
            start_date=start_date, end_date=end_date,
            ignore_convert_error=False)

        # 날짜값이 잘못될 경우는 날짜의 전체 값을 쿼리하지 않고 None을 리턴함.
        if not query:
            return None

        return self._collection.find(
            {'date': query},
            {'date': True}).distinct('date')

    @verify_collection_object
    def get_data_count_from_datetime(
            self, start_date: str = None, end_date: str = None) -> int:
        """데이터 컬렉션으로부터 해당 날짜의 데이터의 개수를 쿼리함.
           (시작날짜 <= 데이터 <= 종료날짜)

        Args:
            start_date (str, optional): YYYYMM 형식의 문자열
            end_date (str, optional): YYYYMM 형식의 문자열

        Returns:
            int: 쿼리 결과 개수
        """
        query = Collection.make_date_query_from_data_collection(
            start_date=start_date, end_date=end_date,
            ignore_convert_error=False)

        # 시작, 종료 날짜가 None이면 전체 데이터 쿼리
        if not query:
            return None

        return self._collection.count_documents({'date': query})

    @classmethod
    def get_data_list_from_data_dict(cls, data: dict) -> list:
        """data dictionary 값을 통해 컬렉션으로부터 데이터를 쿼리함.
           (시작날짜 <= 데이터 <= 종료날짜)

        Args:
            data (dict): 'database', 'collection', 'start_date', 'end_date'

        Raises:
            ValueError: parameter에 필수 key가 없을 경우
            ValueError: 쿼리 결과가 없을 경우

        Returns:
            list: 쿼리 결과를 list로 변환
        """
        if _check_key_in_dict(data) is False:
            raise ValueError("data not in 'database' or 'collection' key.",
                             get_code_line(inspect.currentframe()))

        start_date = data['start_date'] if 'start_date' in data else None
        end_date = data['end_date'] if 'end_date' in data else None

        collection = cls(data['database'], data['collection'])

        # 시작, 종료 날짜가 None이면 전체 데이터 쿼리
        if (start_date is None) and (end_date is None):
            result = collection.object.find({})
        else:
            query = cls.make_date_query_from_data_collection(
                start_date=start_date, end_date=end_date)
            if len(query) == 0:
                result = collection.object.find({})
            else:
                result = collection.object.find({'date': query})

        if result is None:
            raise Exception('Cursor Object None.',
                             get_code_line(inspect.currentframe()))

        return list(result)

    @classmethod
    def query_table_data(cls, table: dict) -> dict:
        """table collection에서 쿼리 결과 리턴.

        Args:
            table (dict): 'database','collection','start_date','end_date'

        Raises:
            ValueError: parameter에 필수 key가 없을 경우
            ValueError: 쿼리 결과가 dictionary type이 아닐 경우

        Returns:
            dict: 쿼리 결과
        """
        if _check_key_in_dict(table) is False:
            raise ValueError(
                "table not in 'database' or 'collection' key.",
                get_code_line(inspect.currentframe()))

        start_date = table['start_date'] if 'start_date' in table else None
        end_date = table['end_date'] if 'end_date' in table else None

        collection = cls(table['database'], table['collection'])
        query = {}
        if start_date:
            start = cls.convert_string_to_datetime(
                date_time_string=start_date)
            if start:
                query['start_date'] = start
        if end_date:
            end = cls.convert_string_to_datetime(
                date_time_string=end_date)
            if end:
                query['end_date'] = end

        if not query:
            query = None

        result = collection.object.find_one(query)
        if result is None:
            raise ValueError("Table Data Query Result None.",
                             get_code_line(inspect.currentframe()))

        if isinstance(result, dict) is False:
            raise ValueError(
                "Table Data Query Result is Not Dictionary Type.",
                get_code_line(inspect.currentframe()))
        return result

    @classmethod
    def get_columns_from_table_dict(cls, table: dict) -> list:
        """컬럼 정보 쿼리.

        Args:
            table (dict): 'database', 'collection',
                               'start_date', 'end_date'

        Raises:
            ValueError: 쿼리 결과에 columns 키가 없을 경우

        Returns:
            list: 컬럼 리스트
        """
        if not table:
            raise ValueError('table is None.',
                             get_code_line(inspect.currentframe()))

        result = cls.query_table_data(table=table)
        if 'columns' not in result:
            raise ValueError("Query Result Not in 'column' key.",
                             get_code_line(inspect.currentframe()))

        return result['columns']

    @classmethod
    def get_id_from_table_dict(cls, table: dict) -> Any:
        """table collection에서 쿼리 결과의 id를 리턴.

        Args:
            table (dict): 'database', 'collection',
                               'start_date', 'end_date'

        Raises:
            ValueError: 파라미터가 잘못된 경우
            ValueError: 쿼리 결과에 _id 키가 없을 경우

        Returns:
            [BSON]: 해당 document의 ObjectID
        """
        if not table:
            raise ValueError('table is None.',
                             get_code_line(inspect.currentframe()))

        result = cls.query_table_data(table=table)
        if '_id' not in result:
            raise ValueError("Query Result Not in 'column' key.",
                             get_code_line(inspect.currentframe()))

        return result['_id']

    @classmethod
    def get_field_list_from_data_dict(cls, data: dict) -> list:
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

        start_date = data['start_date'] if 'start_date' in data else None
        end_date = data['end_date'] if 'end_date' in data else None

        date_query = cls.make_date_query_from_data_collection(
            start_date=start_date, end_date=end_date)

        query = []
        if len(date_query) > 0:
            query.append({"$match": {"date": date_query}})
        # 필드값 -> array 로 변경
        query.append({"$project": {"o": {"$objectToArray": "$$ROOT"}}})
        # list 타입 값이 있으면 각각 document 로 변경
        query.append({"$unwind": "$o"})
        # key값(필드명)만 그룹화.
        query.append({"$group": {"_id": 0, "keys": {"$addToSet": "$o.k"}}})

        columns = []
        collection = Collection(data['database'], data['collection'])
        result = list(collection.object.aggregate(query))
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

    @classmethod
    def insert_model_information_from_dict(
        cls, data: dict, table: dict, model: dict,
        model_save_path_list: list) -> None:
        """모델 컬렉션에 정보 저장.

        Args:
            data (dict): 'database', 'collection', 'start_date', 'end_date'
            table (dict): 'database', 'collection', 'start_date', 'end_date'
            model (dict): 'database', 'collection', 'path'
            model_save_path_list (str): 저장된 모델 경로 리스트
        Raises:
            ValueError: 파라미터가 잘못된 경우
            ValueError: 파라미터에 필수 key 가 없을 경우
        """
        if ((data is None) or (table is None) or (model is None) or
            (not model_save_path_list)):
            raise ValueError("Parameters is None.",
                             get_code_line(inspect.currentframe()))

        if ((_check_key_in_dict(data) is False) or
            (_check_key_in_dict(table) is False) or
            (_check_key_in_dict(model) is False)):
            raise ValueError(
                "Parameters not in 'database' or 'collection' or 'model' key.",
                get_code_line(inspect.currentframe()))

        data_collection = cls(data['database'], data['collection'])
        start = data_collection.query_start_date(
            start_date_string=data['start_date'] \
                if 'start_date' in data else None)
        end = data_collection.query_end_date(
            end_date_string=data['end_date'] if 'end_date' in data else None)

        query = {}
        if start:
            query['start_date'] = start
        if end:
            query['end_date'] = end
        query['table'] = cls.get_id_from_table_dict(table=table)

        model_collection = cls(model['database'], model['collection'])
        model_collection.object.find_one_and_update(
            query,
            {'$set': {'path': model_save_path_list}},
            upsert=True)
        return

    @classmethod
    def get_model_list_from_dict(cls, model: dict, table: dict) -> list:
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
        if ((table is None) or (model is None)):
            raise ValueError("Parameters is None.",
                             get_code_line(inspect.currentframe()))

        if ((_check_key_in_dict(table) is False) or
            (_check_key_in_dict(model) is False)):
            raise ValueError(
                "Parameters not in 'database' or 'collection' key.",
                get_code_line(inspect.currentframe()))

        model_collection = cls(model['database'], model['collection'])
        start = model_collection.query_start_date(
            start_date_string=model['start_date'] \
                if 'start_date' in model else None)
        end = model_collection.query_end_date(
            end_date_string=model['end_date'] if 'end_date' in model else None)

        query = {}
        if start:
            query['start_date'] = start
        if end:
            query['end_date'] = end
        query['table'] = cls.get_id_from_table_dict(table=table)

        result = model_collection.object.find_one(query, {'path': 1})
        if result is None:
            raise ValueError('Query Model Data Empty.',
                             get_code_line(inspect.currentframe()))

        if 'path' not in result:
            raise ValueError("Query Result Not in 'path' key.",
                             get_code_line(inspect.currentframe()))

        return result['path']
