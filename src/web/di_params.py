class BaseQueryParams():
    """Base Query Params"""
    def __init__(
        self, database: str, collection: str,
        start_date: str = '', end_date: str = ''):
        """Initialize

        Args:
            database (str): database명
            collection (str): collection명
            start_date (str, optional): 시작날짜. Defaults to "".
            end_date (str, optional): 종료날짜. Defaults to "".
        """
        self._database = database
        self._collection = collection
        self._start_date = start_date
        self._end_date = end_date

    @property
    def database(self):
        """getter database"""
        return self._database

    @property
    def collection(self):
        """getter collection"""
        return self._collection

    @property
    def start_date(self):
        """getter start date"""
        return self._start_date

    @property
    def end_date(self):
        """getter end date"""
        return self._end_date

class DateQueryParams(BaseQueryParams):
    """DateTime Query Params"""
    def __init__(
        self,
        database: str, collection: str,
        start_date: str = '', end_date: str = ''):
        """Initialize
           입력 값에 따라 Query 조건이 달라지므로 파라미터 체크 method 존재

        Args:
            parameters: super init.
        """
        super().__init__(
            database=database, collection=collection,
            start_date=start_date, end_date=end_date)

    def queryable_start_date(self) -> bool:
        """시작 날짜 쿼리가 가능한지 체크
           시작, 종료 날짜 값이 파라미터에 없어야 함

        Returns:
            bool: 해당 파라미터로 쿼리 가능 여부 리턴
        """
        if (not self.start_date) and (not self.end_date):
            return True
        return False

    def queryable_end_date_in_start_date(self) -> bool:
        """시작 날짜 이후의 종료 날짜가 가능한지 체크
           시작 날짜 값만 존재 해야 함

        Returns:
            bool: 해당 파라미터로 쿼리 가능 여부 리턴
        """
        if (self.start_date) and (not self.end_date):
            return True
        return False

    def queryable_start_date_in_end_date(self) -> bool:
        """종료 날짜 이전의 시작 날짜가 가능한지 체크
           종료 날짜 값만 존재 해야 함

        Returns:
            bool: 해당 파라미터로 쿼리 가능 여부 리턴
        """
        if (not self.start_date) and (not self.end_date):
            return False
        if (self.start_date) and (self.end_date):
            return False
        return True

class CountQueryParams(DateQueryParams):
    """count REST API Query Params"""
    def __init__(
        self,
        database: str, collection: str,
        start_date: str = '', end_date: str = ''):
        """Initialize
           입력 값에 따라 Query 조건이 달라지므로 파라미터 체크 method 존재

        Args:
            database (str): database명
            collection (str): collection명
            start_date (str, optional): 시작날짜. Defaults to "".
            end_date (str, optional): 종료날짜. Defaults to "".
        """
        super().__init__(
            database=database, collection=collection,
            start_date=start_date, end_date=end_date)

    def queryable_collection(self) -> bool:
        """해당 Collection이 있는지 체크
           시작, 종료 날짜 값이 파라미터에 없어야 함

        Returns:
            bool: Collection 존재 여부
        """
        if (not self.start_date) and (not self.end_date):
            return True
        return False

    def queryable_document(self) -> bool:
        """해당 Document가 있는지 체크
           시작, 종료 날짜 값이 둘다 있어야 함

        Returns:
            bool: Document 존재 여부
        """
        if (self.start_date) and (self.end_date):
            return True
        return False
