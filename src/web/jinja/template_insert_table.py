from web.jinja.elements import PageName, Select, Row, Card

class InsertTableId:
    """Insert Table Element Id Attributes."""
    prefix = 'it-'

    def __init__(self):
        """Init."""
        self._attributes = {}
        self._attributes['data'] = {}
        self._attributes['data']['database'] = \
            InsertTableId.prefix + 'data-database'
        self._attributes['data']['collection'] = \
            InsertTableId.prefix + 'data-collection'
        self._attributes['data']['start_date'] = \
            InsertTableId.prefix + 'data-startdate'
        self._attributes['data']['end_date'] = \
            InsertTableId.prefix + 'data-enddate'

        self._attributes['table'] = {}
        self._attributes['table']['database'] = \
            InsertTableId.prefix + 'table-database'
        self._attributes['table']['collection'] = \
            InsertTableId.prefix + 'table-collection'

    def get_collection_element_ids(self):
        """Generator Collection Element Id"""
        yield self._attributes['data']['collection']
        yield self._attributes['table']['collection']

    @property
    def attributes(self): return self._attributes
    @property
    def data_database(self): return self._attributes['data']['database']
    @property
    def data_collection(self): return self._attributes['data']['collection']
    @property
    def data_startdate(self): return self._attributes['data']['start_date']
    @property
    def data_enddate(self): return self._attributes['data']['end_date']
    @property
    def table_database(self): return self._attributes['table']['database']
    @property
    def table_collection(self): return self._attributes['table']['collection']

class InsertTableElementList(InsertTableId):
    """테이블 정보 저장 페이지에 사용되는 HTML Element 요소.

    Args:
        InsertTableId (Class): 각 element의 HTML id.
    """
    def __init__(self):
        """Init."""
        super().__init__()
        self._card_list = []

        data_column_list = [
            Select(
                id=self.data_database,
                label_text='Database',
                floating_message='Database', 
                error_message='유효하지 않은 값입니다.', 
                select_name='select-dbinfo'),
            Select(
                id=self.data_collection,
                label_text='Collection', 
                floating_message='Collection', 
                error_message='유효하지 않은 값입니다.',
                select_name='select-dbinfo'),
            Select(
                id=self.data_startdate,
                label_text='StartDate', 
                floating_message='StartDate', 
                error_message='유효하지 않은 값입니다.',
                select_name='select-dbinfo'),
            Select(
                id=self.data_enddate,
                label_text='EndDate',
                floating_message='EndDate', 
                error_message='유효하지 않은 값입니다.',
                select_name='select-dbinfo'),
        ]

        table_column_list = [
            Select(
                id=self.table_database,
                label_text="Database",
                floating_message='Database',
                select_name='select-dbinfo',
                error_message='유효하지 않은 값입니다.'),
            Select(
                id=self.table_collection,
                label_text="Collection",
                floating_message='Collection',
                select_name='select-dbinfo',
                error_message='유효하지 않은 값입니다.'),
        ]

        self._card_list.append(
            Card(header_small='Data Database Setting',
                 row_list=[Row(column_list=data_column_list)]))
        self._card_list.append(
            Card(header_small='Table Database Setting',
                 row_list=[Row(column_list=table_column_list,)]))

    @property
    def card_list(self):
        return self._card_list

class InsertTablePage():
    """데이터 저장/테이블 저장 페이지."""
    def __init__(self):
        """Init."""
        self._card_list = []
        insert_table_element_list = InsertTableElementList()
        self._card_list.append(
            Card(card_id=PageName.INSERT_TABLE.value,
                header_strong='Insert Table',
                 footer_btn_name='ADD',
                 add_btn_param_name=PageName.INSERT_TABLE.value,
                 row_list=insert_table_element_list.card_list))

    @property
    def card_list(self):
        return self._card_list
