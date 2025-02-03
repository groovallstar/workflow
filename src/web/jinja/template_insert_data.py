from web.jinja.elements import PageName, Input, Select, Row, Card

class InsertDataId:
    """Insert Data Element Id Attributes."""
    prefix = 'id-'
    def __init__(self):
        """Init"""
        self._attributes = {}
        self._attributes['dataset'] = InsertDataId.prefix+'dataset'
        self._attributes['database'] = InsertDataId.prefix+'data-database'
        self._attributes['collection'] = InsertDataId.prefix+'data-collection'
        self._attributes['date'] = InsertDataId.prefix+'date'

    def get_collection_element_ids(self):
        """Generator Collection Element Id"""
        yield self._attributes['collection']

    @property
    def attributes(self): return self._attributes
    @property
    def dataset(self): return self._attributes['dataset']
    @property
    def database(self): return self._attributes['database']
    @property
    def collection(self): return self._attributes['collection']
    @property
    def date(self): return self._attributes['date']

class InsertDataElementList(InsertDataId):
    """데이터 저장 페이지에 사용되는 HTML Element 요소.

    Args:
        InsertDataId (Class): 각 element의 HTML id.
    """
    def __init__(self):
        """Init"""
        super().__init__()
        self._card_list = []
        data_set_column_list = [
            Select(
                id=self.dataset,
                label_text="Select DataSet Name",
                floating_message='Select DataSet Name',
                error_message='',
                select_list=[
                    dict(text='iris', selected=False),
                    dict(text='digits', selected=False),
                    dict(text='wine', selected=False),
                    dict(text='breast_cancer', selected=False)])
        ]

        data_column_list = [
            Select(
                id=self.database,
                label_text='Database',
                floating_message='Database',
                error_message='유효하지 않은 값입니다.',
                select_name='select-dbinfo'),
            Select(
                id=self.collection,
                label_text='Collection',
                floating_message='Collection',
                error_message='유효하지 않은 값입니다.',
                select_name='select-dbinfo'),
            Input(
                label_text='Date', input_type='text',
                floating_message='Date',
                error_message = \
                    '날짜 형식으로 YYYYMM을 입력하세요.',
                #input_class='span2',
                input_id=self.date,
                max_length='7', placeholder=' ',
                invalid_feedback=True)
        ]

        self._card_list.append(
            Card(header_small='Data Set',
                 row_list=[Row(column_list=data_set_column_list)]))
        self._card_list.append(
            Card(header_small='Database Information',
                 row_list=[Row(column_list=data_column_list, p=True)]))

    @property
    def card_list(self):
        """get card list"""
        return self._card_list

class InsertDataPage():
    """데이터 저장/테이블 저장 페이지."""
    def __init__(self):
        """Init"""
        self._card_list = []
        insert_data_element_list = InsertDataElementList()
        self._card_list.append(
            Card(card_id=PageName.INSERT_DATA.value,
                header_strong='Insert DataSet',
                 header_small='',
                 footer_btn_name='ADD',
                 add_btn_param_name=PageName.INSERT_DATA.value,
                 row_list=insert_data_element_list.card_list))

    @property
    def card_list(self):
        """get card list"""
        return self._card_list
