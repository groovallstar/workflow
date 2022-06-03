
import enum
class PageName(str, enum.Enum):
    """Define Page Names"""
    INSERT_DATA = 'INSERT_DATA'
    INSERT_TABLE = 'INSERT_TABLE'
    TRAIN_PREDICT = 'TRAIN_PREDICT'

class Input:
    """HTML Input Element"""
    def __init__(
        self, label_text, floating_message, error_message,
        input_type, input_class: str='', input_id: str='',
        input_name: str='', min: str='', max: str='', step: str='',
        value: str='', placeholder: str='', max_length: str='',
        invalid_feedback:bool=False):
        """Init.

        Args:
            label_text (str): input text.
            floating_message (str): floating label 에 보여줄 메시지.
            error_message (str): 유효하지 않을때 에러 메시지.
            input_type (str): 해당 element의 type.
            input_class (str, optional): HTML class.
            input_id (str, optional): HTML id.
            input_name (str, optional): HTML name.
            min (str, optional): number type 일 때 최소값 지정.
            max (str, optional): number type 일 때 최대값 지정.
            step (str, optional): number type 일 때 증가 값.
            value (str, optional): element 값.
            placeholder (str, optional): 플레이스홀더.
            max_length (str, optional): 최대값.
            invalid_feedback (bool, optional): 각 input 하위에 에러메시지 사용.
        """
        self._type = 'input'
        self._input_id = input_id
        self._label_text = label_text
        self._floating_message = floating_message
        self._error_message = error_message
        self._input_type = input_type
        self._input_class = input_class
        self._input_name = input_name
        self._min = min
        self._max = max
        self._step = step
        self._value = value
        self._placeholder = placeholder
        self._max_length = max_length
        self._invalid_feedback = invalid_feedback

    @property
    def type(self): return self._type
    @property
    def input_class(self): return self._input_class
    @property
    def input_id(self): return self._input_id
    @property
    def label_text(self): return self._label_text
    @property
    def floating_message(self): return self._floating_message
    @property
    def error_message(self): return self._error_message
    @property
    def input_type(self): return self._input_type
    @property
    def input_name(self): return self._input_name
    @property
    def min(self): return self._min
    @property
    def max(self): return self._max
    @property
    def step(self): return self._step
    @property
    def value(self): return self._value
    @property
    def placeholder(self): return self._placeholder
    @property
    def max_length(self): return self._max_length
    @property
    def invalid_feedback(self): return self._invalid_feedback

class DataList:
    """HTML Input With DataList Element"""
    def __init__(self,
        datalist_id: str, label_text: str, floating_message: str,
        error_message: str, input_name: str, input_class: str='',
        input_id: str='', max_length: str='', placeholder: str=''):
        """Init

        Args:
            datalist_id (str): HTML id.
            label_text (str): HTML label text.
            floating_message (str): floating label 에 보여줄 메시지.
            error_message (str): 유효하지 않을때 에러 메시지.
            input_name (str): HTML name.
            input_class (str, optional): HTML class.
            input_id (str, optional): HTML id.
            max_length (str, optional): 최대값.
            placeholder (str, optional): 플레이스홀더.
        """
        self._type = 'input_datalist'
        self._datalist_id = datalist_id
        self._input_class = input_class
        self._input_id = input_id
        self._label_text = label_text
        self._floating_message = floating_message
        self._error_message = error_message
        self._input_name = input_name
        self._max_length = max_length
        self._placeholder = placeholder

    @property
    def type(self): return self._type
    @property
    def datalist_id(self): return self._datalist_id
    @property
    def input_class(self): return self._input_class
    @property
    def input_id(self): return self._input_id
    @property
    def label_text(self): return self._label_text
    @property
    def floating_message(self): return self._floating_message
    @property
    def error_message(self): return self._error_message
    @property
    def input_name(self): return self._input_name
    @property
    def max_length(self): return self._max_length
    @property
    def placeholder(self): return self._placeholder
    
class CheckBox:
    """HTML CheckBox Element"""
    def __init__(self, id: str, text: str, name: str='', checked: bool=False):
        """Init.

        Args:
            id (str): HTML id.
            text (str): checkbox의 label text.
            name (str, optional): HTML name.
            checked (bool, optional): 초기 상태 값.
        """
        self._type = 'checkbox'
        self._id = id
        self._name = name
        self._text = text
        self._checked = checked

    @property
    def type(self): return self._type
    @property
    def id(self): return self._id
    @property
    def name(self): return self._name
    @property
    def text(self): return self._text
    @property
    def checked(self): return self._checked

class Select:
    """HTML Select Element"""
    def __init__(
        self, id: str, label_text: str, floating_message: str,
        error_message: str, select_list: list=[],
        select_name: str='', selected_text: str=''):
        """Init.

        Args:
            id (str): HTML id.
            label_text (str): Select의 label text.
            floating_message (str): floating label 에 보여줄 메시지.
            error_message (str): 유효하지 않을때 에러 메시지.
            select_list (list): 초기 select 하위에 보여줄 목록.
            select_name (str, optional): HTML name.
            selected_text (str, optional): 초기 선택된 항목 표시.
        """
        self._type = 'select'
        self._id = id
        self._label_text = label_text
        self._floating_message = floating_message
        self._error_message = error_message
        self._select_list = select_list
        self._select_name = select_name
        self._selected_text = selected_text

    @property
    def type(self): return self._type
    @property
    def id(self): return self._id
    @property
    def label_text(self): return self._label_text
    @property
    def floating_message(self): return self._floating_message
    @property
    def error_message(self): return self._error_message
    @property
    def select_list(self): return self._select_list
    @property
    def select_name(self): return self._select_name
    @property
    def selected_text(self): return self._selected_text

class Row:
    """Row Component"""
    def __init__(
        self, label_text: str='', column_list: list=[],
        column_class: str='', hr: bool=False, p: bool=False):
        """Init.

        Args:
            label_text (str, optional): Label Text.
            column_list (str, optional): 컬럼 리스트.
            column_class (str, optional): HTML col class.
            hr (bool, optional): 한 row 아래에 hr tag 추가.
            p (bool, optional): 한 row 아래에 p tag 추가.
        """
        self._label_text = None
        if label_text:
            self._label_text = label_text
        self._column_list = column_list
        self._column_class = None
        if column_class:
            self._column_class = column_class
        self._hr = hr
        self._p = p

    @property
    def label_text(self): return self._label_text
    @property
    def column_list(self): return self._column_list
    @property
    def column_class(self): return self._column_class
    @property
    def hr(self): return self._hr
    @property
    def p(self): return self._p

class Card:
    """HTML Card Component"""
    def __init__(
        self, card_id: str='', header_strong: str='', header_small: str='',
        footer_btn_name: str='', add_btn_param_name: str='',
        row_list: list=[]):
        """Init.

        Args:
            card_id (str, optional): HTML id.
            header_strong (str, optional): 헤더의 strong tag.
            header_small (str, optional): 헤더의 small tag.
            footer_btn_name (str, optional): 푸터의 버튼 이름.
            add_btn_param_name (str, optional): 푸터 버튼 클릭 시 넘길 문자열.
            row_list (list, optional): card 안에 넣을 row list.
        """
        self._card_id = None
        if card_id:
            self._card_id = card_id

        self._header_strong = None
        if header_strong:
            self._header_strong = header_strong

        self._header_small = None
        if header_small:
            self._header_small = header_small

        self._footer_btn_name = None
        if footer_btn_name:
            self._footer_btn_name = footer_btn_name

        self._add_btn_param_name = None
        if add_btn_param_name:
            self._add_btn_param_name = add_btn_param_name

        self._row_list = row_list

    @property
    def card_id(self): return self._card_id
    @property
    def header_strong(self): return self._header_strong
    @property
    def header_small(self): return self._header_small
    @property
    def footer_btn_name(self): return self._footer_btn_name
    @property
    def add_btn_param_name(self): return self._add_btn_param_name
    @property
    def row_list(self): return self._row_list
