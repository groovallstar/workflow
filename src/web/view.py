
from fastapi import Request, status
from fastapi.templating import Jinja2Templates
from starlette.templating import _TemplateResponse

from web.jinja.elements import PageName
from web.jinja.template_insert_data import InsertDataId, InsertDataPage
from web.jinja.template_insert_table import InsertTableId, InsertTablePage
from web.jinja.template_train_predict import TrainPredictId, TrainPredictPage

from web.handler import EndSlashRemoveRouter, PrettyJSONResponse

templates = Jinja2Templates(directory='templates')

view_router = EndSlashRemoveRouter(redirect_slashes=False)

@view_router.get('/')
async def load_base_page(request: Request) -> _TemplateResponse:
    """최초 페이지 Load.

    Args:  
        request (Request): Request Object

    Returns:  
        _TemplateResponse: base.html
    """
    return templates.TemplateResponse(
        name='base.html',
        context={
            'request': request,
        }, status_code=status.HTTP_200_OK)

@view_router.get('/contents')
async def get_contents_page(
    request: Request, page: PageName) -> _TemplateResponse:
    """각 페이지의 Jinja2 Template이 포함된 HTML 데이터 전달.

    Args:  
        request (Request): Request Object  
        page (PageName): Page 명

    Returns:  
        _TemplateResponse: contents.html
    """
    card_list = []
    if page == PageName.INSERT_DATA:
        card_list = InsertDataPage().card_list
    elif page == PageName.INSERT_TABLE:
        card_list = InsertTablePage().card_list
    elif page == PageName.TRAIN_PREDICT:
        card_list = TrainPredictPage().card_list

    return templates.TemplateResponse(
        name='contents.html',
        context={
            'request': request,
            'page_card_list': card_list
        }, status_code=status.HTTP_200_OK)

@view_router.get('/attributes')
async def get_page_attributes(page: PageName) -> PrettyJSONResponse:
    """Element의 ID 값 전달

    Args:  
        page (PageName): Page 명

    Returns:  
        PrettyJSONResponse: HTML Page의 ID 값 Dictioary
    """
    response_data = {}
    if page == PageName.INSERT_DATA:
        response_data['id'] = InsertDataId().attributes
    elif page == PageName.INSERT_TABLE:
        response_data['id'] = InsertTableId().attributes
    elif page == PageName.TRAIN_PREDICT:
        response_data['id'] = TrainPredictId().attributes
        response_data['train_prefix'] = TrainPredictId().train_prefix
        response_data['predict_prefix'] = TrainPredictId().predict_prefix
        response_data['all_prefix'] = TrainPredictId().all_prefix

    return response_data
