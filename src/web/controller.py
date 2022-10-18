from typing import AsyncGenerator, Union

from fastapi import Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.params import Depends

from aioredis import Channel, Redis

from fastapi_plugins import depends_redis
from sse_starlette.sse import EventSourceResponse

from common.container.mongo import MongoDB, Collection
from common.trace_log import TraceLog

from web.settings import get_settings
from web.handler import EndSlashRemoveRouter, JsonException
from web.handler import PrettyJSONResponse
from web.di_params import CountQueryParams, DateQueryParams
from web.jinja.elements import PageName

from web.worker import celery_object

controller_router = EndSlashRemoveRouter(redirect_slashes=False)

async def subscribe(channel: str, redis: Redis) -> AsyncGenerator:
    """Redis 에서 Publish 한 내용 Subscribe Procedure.

    Args:  
        channel (str): subscribe 채널.  
        redis (Redis): subscribe redis object

    Yields:  
        Iterator[dict]: task id를 브라우저로 전달
    """
    (channel_subscription,) = await redis.subscribe(
        channel=Channel(channel, False))
    while await channel_subscription.wait_message():
        message = await channel_subscription.get()

        import json
        json_message = json.loads(message)

        # celery 로부터 전달된 message 형식. id = task id
        if 'id' in json_message:
            celery_object.AsyncResult(json_message['id']).forget()
            yield {"event": channel, "data": json_message['id']}

@controller_router.get("/stream")
async def stream(
    channel: str=get_settings().redis_subscribe_channel,
    redis: Redis=Depends(depends_redis)) -> EventSourceResponse:
    """SSE Subscribe.

    Args:  
        channel (str, optional): subscribe 채널.  
        redis (Redis, optional): subscribe redis 정보.  

    Raises:  
        Exception: REDIS_SUBSCRIBE_CHANNEL 환경변수 값을 읽어오지 못할 경우  

    Returns:  
        EventSourceResponse: SSE Registration
    """
    if not channel:
        TraceLog().info(
            'REDIS_SUBSCRIBE_CHANNEL environment variable not found.')
        raise Exception(
            'REDIS_SUBSCRIBE_CHANNEL environment variable not found.')

    return EventSourceResponse(subscribe(channel, redis))

@controller_router.get('/settings')
async def get_last_settings(
    page: PageName) -> Union[Response, PrettyJSONResponse]:
    """각 페이지에서 사용자가 마지막으로 설정한 element 값 전달

    Args:  
        page (PageName): page 명

    Returns:  
        Response: 마지막으로 설정한 값이 없을 경우 204 NO CONTENT  
        PrettyJSONResponse: 사용자가 마지막으로 설정한 Element 값
    """
    response_data = {}
    try:
        col = Collection('web', 'last_setting')
        response_data = col.object.find_one(
            {'_id': page},
            {'_id': 0})
    except BaseException as ex:
        TraceLog().info(ex)
        response_data = {}

    if not response_data:
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return PrettyJSONResponse(
        content=response_data, status_code=status.HTTP_200_OK)

@controller_router.get('/sel_list_db')
async def select_list_for_database() -> Union[Response, PrettyJSONResponse]:
    """Query Database list.

    Returns:  
        Response: 마지막으로 설정한 값이 없을 경우 204 NO CONTENT  
        PrettyJSONResponse: 'web' database의 select_list에 설정된 database list
    """
    response_data = {}
    try:
        col_select_list = Collection('web', 'select_list')
        query_result = col_select_list.object.find_one(
            {'_id': 'database'}, {'database': 1, '_id': 0})
        response_data = query_result['database']
    except BaseException as ex:
        TraceLog().info(ex)
        response_data = {}

    if not response_data:
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return PrettyJSONResponse(
        content=response_data, status_code=status.HTTP_200_OK)

@controller_router.get('/sel_list_col')
async def select_list_for_collection(
    element_id: str, database: str) -> Union[Response, PrettyJSONResponse]:
    """Query Collection list.

    Args:  
        element_id (str): '페이지명-카테고리-쿼리 키'로 이뤄진 element id  
        database (str): database명

    Returns:  
        Response: Query 결과가 없을 경우 204 NO CONTENT  
        PrettyJSONResponse: collection list
    """
    response_data = []

    try:
        web_select = Collection('web', 'select_list')
        query = {}
        # element id 형식이 '페이지명-컬렉션 카테고리-쿼리할 키'이므로
        # 0번째와 마지막 인덱스를 제거한 문자열을 만듦.
        query['_id'] = '-'.join(element_id.split('-')[1:-1])
        result = web_select.object.find_one(query, {'collection': 1})

        db = MongoDB(database)
        collection_list = db.get_collection_list()

        if (result) and ('collection' in result) and (result['collection']):
            for name in collection_list:
                for post_fix in result['collection']:
                    # 각 collection 명이 '문서.post-fix' 로 되어 있으므로
                    # 해당 문자열을 찾음.
                    if name.endswith(post_fix):
                        response_data.append(name)
        else:
            # list에서 제한할 목록이 없을 경우 전체 collection을 전달함.
            response_data = collection_list

    except BaseException as ex:
        TraceLog().info(ex)
        response_data = []

    if not response_data:
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return PrettyJSONResponse(
        content=response_data, status_code=status.HTTP_200_OK)

@controller_router.get('/sel_list_date')
async def select_list_for_date(
    params: DateQueryParams = Depends()
    ) -> Union[Response, PrettyJSONResponse]:
    """날짜 데이터 쿼리.

    Args:  
        DateQueryParams.database (str): database명  
        DateQueryParams.collection (str): collection명  
        DateQueryParams.start_date (Union[str, None]): 시작날짜.  
        DateQueryParams.end_date (Union[str, None]): 종료날짜.

    Raises:  
        JsonException: start_date, end_date 둘다 있을경우 예외처리

    Returns:  
        Response: Query 결과가 없을 경우 204 NO CONTENT  
        PrettyJSONResponse: 'start_date' = 시작날짜 list,
                            'end_date' = 종료날짜 list
    """
    response_data = {}
    result = []

    # 파라미터가 database, collection 일 경우
    if params.queryable_start_date():
        try:
            col = Collection(params.database, params.collection)
            if col.exists(key_name='date'):
                result = col.object.distinct('date')
            else:
                result = col.object.distinct('start_date')
        except BaseException as ex:
            TraceLog().info(ex)
            result = []

        if result:
            response_data['start_date'] = jsonable_encoder(result)

    # 파라미터가 database, collection, start_date 일 경우
    elif params.queryable_end_date_in_start_date():
        try:
            col = Collection(params.database, params.collection)
            result = col.get_datetime_list_from_collection(
                start_date=params.start_date, end_date=params.end_date)
        except BaseException as ex:
            TraceLog().info(ex)
            result = []

        if result:
            response_data['end_date'] = jsonable_encoder(result)

    # database, collection, end_date 입력할 경우
    elif params.queryable_start_date_in_end_date():
        try:
            col = Collection(params.database, params.collection)
            result = col.get_datetime_list_from_collection(
                start_date=params.start_date, end_date=params.end_date)
        except BaseException as ex:
            TraceLog().info(ex)
            result = []

        if result:
            response_data['start_date'] = jsonable_encoder(result)
    else:
        TraceLog().info('Query Parameter Incorrected.')
        raise JsonException(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                'status_code': status.HTTP_404_NOT_FOUND,
                'status': 'Query Parameter Incorrected.'})

    if not response_data:
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return PrettyJSONResponse(
        content=response_data, status_code=status.HTTP_200_OK)

@controller_router.get('/count', response_class=Response)
async def validation_select_list(
    params: CountQueryParams = Depends()) -> Response:
    """웹에서 설정한 값이 실제 존재하는지 체크.

    Args:  
        CountQueryParams.database (str): database명  
        CountQueryParams.collection (str): collection명  
        CountQueryParams.start_date (Union[str, None]): 시작날짜.  
        CountQueryParams.end_date (Union[str, None]): 종료날짜.

    Raises:  
        JsonException: start_date, end_date 값이 하나라도
                      없을 경우 HTTP_400_BAD_REQUEST  
        JsonException: 쿼리 결과가 없을 경우 HTTP_404_NOT_FOUND

    Returns:  
        Response: Query 결과가 있으면 HTTP_200_OK
    """
    document_count = 0
    # 컬렉션 쿼리가 가능한지 체크
    if params.queryable_collection():
        try:
            db = MongoDB(params.database)
            document_count = params.collection in db.get_collection_list()
        except BaseException as ex:
            TraceLog().info(ex)
            document_count = 0
    # Document 쿼리가 가능한지 체크
    elif params.queryable_document():
        try:
            col = Collection(params.database, params.collection)
            document_count = col.get_data_count_from_datetime(
                start_date=params.start_date, end_date=params.end_date)
        except BaseException as ex:
            TraceLog().info(ex)
            document_count = 0
    else:
        TraceLog().info('Query Parameter Incorrected.')
        raise JsonException(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                'status_code': status.HTTP_400_BAD_REQUEST,
                'status': 'Query Parameter Incorrected.'})

    if not document_count:
        TraceLog().info('Query Result Empty.')
        raise JsonException(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                'status_code': status.HTTP_404_NOT_FOUND,
                'status': 'Query Result Empty.'})

    return Response(status_code=status.HTTP_200_OK)

@controller_router.post('/task', response_class=Response)
async def send_task(parameters: dict) -> Response:
    """웹에서 설정한 Element의 값을 celery에 전달

    Args:  
        parameters (dict): 페이지에서 설정한 Element 값

    Raises:  
        JsonException: page_name 또는 task_id 키값이 잘못될 경우  
        JsonException: page 명을 못 찾을 경우  

    Returns:  
        Response: HTTP_200_OK
    """
    if (('page_name' not in parameters) or ('task_id' not in parameters) and
        (not parameters['page_name']) or (not parameters['task_id'])):
        TraceLog().info('page_name, task_id Value Incorrected.')
        raise JsonException(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                'status_code': status.HTTP_400_BAD_REQUEST,
                'status': 'page_name, task_id Value Incorrected.'})

    page_name = parameters['page_name']
    task_id = parameters['task_id']

    task_function = ""
    if page_name == PageName.INSERT_DATA:
        task_function = 'tasks.insertdata'
    elif page_name == PageName.INSERT_TABLE:
        task_function = 'tasks.inserttable'
    elif page_name == PageName.TRAIN_PREDICT:
        task_function = 'tasks.pipeline'
    elif page_name == PageName.FEATURE_SELECTION:
        task_function = 'tasks.featureselection'
    else:
        TraceLog().info('Page Name Not Found.')
        raise JsonException(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                'status_code': status.HTTP_400_BAD_REQUEST,
                'status': 'Page Name Not Found.'})

    # 마지막 설정 값 저장.
    try:
        col = Collection('web', 'last_setting')
        query = parameters.copy()
        query['_id'] = page_name
        col.find_one_and_replace({'_id': page_name}, query, upsert=True)
    except BaseException as ex:
        TraceLog().info(ex)

    # print(task_id)
    celery_object.send_task(
        task_function,
        args=[],
        kwargs=parameters,
        task_id=task_id,
        ignore_result=True)

    return Response(status_code=status.HTTP_200_OK)
