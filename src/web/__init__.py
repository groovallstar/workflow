import os
import json
from datetime import timedelta

import redis
from flask import Flask, make_response, Response, request
from flask import render_template, jsonify

from common.container.mongo import MongoDB, Collection

from web.jinja.elements import PageName
from web.jinja.template_insert_data import InsertDataId, InsertTableId
from web.jinja.template_insert_data import InsertDataPage
from web.jinja.template_train_predict import TrainPredictId
from web.jinja.template_train_predict import TrainPredictPage

from web.worker import celery_object

app = Flask(__name__)
app.debug = False

# trim_blocks app config
#app.jinja_env.trim_blocks = True

app.config.update(
    SECRET_KEY='X1243yRH!mMwf',
    SESSION_COOKIE_NAME='pyweb_flask_session',
    PERMANENT_SESSION_LIFETIME=timedelta(31))

REDIS_HOST_IP = os.environ.get('REDIS_HOST_IP')
strict_redis = redis.StrictRedis(
    host=REDIS_HOST_IP, port=6379, encoding='utf-8', decode_responses=True)

def event_stream():
    """Server-Sent-Event 응답

    Yields:
        str: event:'flask-event', data:'task 아이디'
    """
    pub = strict_redis.pubsub()
    pub.subscribe('background')
    for msg in pub.listen():
        if msg['type'] != 'subscribe':
            event, task_id = json.loads(msg['data'])
            celery_object.AsyncResult(task_id).forget()
            #print(f'event=({event}) data=({task_id})')
            yield 'event: {0}\ndata: {1}\n\n'.format(event, task_id)

        else:
            yield 'data: {0}\n\n'.format(msg['data'])

@app.route('/stream')
def stream():
    """Server-Sent-Event 초기화"""
    return Response(event_stream(), mimetype="text/event-stream")

@app.route('/')
def index():
    """최초 페이지 route"""
    return render_template('base.html')

@app.route('/contents', methods=['GET'])
def load_contents_page():
    """각 페이지의 Jinja2 Template이 포함된 HTML 데이터 전달"""
    response_data = jsonify({'success': True})
    status_code = 200

    if isinstance(request.args, dict) is False:
        response_data = jsonify({'error': 'NotFound'})
        status_code = 400
        return make_response(response_data, status_code)

    args_dict = request.args.to_dict()
    if 'page' not in args_dict:
        response_data = jsonify({'error': 'NotFound'})
        status_code = 400
        return make_response(response_data, status_code)

    page = None
    result = {}
    if (args_dict['page'] == PageName.INSERT_DATA):
        page = InsertDataPage()
    elif (args_dict['page'] == PageName.TRAIN_PREDICT):
        page = TrainPredictPage()

    if page:
        result['html'] = render_template(
            'contents.html',
            page_card_list=page.card_list)

    # 값 설정 확인.
    if (page is not None) and (len(result) > 0):
        response_data = jsonify(result)
        status_code = 200
    else:
        response_data = jsonify({'error': 'NotFound'})
        status_code = 400

    return make_response(response_data, status_code)

@app.route('/lists', methods=['GET'])
def query_select_lists():
    """웹에서 설정한 값 MongoDB에서 쿼리.

    Returns:
       파라미터 없을 경우: MongoDB Document 목록 리턴
       'database'만 있을 경우: 해당 Document의 Collection 목록 리턴
       'database', 'collection' 만 있을 경우: 시작날짜 목록 리턴
       'database', 'collection', 'startdate' 만 있을 경우: 종료날짜 목록 리턴
       'database', 'collection', 'enddate' 만 있을 경우: 시작날짜 목록 리턴
    """
    response_data = jsonify({'error': 'NotFound'})
    status_code = 200

    if isinstance(request.args, dict) is False:
        response_data = jsonify({'error': 'NotFound'})
        status_code = 400
        return make_response(response_data, status_code)

    args_dict = request.args.to_dict().copy()

    # element id 는 database 쿼리시 제외하기 위해 있으면 따로 저장함.
    element_id = None
    if 'id' in args_dict:
        element_id = args_dict.pop('id', None)

    query_result = {}
    result = []

    # 파라미터 없이 호출할 경우
    if ((len(args_dict) == 0)):
        try:
            col_select_list = None
            # 기존 전체 document를 쿼리하지 않고 제한된 document만 쿼리하도록 수정
            col_select_list = Collection('web', 'select_list')
            query = {}
            query['_id'] = 'database'
            cursor = col_select_list.object.find_one(query, {'database': 1})
            if cursor:
                result = cursor['database']
        finally:
            if col_select_list:
                col_select_list.close()

        query_result['database'] = result

    # database만 입력할 경우
    elif ((len(args_dict) == 1) and ('database' in args_dict)):
        # element id 가 없을 경우 None 리턴
        if element_id is None:
            result = []
        else:
            try:
                db = None
                col_select_list = None

                db = MongoDB(args_dict['database'])
                collection_list = db.get_collection_list()

                col_select_list = Collection('web', 'select_list')
                query = {}
                # element id 형식이 '페이지명-컬렉션 카테고리-쿼리할 키'이므로
                # 0번째와 마지막 인덱스를 제거한 문자열을 만듦.
                query['_id'] = '-'.join(element_id.split('-')[1:-1])
                find_result = col_select_list.object.find_one(
                    query, {'collection': 1})

                if find_result is None:
                    result = collection_list
                else:
                    post_fix_string_list = find_result['collection']
                    if len(post_fix_string_list) > 0:
                        for collection_name in collection_list:
                            # 각 collection 명이 '문서.post-fix' 로 되어 있으므로
                            # 해당 문자열을 찾음.
                            for post_fix_string in post_fix_string_list:
                                if collection_name.endswith(post_fix_string):
                                    result.append(collection_name)
                    else:
                        result = collection_list
            finally:
                if db: 
                    db.close()
                if col_select_list: 
                    col_select_list.close()

        query_result['collection'] = result

    # database, collection 입력할 경우
    elif ((len(args_dict) == 2) and
          ('database' in args_dict) and ('collection' in args_dict)):
        try:
            col = None
            col = Collection(args_dict['database'], args_dict['collection'])
            if col.exists(key_name='date'):
                result = col.object.distinct('date')
            else:
                result = col.object.distinct('start_date')
        finally:
            if col:
                col.close()

        query_result['start_date'] = result

    # database, collection, start_date 입력할 경우
    elif ((len(args_dict) == 3) and
          ('database' in args_dict) and ('collection' in args_dict) and
          ('start_date' in args_dict)):
        try:
            col = None
            col = Collection(args_dict['database'], args_dict['collection'])

            if col.exists(key_name='date'):
                result = col.get_datetime_list_from_data_collection(
                    start_date=args_dict['start_date'])
            else:
                start_date = Collection.convert_string_to_datetime(
                    date_time_string=args_dict['start_date'])
                # 시작 날짜 변환에 실패했으면 None 처리.
                if start_date:
                    query = {}
                    query['start_date'] = start_date
                    result = col.object.distinct('end_date', query)
                else:
                    result = []
        finally:
            if col:
                col.close()

        query_result['end_date'] = result

    # database, collection, end_date 입력할 경우
    elif ((len(args_dict) == 3) and
          ('database' in args_dict) and ('collection' in args_dict) and
          ('end_date' in args_dict)):
        try:
            col = None
            col = Collection(args_dict['database'], args_dict['collection'])
            if col.exists(key_name='date'):
                result = col.get_datetime_list_from_data_collection(
                    start_date=None,
                    end_date=args_dict['end_date'])
            else:
                end_date = Collection.convert_string_to_datetime(
                    date_time_string=args_dict['end_date'])
                # 종료 날짜 변환에 실패했으면 None 처리.
                if end_date:
                    query = {}
                    query['end_date'] = end_date
                    result = col.object.distinct('start_date', query)
                else:
                    result = []
        finally:
            if col:
                col.close()

        query_result['start_date'] = result

    else:
        query_result = {'error': 'NotFound'}
        status_code = 204

    response_data = jsonify(query_result)
    return make_response(response_data, status_code)

@app.route('/count', methods=['GET'])
def query_data_count():
    """웹에서 설정한 값이 MongoDB에 유효한지 쿼리하여 체크.

    Returns:
        'database', 'collection' 만 있을 경우
        -> 쿼리 결과 해당 collection 목록이 있으면 1, 없으면 0
        'database', 'collection', 'start_date', 'end_date' 가 있을 경우
        -> 쿼리 개수 리턴
    """
    response_data = jsonify({'error': 'NotFound'})
    status_code = 200

    if isinstance(request.args, dict) is False:
        response_data = jsonify({'error': 'NotFound'})
        status_code = 400
        return make_response(response_data, status_code)

    args_dict = request.args.to_dict()
    query_result = {}
    result = None

    # database, collection 값일 경우
    if ((len(args_dict) == 2) and
            ('database' in args_dict) and ('collection' in args_dict)):
        try:
            db = None
            db = MongoDB(args_dict['database'])
            if args_dict['collection'] in db.get_collection_list():
                result = 1
            else:
                result = 0
        finally:
            if db:
                db.close()

        query_result['count'] = result

    # database, collection, start_date, end_date 값일 경우
    elif ((len(args_dict) == 4) and
          ('database' in args_dict) and ('collection' in args_dict) and
          ('start_date' in args_dict) and ('end_date' in args_dict)):
        try:
            col = None
            col = Collection(args_dict['database'], args_dict['collection'])
            if col.exists(key_name='date'):
                result = col.get_data_count_from_datetime(
                     start_date=args_dict['start_date'],
                     end_date=args_dict['end_date'])
            else:
                start_date = Collection.convert_string_to_datetime(
                    date_time_string=args_dict['start_date'])
                end_date = Collection.convert_string_to_datetime(
                    date_time_string=args_dict['end_date'])
                # 시작/종료 날짜 변환에 실패했으면 None 처리.
                if (start_date) and (end_date):
                    query = {}
                    if start_date:
                        query['start_date'] = start_date
                    if end_date:
                        query['end_date'] = end_date
                    result = col.object.count_documents(query)
                else:
                    result = 0
        finally:
            if col:
                col.close()

        query_result['count'] = result
    else:
        query_result = {'error': 'NotFound'}
        status_code = 204

    response_data = jsonify(query_result)
    return make_response(response_data, status_code)

@app.route('/attributes', methods=['GET'])
def attributes():
    """각 element 의 id 값 전달"""
    response_data = jsonify({'success': True})
    status_code = 200

    if isinstance(request.args, dict) is False:
        response_data = jsonify({'error': 'NotFound'})
        status_code = 400
        return make_response(response_data, status_code)

    args_dict = request.args.to_dict()
    if ('page' not in args_dict):
        response_data = jsonify({'error': 'NotFound'})
        status_code = 400
        return make_response(response_data, status_code)

    result = {}
    if (args_dict['page'] == PageName.INSERT_DATA):
        result['id'] = InsertDataId().attributes
        result['id'].update(InsertTableId().attributes)  # append

    elif (args_dict['page'] == PageName.TRAIN_PREDICT):
        id = TrainPredictId()
        result['id'] = id.attributes
        result['train_prefix'] = id.train_prefix
        result['predict_prefix'] = id.predict_prefix
        result['all_prefix'] = id.all_prefix

    if len(result) == 0:
        response_data = jsonify({'error': 'NotFound'})
        status_code = 204
    else:
        response_data = jsonify(result)
        status_code = 200

    return make_response(response_data, status_code)

@app.route('/settings', methods=['GET'])
def settings():
    """각 페이지에서 사용자가 마지막으로 설정한 element 값 전달"""
    response_data = jsonify({'success': True})
    status_code = 200

    if isinstance(request.args, dict) is False:
        response_data = jsonify({'error': 'NotFound'})
        status_code = 400
        return make_response(response_data, status_code)

    args_dict = request.args.to_dict()
    if ('page' not in args_dict):
        response_data = jsonify({'error': 'NotFound'})
        status_code = 400
        return make_response(response_data, status_code)

    result = {}
    try:
        col = None
        col = Collection('web', 'last_setting')
        result = col.object.find_one(
            {'_id': args_dict['page']},
            {'_id': 0})
    finally:
        if col:
            col.close()

    if result is None:
        response_data = jsonify({'error': 'NotFound'})
        status_code = 204
        return make_response(response_data, status_code)

    query_result = result
    response_data = jsonify(query_result)

    return make_response(response_data, status_code)

@app.route('/task', methods=['GET', 'POST'])
def task():
    """웹에서 설정한 task 들을 전달 받아 celery에 Server-Sent-Event로 전달"""
    response_data = jsonify({'success': True})
    status_code = 200

    if isinstance(request.args, dict) is False:
        response_data = jsonify({'error': 'NotFound'})
        status_code = 400
        return make_response(response_data, status_code)

    if (request.method == 'GET'):
        args_dict = request.args.to_dict()
    else:
        args_dict = request.get_json()

    if ('page_name' not in args_dict) or ('task_id' not in args_dict):
        response_data = jsonify({'error': 'NotFound'})
        status_code = 400
        return make_response(response_data, status_code)

    page_name = args_dict['page_name']
    task_id = args_dict['task_id']

    task_function = None
    if (page_name == PageName.INSERT_DATA):
        task_function = 'tasks.insertdata'
    elif (page_name == PageName.INSERT_TABLE):
        task_function = 'tasks.inserttable'
    elif (page_name == PageName.TRAIN_PREDICT):
        task_function = 'tasks.pipeline'

    if not task_function:
        response_data = jsonify({'error': 'NotFound'})
        status_code = 400
        return make_response(response_data, status_code)

    try:
        col = None
        col = Collection('web', 'last_setting')

        # 마지막 설정 값 저장.
        query = args_dict.copy()
        query['_id'] = page_name
        col.find_one_and_replace(
            {'_id': page_name}, query, upsert=True)
    finally:
        if col:
            col.close()

    # print(task_id)
    celery_object.send_task(
        task_function,
        args=[],
        kwargs=request.get_json(),
        task_id=task_id,
        ignore_result=True)

    return make_response(response_data, status_code)

@app.route('/test', methods=['GET'])
def tests():
    """Test Rest API"""
    status_code = 200
    if isinstance(request.args, dict) is False:
        response_data = jsonify({'error': 'NotFound'})
        status_code = 400
    else:
        print(request)
        response_data = jsonify({'success': True})

    return make_response(response_data, status_code)
