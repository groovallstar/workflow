import pytest
from web import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

@pytest.mark.skip()
def test_test(client):
    """test 용 rest api."""
    rv = client.get('/test')
    resp_json = rv.get_json()
    assert 'success' in resp_json

@pytest.mark.skip()
def test_lists_database(client):
    """Query database list"""
    rv = client.get('/lists')
    assert rv.status_code == 200

    data = rv.get_json()
    assert 'database' in data
    assert 'documents' in data['database']
    #print(data['database'])
    return data['database']

@pytest.mark.skip()
def test_lists_collection(client):
    """Query collection list"""
    from web.jinja.template_insert_data import InsertDataId, InsertTableId
    from web.jinja.template_train_predict import TrainPredictId

    database_list = test_lists_database(client)

    element_id_list = []
    element_id_list.append(InsertDataId())
    element_id_list.append(InsertTableId())
    element_id_list.append(TrainPredictId())

    collection_list = []

    test_uri = '/lists?database=' + database_list[0] + '&id='
    for attributes in element_id_list:
        for element_id in attributes.get_element_ids():
            append_id_url = test_uri + element_id
            rv = client.get(append_id_url)
            assert rv.status_code == 200
            data = rv.get_json()
            # 중복된 collection 제거.
            for x in data['collection']:
                if x not in collection_list:
                    collection_list.append(x)
            #print(resp_json)
            assert 'collection' in data

    #print(collection_list)
    return collection_list

@pytest.mark.skip()
def test_lists_date(client):
    """Query start/end date"""
    database_list = test_lists_database(client)
    collection_list = test_lists_collection(client)

    test_uri = '/lists?database=' + database_list[0]

    from dateutil.parser import parse
    for col_name in collection_list:
        append_collection_url = test_uri + "&collection=" + col_name
        rv = client.get(append_collection_url)
        assert rv.status_code == 200
        start_date_data = rv.get_json()
        assert 'start_date' in start_date_data
        #print(append_collection_url)
        #print(start_date_data['start_date'])
        assert len(start_date_data['start_date']) > 0

        for date in start_date_data['start_date']:
            start_date = parse(date).strftime('%Y%m')
            append_start_date_url=\
                append_collection_url + '&start_date=' + start_date
            rv = client.get(append_start_date_url)
            assert rv.status_code == 200
            data = rv.get_json()
            #print(data)
            assert 'end_date' in data

#@pytest.mark.skip()
def test_lists_and_count(client):
    """Query Database and Collection list"""
    rv = client.get("/count")
    assert rv.status_code == 204

    database_list = test_lists_database(client)
    collection_list = test_lists_collection(client)

    list_uri = '/lists?database=' + database_list[0]
    count_uri = '/count?database=' + database_list[0]

    from dateutil.parser import parse
    for col_name in collection_list:
        # database, collection 파라미터를 통해 실제 collection 이 있는지 체크.
        count_col_url = count_uri + '&collection=' + col_name
        rv = client.get(count_col_url)
        assert rv.status_code == 200
        count_data = rv.get_json()
        assert 'count' in count_data
        assert count_data['count'] == 1

        list_col_url = list_uri + '&collection=' + col_name
        rv = client.get(list_col_url)
        assert rv.status_code == 200
        start_date_data = rv.get_json()
        assert 'start_date' in start_date_data

        for start_date in start_date_data['start_date']:
            convert_start_date = parse(start_date).strftime('%Y%m')
            append_start_date_url=\
                list_col_url + '&start_date=' + convert_start_date

            rv = client.get(append_start_date_url)
            assert rv.status_code == 200
            end_date_data = rv.get_json()
            #print(data)
            assert 'end_date' in end_date_data

            for end_date in end_date_data['end_date']:
                convert_end_date = parse(end_date).strftime('%Y%m')
                append_count_url = count_col_url + \
                    '&start_date=' + convert_start_date + \
                    '&end_date=' + convert_end_date
                rv = client.get(append_count_url)
                #print(append_count_url)
                assert rv.status_code == 200

#@pytest.mark.skip()
def test_contents(client):
    """Query Contents"""
    from web.jinja.elements import PageName
    rv = client.get("/contents")
    assert rv.status_code == 400

    page_name_list = [
        PageName.INSERT_DATA,
        PageName.TRAIN_PREDICT
    ]

    for page_name in page_name_list:
        test_uri = "/contents?" + 'page=' + page_name
        rv = client.get(test_uri)
        assert rv.status_code == 200
        data = rv.get_json()
        assert 'html' in data

#@pytest.mark.skip()
def test_attributes(client):
    """Query Attributes"""
    from web.jinja.elements import PageName

    rv = client.get("/attributes")
    assert rv.status_code == 400

    page_name_list = [
        PageName.INSERT_DATA, 
        PageName.TRAIN_PREDICT
    ]

    for page_name in page_name_list:
        test_uri = "/attributes?" + 'page=' + page_name
        rv = client.get(test_uri)
        assert rv.status_code == 200
        data = rv.get_json()
        assert 'id' in data

#@pytest.mark.skip()
def test_settings(client):
    """Query Settings"""
    from web.jinja.elements import PageName

    rv = client.get("/settings")
    assert rv.status_code == 400

    page_name_list = [
        PageName.INSERT_DATA,
        PageName.TRAIN_PREDICT
    ]

    for page_name in page_name_list:
        test_uri = "/settings?" + 'page=' + page_name
        rv = client.get(test_uri)
        assert rv.status_code == 200
