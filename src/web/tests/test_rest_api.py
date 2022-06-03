from fastapi.testclient import TestClient
import pytest

from web.main import app
from web.jinja.elements import PageName

client = TestClient(app)

@pytest.mark.skip()
def test_base_page():
    """Query Base Page"""
    response = client.get('/')
    assert response.status_code == 200

@pytest.mark.skip()
def test_attributes():
    """Query Attributes"""
    for name in PageName:
        test_uri = '/attributes?' + 'page=' + name.value
        response = client.get(test_uri)
        assert response.status_code == 200
        data = response.json()
        assert 'id' in data

@pytest.mark.skip()
def test_contents():
    """Query Contents"""
    for name in PageName:
        test_uri = '/contents?' + 'page=' + name.value
        response = client.get(test_uri)
        assert response.status_code == 200

@pytest.mark.skip()
def test_settings():
    """Query Settings"""
    for name in PageName:
        test_uri = '/settings?' + 'page=' + name.value
        response = client.get(test_uri)
        assert (response.status_code == 200 or response.status_code == 204)
        if response.status_code == 200:
            assert response.json()

@pytest.mark.skip()
def test_select_list_for_database():
    """Query database list"""
    response = client.get('/sel_list_db')
    assert response.status_code == 200
    assert response.json()
    assert 'documents' in response.json()
    return response.json()[0]

@pytest.mark.skip()
def test_select_list_for_collection():
    """Query collection list"""
    from web.jinja.template_insert_data import InsertDataId
    from web.jinja.template_insert_table import InsertTableId
    from web.jinja.template_train_predict import TrainPredictId
    from web.jinja.template_feature_selection import FeatureSelectionId

    documents = test_select_list_for_database()
    assert 'documents' == documents

    element_id_list = []
    element_id_list.append(InsertDataId())
    element_id_list.append(InsertTableId())
    element_id_list.append(TrainPredictId())
    element_id_list.append(FeatureSelectionId())

    collection_list = []

    test_uri = '/sel_list_col?database=' + documents + '&element_id='
    for attributes in element_id_list:
        for element_id in attributes.get_collection_element_ids():
            append_id_url = test_uri + element_id
            response = client.get(append_id_url)
            assert response.status_code == 200
            assert response.json()

            data = response.json()
            assert isinstance(data, list)

            # 중복된 collection 제거.
            for x in data:
                if x not in collection_list:
                    collection_list.append(x)
    #print(collection_list)
    return collection_list

@pytest.mark.skip()
def test_lists_date():
    """Query start/end date"""
    documents = test_select_list_for_database()
    collection_list = test_select_list_for_collection()

    test_uri = '/sel_list_date?database=' + documents

    from dateutil.parser import parse
    for col_name in collection_list:
        append_collection_url = test_uri + "&collection=" + col_name
        #print(append_collection_url)
        response = client.get(append_collection_url)
        assert (response.status_code == 200 or response.status_code == 204)
        if response.status_code == 200:
            assert 'start_date' in response.json()
            start_date_list = response.json()['start_date']
            assert len(start_date_list) > 0

            for date in start_date_list:
                start_date = parse(date).strftime('%Y%m')
                append_start_date_url=\
                    append_collection_url + '&start_date=' + start_date
                response = client.get(append_start_date_url)
                assert response.status_code == 200
                assert 'end_date' in response.json()

#@pytest.mark.skip()
def test_count():
    """Query Document Cound"""
    documents = test_select_list_for_database()
    collection_list = test_select_list_for_collection()

    date_url = '/sel_list_date?database=' + documents
    count_url = '/count?database=' + documents

    from dateutil.parser import parse
    for col_name in collection_list:
        count_col_url = count_url + '&collection=' + col_name
        response = client.get(count_col_url)
        assert response.status_code == 200

        list_col_url = date_url + '&collection=' + col_name
        response = client.get(list_col_url)
        assert (response.status_code == 200 or response.status_code == 204)
        if response.status_code == 200:
            assert 'start_date' in response.json()
            start_date_list = response.json()['start_date']
            assert len(start_date_list) > 0

            for start_date in start_date_list:
                convert_start_date = parse(start_date).strftime('%Y%m')
                append_start_date_url=\
                    list_col_url + '&start_date=' + convert_start_date
                response = client.get(append_start_date_url)
                assert response.status_code == 200
                assert 'end_date' in response.json()
                end_date_list = response.json()['end_date']

                for end_date in end_date_list:
                    convert_end_date = parse(end_date).strftime('%Y%m')
                    count_end_date_url = count_col_url + \
                        '&start_date=' + convert_start_date + \
                        '&end_date=' + convert_end_date
                    response = client.get(count_end_date_url)
                    assert response.status_code == 200
                    #print(count_end_date_url)
