import pandas as pd

from common.function import timeit
from common.container.mongo import Collection
from common.trace_log import TraceLog, get_log_file_name

import ray
@ray.remote
def procedure(
    database: str, collection: str, item: dict, date_time: str) -> None:
    """procedure

    Args:
        database (str): database 명
        collection (str): collection 명
        item (dict): insert 할 dict 데이터
        date_time (str): insert 할 날짜 (str)
                         e.g. YYYYMM

    Raises:
        ValueError: 날짜 문자열이 잘못될 경우
        ValueError: 중복된 document인 경우
    """
    try:
        if ((isinstance(database, str) is False) or
            (isinstance(collection, str) is False) or
            (isinstance(item, dict) is False) or
            (isinstance(date_time, str) is False)):
            raise ValueError("Invalid Parameter.")

        collection = Collection(database, collection)
        date = Collection.convert_string_to_datetime(date_time)
        if not date:
            raise ValueError("convert_string_to_datetime failed."
                            f" date={date_time}")
        item['date'] = date
        result = collection.find_one_and_replace(item, item, upsert=True)
        if result:
            raise ValueError(f"DocumentAlready Exist. data={item}")

    except BaseException as ex:
        TraceLog.info(ex)
        return

# VSCODE launch.json parameter example.
# "args" : [
#     "--dataset=iris",
#     "--database=database", "--collection=collection", "--date=202201"
# ]

def parse_commandline() -> dict:
    """"commandline parsing"""
    import argparse
    import textwrap

    description = textwrap.dedent("""
    ===========================================================================
    --dataset : sklearn Datasets (str)
                e.g. iris, digits, wine, breast_cancer
    --database : Database Name
    --collection : Collection Name
    --date : DateTime String (YYYYMM) 
             e.g. 202001
    ===========================================================================
    """)

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '--dataset', type=str, required=True, help='Sklearn Datasets.')
    parser.add_argument(
        '--database', type=str, required=True, help='Database Name.')
    parser.add_argument(
        '--collection', type=str, required=True, help="Collection Name.")
    parser.add_argument(
        '--date', type=str, required=True, help='Date.')
    
    # argparse.Namespace to dict
    args, _ = parser.parse_known_args()
    return vars(args)

@timeit
def insert_data(parameters: dict) -> None:
    """Insert Data

    Args:
        parameters (dicx): commandline dictionary

    Raises:
        ValueError: dataset 파라미터가 'iris' or 'digits' or
                    'wine' or 'breast_cancer' 이 아닐 경우
        BufferError: 데이터 로드 실패할 경우
    """
    from sklearn.utils import Bunch
    load_data = None
    if parameters['dataset'] == 'iris':
        from sklearn.datasets import load_iris
        load_data = load_iris()
    elif parameters['dataset'] == 'digits':
        from sklearn.datasets import load_digits
        load_data = load_digits()
    elif parameters['dataset'] == 'wine':
        from sklearn.datasets import load_wine
        load_data = load_wine()
    elif parameters['dataset'] == 'breast_cancer':
        from sklearn.datasets import load_breast_cancer
        load_data = load_breast_cancer()
    else:
        raise ValueError("dataset string 'iris' or 'digits'"
                          " or 'wine' or 'breast_cancer' ")

    if (not load_data) or (isinstance(load_data, Bunch) is False):
        raise BufferError('load data failed.')
    
    df = pd.DataFrame(data=load_data.data, columns=load_data['feature_names'])
    df['label'] = load_data.target
    convert_dict_items = df.to_dict('records')

    # """Default Behavior."""
    # collection = Collection(parameters['database'], parameters['collection'])
    # collection.insert_document_many(convert_dict_items)

    # """For Debugging."""
    # for item in convert_dict_items:
    #     procedure(
    #         parameters['database'],
    #         parameters['collection'],
    #         item,
    #         parameters['date'])

    ray.init()
    ray.get(
        [procedure.remote(
            parameters['database'],
            parameters['collection'],
            item,
            parameters['date']) for item in convert_dict_items])

    return

def main():
    """main"""
    try:
        TraceLog().initialize(log_file_name=get_log_file_name(__file__))
        TraceLog().info('= Start =')
        parameters = parse_commandline()
        if isinstance(parameters, dict) is False:
            raise ValueError('Parameter Type Must Be Dict.')
        TraceLog().info(f"parameters: {parameters}")
        insert_data(parameters)

    except BaseException as e:
        TraceLog().info(f"Error occurred. Message: {e}")

    finally:
        TraceLog().info('=  End =')

if __name__ == '__main__':
    main()   
