from common.container.mongo import Collection
from common.function import timeit
from common.trace_log import TraceLog, get_log_file_name

import inspect
from common.function import get_code_line

def insert_table(data, table):
    """data 컬렉션에서 column 정보 저장.

    Args:
        data (dict) : 'database', 'collection', 'start_date', 'end_date'
        table (dict) : 'database', 'collection'
    """

    if ((('database' not in data) or ('collection' not in data)) or
        (('database' not in table) or ('collection' not in table))):
        raise ValueError("Parameters not in 'database' or 'collection' key.",
                         get_code_line(inspect.currentframe()))

    # data collection에서 필드명 쿼리
    column_list = Collection.get_field_list_from_data_dict(data)
    if len(column_list) <= 0:
        raise ValueError('Column List Empty.',
                         get_code_line(inspect.currentframe()))

    start_date = data['start_date'] if 'start_date' in data else None
    end_date = data['end_date'] if 'end_date' in data else None

    query = {}
    data_collection = Collection(data['database'], data['collection'])
    query['start_date'] = data_collection.query_start_date(start_date)
    query['end_date'] = data_collection.query_end_date(end_date)

    table_collection = Collection(table['database'], table['collection'])
    table_collection.object.find_one_and_update(
        query,
        {'$set': {'columns': column_list}},
        upsert=True)

    return

# VSCODE launch.json parameter example.
# "args" : [
#         "--data={\"database\": \"test\", \"collection\": \"iris.data\",
#                  \"start_date\": \"202201\", \"end_date\": \"202201\"}",
#         "--table={\"database\": \"test\", \"collection\": \"iris.table\"}",
# ]

def parse_commandline() -> dict:
    """"commandline parsing"""
    import argparse
    import json
    import textwrap

    description = textwrap.dedent("""
    ===========================================================================
    --data Example
    Format : Dictionary Format
    Key : "database", "collection", "start_date", "end_date"
          "database" : Database Name
          "collection" : Collection Name
          "start_date" : Query Start Date(String)
          "end_date" : Query End Date(String)
    e.g. --data='{"database": "database", "collection": "collection",
                  "start_date": "202201", "end_date": "202202"}'
                  => Query 202201 ~ 202202 Column List.

    --table Example
    Format : Dictionary Format
    Key : "database", "collection"
          "database" : Database Name
          "collection" : Collection Name
    e.g. --table='{"database":"database", "collection":"collection"}' 
         => Save Data's Column List and Start/End Date.
    ===========================================================================
    """)

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '--data', type=json.loads, required=True, help='Data Dictionary.')
    parser.add_argument(
        '--table', type=json.loads, required=True, help='Table Dictionary.')

    # argparse.Namespace to dict
    args, _ = parser.parse_known_args()
    return vars(args)

@timeit
def main():
    """main"""
    try:
        TraceLog().initialize(log_file_name=get_log_file_name(__file__))
        TraceLog().info('= Start =')
        args = parse_commandline()
        TraceLog().info(f"args: {args}")

        insert_table(data=args['data'], table=args['table'])

    except BaseException as e:
        TraceLog().info(f"Error occurred. Message: {e}")

    finally:
        TraceLog().info('= End =')

if __name__ == '__main__':
    main()
