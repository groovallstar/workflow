# Apply utf-8 to MLflow Database Columns.

# sqlalchemy.exc.DataError: (pymysql.err.DataError) 
# (1366, "Incorrect string value: '\\xE3\\x85\\x87\\xE3\\x85\\x87...' for column 'value' at row 1")
# [SQL: UPDATE tags SET value=%(value)s WHERE tags.`key` = %(tags_key)s AND tags.run_uuid = %(tags_run_uuid)s]
# [parameters: {'value': 'ㅇㅇㅇ', 'tags_key': 'mlflow.note.content', 'tags_run_uuid': 'd0efa63c94c941c9a1f781bef0b24484'}]
# (Background on this error at: https://sqlalche.me/e/14/9h9h)
# -----------------------------------------------------------------------------
# ALTER TABLE registered_models MODIFY COLUMN description VARCHAR(255) CHARACTER SET utf8 COLLATE utf8_general_ci;
# ALTER TABLE tags MODIFY COLUMN value VARCHAR(5000) CHARACTER SET utf8 COLLATE utf8_general_ci;
# ALTER TABLE model_versions MODIFY COLUMN description VARCHAR(255) CHARACTER SET utf8 COLLATE utf8_general_ci;
# -----------------------------------------------------------------------------

docker compose -f docker-compose.yml up -d
