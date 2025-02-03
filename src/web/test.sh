#!/bin/sh

pytest tests/test_rest_api.py -s; exit_code=$?
exit $exit_code
