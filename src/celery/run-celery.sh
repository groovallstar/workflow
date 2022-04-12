#!/bin/sh

celery -A tasks worker --loglevel=info -E -O fair --concurrency=1
