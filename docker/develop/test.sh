#!/bin/bash

cd build_and_test
docker-compose -f docker-compose.yml --env-file ../.env up

docker rm sklearn_build_and_test
docker rm flask_build_and_test
