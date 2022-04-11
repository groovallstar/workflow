#!/bin/bash

cd build_and_test
docker-compose -f docker-compose.yml --env-file ../.env build
