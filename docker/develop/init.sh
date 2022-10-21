#!/bin/bash

docker compose -f docker-compose.yml --env-file .env build --no-cache
docker compose -f docker-compose.yml --env-file .env up -d
