#!/bin/bash

cd deploy
docker-compose down --remove-orphans
docker-compose -f docker-compose.yml --env-file ../.env up -d
