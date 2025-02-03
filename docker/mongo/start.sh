#!/bin/bash
docker compose up -d

docker cp select_list.json mongodb_dev:/root
docker exec mongodb_dev mongoimport --db web --collection select_list --type json --file /root/select_list.json --jsonArray -u root -p root --authenticationDatabase=admin
