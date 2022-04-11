#!/bin/bash
docker build -t setup-rspl .
sleep 1
docker-compose up -d