Workflow
==============================

## Folder Structure ##

- docker/
  - docker 설정

- src/celery/
  - Python Celery 

- src/common/
  - 공통으로 사용하는 method 집합
  - 공통으로 데이터를 로드하기 위한 클래스

- src/learning/
  - 학습/예측/파이프라인

- src/make_data/
  - 데이터 저장 및 가공

- src/scripts/
  - insert_data.sh: sklearn Toy dataset을 Database에 추가
  - insert_table.sh: sklearn Toy dataset의 column 정보를 Database에 추가
  - train.sh, predict.sh : 학습/예측

- src/tests/
  - Unit Test 코드

- src/web/
  - WorkFlow를 실행하기 위한 웹 관리 툴
