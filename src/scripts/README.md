## Scripts ##

- insert_data.sh
    - 실행 Parameter
        - --dataset : 'iris', 'digits', 'wine', 'breast_cancer'
        - --database :  database 명 
        - --collection : collection 명 
        - --date : 데이터 저장 날짜 (YYYYMM 문자열 형식)

- insert_table.sh
    - 실행 Parameter 
        - --data={'database': '데이터가 저장된 database 명',  
                  'collection': '데이터가 저장된 collection 명',  
                  'start_date': '시작날짜', 'end_date': '종료날짜'}
        - --table={'database': 'column을 저장할 database 명',  
                   'collection': 'column을 저장할 collection 명'}

- src/scripts/train.sh, src/scripts/predict.sh
    - 실행 Parameter
        - --data={'database': 'database명', 'collection': 'collection명',  
                'start_date': '시작날짜', 'end_date': '종료날짜'}

        - --table={'database': 'database명', 'collection': 'collection명',  
                'start_date': '시작날짜', 'end_date': '종료날짜'}

        - --experiment=실험 결과가 기록될 MLflow Experiment 명
        - --run_name=MLflow Run 이름

        - --show_data=학습/검증/테스트 데이터에 대한 비율 출력
        - --seed=데이터 분할 및 모델 학습에 사용할 Random Seed 값
        - --classification_file_name=dataset에서 사용할 모델 및 Hyper Parameter 정보 파일명
        - --train_with_tuning=파라미터 튜닝 후 학습
        - 각 알고리즘의 하이퍼 파라미터 갱신 후 학습
        - '--classification_file_name'의 파일명에 hyper_parameter 갱신
        - --train=학습
        - --load_model={'database': 'database명', 'collection': 'collection명', 
                        'start_date': '시작날짜', 'end_date': '종료날짜'}
            - 'database', 'collection' 값을 통해 database 로부터 모델 로드
        
        - --evaluate=예측 실행
        - --show_metric_by_thresholds=예측 결과의 평가 메트릭을 특정 임계값 리스트를 통해 출력
            - 평가 메트릭 : Accuracy, Precision, Recall, F1-Score, ROC-AUC
        - --show_optimal_metric=각 모델의 F1-Score가 가장 높은 임계값을 출력
        - --find_best_model=모델 중에서 F1-Score가 가장 높은 모델 정보 출력
        - --save_model={'database': 'database명', 'collection': 'collection명'}
