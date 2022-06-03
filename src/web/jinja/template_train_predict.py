from web.jinja.elements import PageName, Input, Card, CheckBox, Select, Row

class TrainPredictId:
    """Train/Predict Component Id Attributes."""
    train_prefix = 't-'
    predict_prefix = 'p-'
    all_prefix = 'a-'

    def __init__(self):
        """Init"""
        self._attributes = {}
        self._attributes['experiment'] = \
            TrainPredictId.all_prefix+'experiment'
        self._attributes['run_name'] = \
            TrainPredictId.all_prefix+'run-name'
        self._attributes['comments'] = \
            TrainPredictId.all_prefix+'comments'

        self._attributes['data'] = dict()
        self._attributes['data']['database'] = \
            TrainPredictId.all_prefix + 'data-database'
        self._attributes['data']['collection'] = \
            TrainPredictId.all_prefix + 'data-collection'
        self._attributes['data']['start_date'] = \
            TrainPredictId.all_prefix + 'data-startdate'
        self._attributes['data']['end_date'] = \
            TrainPredictId.all_prefix+'data-enddate'

        self._attributes['table'] = {}
        self._attributes['table']['database'] = \
            TrainPredictId.all_prefix + 'table-database'
        self._attributes['table']['collection'] = \
            TrainPredictId.all_prefix + 'table-collection'
        self._attributes['table']['start_date'] = \
            TrainPredictId.all_prefix + 'table-startdate'
        self._attributes['table']['end_date'] = \
            TrainPredictId.all_prefix + 'table-enddate'

        self._attributes['classification_file_name'] = \
            TrainPredictId.all_prefix + 'classification-file-name'

        self._attributes['sampling'] = \
            TrainPredictId.all_prefix+'sampling'
        self._attributes['seed'] = \
            TrainPredictId.all_prefix+'seed'

        self._attributes['load_model'] = {}
        self._attributes['load_model']['database'] = \
            TrainPredictId.predict_prefix + 'load-model-database'
        self._attributes['load_model']['collection'] = \
            TrainPredictId.predict_prefix + 'load-model-collection'
        self._attributes['load_model']['start_date'] = \
            TrainPredictId.predict_prefix + 'load-model-startdate'
        self._attributes['load_model']['end_date'] = \
            TrainPredictId.predict_prefix + 'load-model-enddate'

        self._attributes['save_model'] = {}
        self._attributes['save_model_checkbox'] = \
            TrainPredictId.train_prefix + 'save-model'
        self._attributes['save_model']['database'] = \
            TrainPredictId.train_prefix + 'save-model-database'
        self._attributes['save_model']['collection'] = \
            TrainPredictId.train_prefix + 'save-model-collection'
        self._attributes['save_model']['path'] = \
            TrainPredictId.train_prefix + 'save-model-path'

        self._attributes['show_data'] = \
            TrainPredictId.all_prefix+'show-data'
        self._attributes['train'] = \
            TrainPredictId.train_prefix+'train'

        self._attributes['evaluate'] = \
            TrainPredictId.all_prefix+'evaluate'
        self._attributes['show_metric_by_thresholds'] = \
            TrainPredictId.all_prefix + 'show-metric-by-thresholds'
        self._attributes['thresholds'] = \
            TrainPredictId.all_prefix+'thresholds'
        self._attributes['show_optimal_metric'] = \
            TrainPredictId.all_prefix+'show_optimal_metric'
        self._attributes['find_best_model'] = \
            TrainPredictId.all_prefix+'find_best_model'

        self._attributes['grid_search'] = \
            TrainPredictId.train_prefix+'gridsearch'
        self._attributes['bayesian_optimizer'] = \
            TrainPredictId.train_prefix + 'bayesian-optimizer'

        self._attributes['split_ratio'] = {}
        self._attributes['split_ratio']['train'] = \
            TrainPredictId.train_prefix + 'split-ratio-train'
        self._attributes['split_ratio']['validation'] = \
            TrainPredictId.train_prefix + 'split-ratio-validation'
        self._attributes['split_ratio']['test'] = \
            TrainPredictId.all_prefix + 'split-ratio-test'

    def get_collection_element_ids(self):
        """Generator Collection Element Id"""
        yield self._attributes['data']['collection']
        yield self._attributes['table']['collection']
        yield self._attributes['load_model']['collection']
        yield self._attributes['save_model']['collection']

    @property
    def attributes(self): return self._attributes
    @property
    def experiment(self): return self._attributes['experiment']
    @property
    def run_name(self): return self._attributes['run_name']
    @property
    def comments(self): return self._attributes['comments']
    @property
    def data_database(self): return self._attributes['data']['database']
    @property
    def data_collection(self): return self._attributes['data']['collection']
    @property
    def data_startdate(self): return self._attributes['data']['start_date']
    @property
    def data_enddate(self): return self._attributes['data']['end_date']
    @property
    def table_database(self): return self._attributes['table']['database']
    @property
    def table_collection(self): return self._attributes['table']['collection']
    @property
    def table_startdate(self): return self._attributes['table']['start_date']
    @property
    def table_enddate(self): return self._attributes['table']['end_date']
    @property
    def classification_file(self):
        return self._attributes['classification_file_name']
    @property
    def sampling(self): return self._attributes['sampling']
    @property
    def seed(self): return self._attributes['seed']
    @property
    def load_model_database(self):
        return self._attributes['load_model']['database']
    @property
    def load_model_collection(self):
        return self._attributes['load_model']['collection']
    @property
    def load_model_startdate(self):
        return self._attributes['load_model']['start_date']
    @property
    def load_model_enddate(self):
        return self._attributes['load_model']['end_date']
    @property
    def show_data(self): return self._attributes['show_data']
    @property
    def save_model_checkbox(self):
        return self._attributes['save_model_checkbox']
    @property
    def save_model_database(self):
        return self._attributes['save_model']['database']
    @property
    def save_model_collection(self):
        return self._attributes['save_model']['collection']
    @property
    def train(self): return self._attributes['train']
    @property
    def evaluate(self): return self._attributes['evaluate']
    @property
    def show_metric_thresholds(self):
        return self._attributes['show_metric_by_thresholds']
    @property
    def thresholds(self): return self._attributes['thresholds']
    @property
    def show_optimal_metric(self):
        return self._attributes['show_optimal_metric']
    @property
    def find_best_model(self): return self._attributes['find_best_model']
    @property
    def gridsearch(self): return self._attributes['grid_search']
    @property
    def bayesian_optimizer(self): return self._attributes['bayesian_optimizer']
    @property
    def split_train(self): return self._attributes['split_ratio']['train']
    @property
    def split_validation(self):
        return self._attributes['split_ratio']['validation']
    @property
    def split_test(self): return self._attributes['split_ratio']['test']

class TrainPredictElementList(TrainPredictId):
    """학습/예측 페이지에 사용되는 HTML Element 요소.

    Args:
        TrainPredictId (Class): 각 element의 HTML id.
    """
    def __init__(self):
        """Init"""
        super().__init__()
        self._card_list = []

        mltype_column_list = [
            Select(id="ml-type", label_text="Select Machine Learning Type",
                   floating_message='Select Machine Learning Type',
                   error_message='',
                   select_list=[
                       dict(text='Train', selected=False),
                       dict(text='Predict', selected=False),],
                   selected_text='Select Type')
        ]

        experiment_column_list = [
            Select(id=self.experiment, label_text="Select Experiment",
                   floating_message='Select Experiment Name',
                   error_message='',
                   select_list=[
                       dict(text='iris', selected=False),
                       dict(text='digit', selected=False),
                       dict(text='wine', selected=False),
                       dict(text='breast_cancer', selected=False),]),
            Input(label_text='Run Name',
                  floating_message='Run Name',
                  error_message='',
                  input_type='text',
                  input_id=self.run_name, max_length='128',
                  placeholder=' '),
        ]

        data_column_list = [
            Select(
                id=self.data_database,
                label_text='Database',
                floating_message='Database',
                error_message='유효하지 않은 값입니다.',
                select_name='select-dbinfo'),
            Select(
                id=self.data_collection,
                label_text='Collection',
                floating_message='Collection',
                error_message='유효하지 않은 값입니다.',
                select_name='select-dbinfo'),
            Select(
                id=self.data_startdate,
                label_text='StartDate',
                floating_message='StartDate',
                error_message='유효하지 않은 값입니다.',
                select_name='select-dbinfo'),
            Select(
                id=self.data_enddate,
                label_text='EndDate',
                floating_message='EndDate',
                error_message='유효하지 않은 값입니다.',
                select_name='select-dbinfo'),
        ]

        table_column_list = [
            Select(
                id=self.table_database,
                label_text='Database',
                floating_message='Database',
                error_message='유효하지 않은 값입니다.',
                select_name='select-dbinfo'),
            Select(
                id=self.table_collection,
                label_text='Collection',
                floating_message='Collection',
                error_message='유효하지 않은 값입니다.',
                select_name='select-dbinfo'),
            Select(
                id=self.table_startdate,
                label_text='StartDate',
                floating_message='StartDate',
                error_message='유효하지 않은 값입니다.',
                select_name='select-dbinfo'),
            Select(
                id=self.table_enddate,
                label_text='EndDate',
                floating_message='EndDate',
                error_message='유효하지 않은 값입니다.',
                select_name='select-dbinfo'),
        ]

        classification_file_column_list = [
            Select(
                id=self.classification_file,
                label_text="Select Classification File Name",
                floating_message='Select Classification File Name',
                error_message='',
                select_list=[
                    dict(text='iris.yaml', selected=False),
                    dict(text='digit.yaml', selected=False),
                    dict(text='wine.yaml', selected=False),
                    dict(text='breast_cancer.yaml', selected=False)])
        ]

        etc_column_list = [
            Input(
                label_text='Sampling',
                floating_message='Sampling',
                error_message='값을 넘을수 없습니다.',
                input_type='number', input_id=self.sampling,
                input_name='input-etc',
                min='0.01', max='1.0', step='0.01', placeholder='number..',
                invalid_feedback=True),
            Input(
                label_text='Seed',
                floating_message='Seed',
                error_message='값을 넘을수 없습니다.',
                input_type='number', input_id=self.seed,
                input_name='input-etc',
                min='0', max='4294967295', step='1', placeholder='number..',
                invalid_feedback=True),
        ]

        load_model_column_list = [
            Select(
                id=self.load_model_database,
                label_text='Database',
                floating_message='Database',
                error_message='유효하지 않은 값입니다.',
                select_name='select-dbinfo'),
            Select(
                id=self.load_model_collection,
                label_text='Collection',
                floating_message='Collection',
                error_message='유효하지 않은 값입니다.',
                select_name='select-dbinfo'),
            Select(
                id=self.load_model_startdate,
                label_text='StartDate',
                floating_message='StartDate',
                error_message='유효하지 않은 값입니다.',
                select_name='select-dbinfo'),
            Select(
                id=self.load_model_enddate,
                label_text='EndDate',
                floating_message='EndDate',
                error_message='유효하지 않은 값입니다.',
                select_name='select-dbinfo'),
        ]

        show_data_column_list = [
            CheckBox(
                id=self.show_data,
                text='Show Data', checked=False)
        ]

        save_model_checkbox_column_list = [
            CheckBox(
                id=self.save_model_checkbox,
                text='Save Model',
                checked=False)
        ]

        save_model_column_list = [
            Select(
                id=self.save_model_database,
                label_text='Database',
                floating_message='Database',
                error_message='유효하지 않은 값입니다.',
                select_name='select-dbinfo'),
            Select(
                id=self.save_model_collection,
                label_text='Collection',
                floating_message='Collection',
                error_message='유효하지 않은 값입니다.',
                select_name='select-dbinfo'),
        ]

        train_column_list = [
            CheckBox(id=self.train, text='Train', checked=False)
        ]

        evaluate_column_list = [
            CheckBox(id=self.evaluate, text='Evaluate', checked=False),
            CheckBox(
                id=self.show_optimal_metric, text='Show Optimal Metric',
                checked=False)
        ]

        threshold_column_list = [
            CheckBox(
                id=self.thresholds, text='Show Metric By Thresholds',
                checked=False)
        ]

        show_metric_by_thresholds_column_list = [
            Input(
                label_text='Threshold', input_type='text',
                floating_message='Threshold',
                error_message='쉼표 구분자와 0과 1사이의 숫자로만 입력하세요.',
                input_id=self.show_metric_thresholds,
                max_length='64', placeholder=' ')
        ]

        find_best_model_list = [
            CheckBox(
                id=self.find_best_model,
                text='Find Best Model',
                checked=False)
        ]

        hyper_parameter_column_list = [
            CheckBox(id=self.gridsearch, text='GridSearch', checked=False),
            CheckBox(
                id=self.bayesian_optimizer, text='Bayesian Optimizer',
                checked=False)
        ]

        split_column_list = [
            Input(
                label_text='Train', input_type='number',
                floating_message='Train',
                error_message='전체 합이 1.0 이 되어야 합니다.',
                input_name='input-data-split',
                input_id=self.split_train,
                min='0.01', max='1.0', step='0.01', placeholder='number..'),
            Input(
                label_text='Validation', input_type='number',
                floating_message='Validation',
                error_message='전체 합이 1.0 이 되어야 합니다.',
                input_name='input-data-split',
                input_id=self.split_validation,
                min='0.01', max='1.0', step='0.01', placeholder='number..'),
            Input(
                label_text='Test', input_type='number',
                floating_message='Test',
                error_message='전체 합이 1.0 이 되어야 합니다.',
                input_name='input-data-split',
                input_id=self.split_test,
                min='0.01', max='1.0', step='0.01', placeholder='number..'),
        ]

        self._card_list.append(
            Card(header_small='Machine Learning Type',
                 row_list=[
                     Row(column_list=mltype_column_list,
                         column_class="col-md-6")])
        )

        self._card_list.append(
            Card(header_small='MLflow Information',
                 row_list=[Row(column_list=experiment_column_list)])
        )

        self._card_list.append(
            Card(header_small='DataSet Database Setting',
                 row_list=[Row(column_list=data_column_list)])
        )

        self._card_list.append(
            Card(header_small='Table Database Setting',
                 row_list=[Row(column_list=table_column_list)])
        )

        self._card_list.append(
            Card(header_small='Data Split Information',
                 row_list=[Row(label_text='Data Split Information',
                               column_list=split_column_list)])
        )

        self._card_list.append(
            Card(header_small='Classification File Name',
                 row_list=[Row(column_list=classification_file_column_list,
                               column_class="col-md-6")])
        )

        self._card_list.append(
            Card(header_small='Etc',
                 row_list=[Row(column_list=etc_column_list)])
        )

        self._card_list.append(
            Card(header_small='Load Model Information',
                 row_list=[Row(column_list=load_model_column_list)])
        )

        self._card_list.append(
            Card(row_list=[Row(column_list=show_data_column_list)])
        )

        self._card_list.append(
            Card(header_small='Train',
                 row_list=[Row(column_list=train_column_list, hr=True),
                           Row(column_list=save_model_checkbox_column_list,
                               p=True),
                           Row(column_list=save_model_column_list)])
        )

        self._card_list.append(
            Card(header_small='Predict',
                 row_list=[Row(column_list=evaluate_column_list, hr=True),
                           Row(column_list=threshold_column_list, p=True),
                           Row(column_list=\
                               show_metric_by_thresholds_column_list, hr=True),
                           Row(column_list=find_best_model_list)])
        )

        self._card_list.append(
            Card(header_small='Find Hyper Parameters',
                 row_list=[Row(column_list=hyper_parameter_column_list)])
        )

    @property
    def card_list(self):
        """get card list"""
        return self._card_list

class TrainPredictPage():
    """학습/예측 페이지."""
    def __init__(self):
        """Init"""
        self._card_list = []

        element_list = TrainPredictElementList()
        self._card_list.append(
            Card(card_id=PageName.TRAIN_PREDICT.value,
                 header_strong='Train / Predict',
                 header_small='',
                 footer_btn_name='ADD',
                 add_btn_param_name=PageName.TRAIN_PREDICT.value,
                 row_list=element_list.card_list))

    @property
    def card_list(self):
        """get card list"""
        return self._card_list
