import os
import inspect
from typing import Iterator, List, Tuple, Any, Union

from common.data_type import MetricData, MetricScoreInfo, BestModelScoreInfo
from common.function import _validate, get_code_line
from common.trace_log import TraceLog, init_log_object, get_log_file_name

from learning.classifier import Classifier
from learning.model import Model

def _get_dicts(indict, prefix=None) -> Iterator[List[str]]:
    """nested dictionary 구조에서 [key, subkey, ..., value] 값 리턴"""
    prefix = prefix if prefix else []
    if not indict:
        return None
    elif isinstance(indict, dict):
        for key, value in indict.items():
            if isinstance(value, dict):
                for v in _get_dicts(value, prefix + [key]):
                    yield v
            else:
                yield prefix + [key, value]
    else:
        yield prefix + [indict]

# VSCODE launch.json parameter example.
# "args" : [
#   "--experiment=test", "--run_name=test",
#   "--_data={\"database\" : \"test\", \"collection\" : \"iris.data\",
#                  \"start_date\" : \"202201\", \"end\" : \"202201\"}",
#   "--table={\"database\":\"test\",\"collection\":\"table\",
#             \"start_date\" : \"202201\",\"end\" : \"202201\"}",
#   "--seed=123", "--show_data", "--classification_file_name=iris.yaml",
#   "--show_metric_by_thresholds=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
#   "--train", "--evaluate", "--show_optimal_metric",
#   "--load_model={\"database\" : \"test\", \"collection\" : \"model\",
#                  \"start_date\" : \"202201\", \"end_date\" : \"202001\"}",
#   "--find_best_model", "--grid_search", "--bayesian_optimizer",
#   "--save_model={\"database\" : \"test\", \"collection\" : \"model\"}",
# ]

def parse_commandline() -> dict:
    """"커맨드라인 파싱"""

    import argparse
    import json
    import textwrap

    description = textwrap.dedent("""
    ===========================================================================
    --data, table Example
    Format: Dictionary Format
    Key: "database", "collection", "start_date", "end_date"
         "database": Database Name
         "collection": Collection Name
         "start_date": Gather Start Date(str)
         "end_date": Gather End Date(str)
    e.g. --data='{"database": "database", "collection": "collection",
                  "start_date": "202201", "end_date": "202201"}'
         => Query 202201 Data

    --split_ratio Example
    Format: Dictionary Format
    Key: "train", "validation", "test"
    e.g. 1) --split_ratio='{"train":"1.0","validation":"0.0","test":"0.0"}' 
            => Train Set 100%
         2) --split_ratio='{"train":"0.8","validation":"0.1","test":"0.1"}'
            => Train Set 80% / Validation Set 10% / Test Set 10%
         3) --split_ratio='{"train":"0.7","validation":"0.0","test":"0.3"}' 
            => Train Set 70% / Test Set 30%
         4) --split_ratio='{"train":"0.0","validation":"0.0","test":"1.0"}' 
            => Test Set 100%
         5) None => Train Set 100%

    --show_metric_by_thresholds Example
    e.g. --show_metric_by_thresholds 0.2,0.3,0.6 
         => String With Separator ',' 
            (Evaluate by Threshold 0.2, 0.3, 0.6 Value)

    --classification_file_name: Classification File Name (YAML Format)
    
    --load_model: Load Model Dictionary
    Format: Dictionary Format
    Key: "database": Database Name
         "collection": Collection Name
         "start_date": Data Start Date Used For Training
         "end_date": Data End Date Used For Training
    e.g. --load_model={"database": "test","collection": "model",
                       "start_date": "202201","end_date": "202201"}"

    --save_model: Save Model Dictionary
    Format: Dictionary Format
    Key: "database": Database Name
         "collection": Collection Name
    e.g. "--save_model={"database": "test","collection": "model"}",
    ===========================================================================
    """)

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '--experiment', default=None, type=str, help="MLFlow Experiment Name.")
    parser.add_argument(
        '--run_name', default=None, type=str, help='Run Name.')
    parser.add_argument(
        '--data', type=json.loads, help='Train Data Dictionary.')
    parser.add_argument(
        '--seed', default=None, type=int, help="Seed.")
    parser.add_argument(
        '--table', type=json.loads, required=True, help='Table Dictionary.')
    parser.add_argument(
        '--split_ratio', default=None, type=json.loads,
        help="Split Ratio Dictionary.")
    parser.add_argument(
        '--classification_file_name', required=True, type=str,
        help="Classification File Name. (YAML Format File)")
    parser.add_argument(
        '--show_data', action='store_true', help="Show Data Information.")
    parser.add_argument(
        '--train', action='store_true', help="Do Train.")
    parser.add_argument(
        '--load_model', default=None, type=json.loads,
        help="Load Model Dictionary.")
    parser.add_argument(
        '--evaluate', action='store_true', help="Do Evaluate.")
    parser.add_argument(
        '--show_metric_by_thresholds', default=None, type=str,
        help="Apply Threshold Value to Evaluation. String(Separator=',')")
    parser.add_argument(
        '--show_optimal_metric', action='store_true',
        help="Show Optimal Metric.")
    parser.add_argument(
        '--find_best_model', action='store_true', help="Find Best Model.")
    parser.add_argument(
        '--grid_search', action='store_true',
        help="Find Hyper Parameters by Grid Search.")
    parser.add_argument(
        '--bayesian_optimizer', action='store_true',
        help="Find Hyper Parameters by Bayesian Optimizer.")
    parser.add_argument(
        '--save_model', default=None, type=json.loads,
        help="Save Model Dictionary.")

    # argparse.Namespace to dict
    args, _ = parser.parse_known_args()
    return vars(args)

def parse_commandline_to_pipeline_dict() -> dict:
    """
    파싱된 커맨드라인을 pipeline에서 사용할 dictionary 값으로 다시 파싱.

    웹에서 설정하지 않은 파라미터는 argparse 를 통과하면 기본 False 로
    세팅되므로 True 거나 실제 값이 있는 파라미터만 사용하기 위해
    다시 dictionary 로 세팅함.
    """
    parameters = parse_commandline()
    if isinstance(parameters, dict) is False:
        raise Exception('Parameters must be dict.',
                        get_code_line(inspect.currentframe()))

    return_value = {}
    for key in parameters:
        if isinstance(parameters[key], dict):
            value_dict = parameters[key]
            make_dict = {}
            # dict 내에 값이 dict 형태라면 각각의 value 가 있을 경우만 재 조합.
            for value_key in value_dict:
                if value_dict[value_key]:
                    make_dict[value_key] = value_dict[value_key]
            if len(make_dict) > 0:
                return_value[key] = make_dict
        else:
            # 웹에서 설정한 값이 True거나 ''가 아닐 경우.
            if ((parameters[key]) or (parameters[key] is True)):
                return_value[key] = parameters[key]

    # split 값은 str이므로 float 타입으로 변환.
    if 'split_ratio' in return_value:
        split_ratio = return_value['split_ratio']
        for key, value in split_ratio.items():
            if value:
                split_ratio[key] = float(value)
        return_value['split_ratio'] = split_ratio

    if 'show_metric_by_thresholds' in return_value:
        # 구분자 컴마(,) 파싱.
        thresholds_list = return_value['show_metric_by_thresholds']
        list_data = list(thresholds_list.split(','))
        return_value['show_metric_by_thresholds']=\
            [float(x.strip()) for x in list_data if x]

    if not return_value:
        raise Exception('argparse convert failed.',
                        get_code_line(inspect.currentframe()))

    return return_value

class PipeLine:
    """Pipeline class"""
    def __init__(self, parameters: dict, artifact_path: str):
        """생성자"""
        # mlflow 사용 flag.
        self._use_mlflow = True

        self._parameters = parameters
        if ((isinstance(parameters, dict) is False) or (not parameters)):
            raise Exception('Parameter is Invalid.')

        self._experiment = None
        self._artifact_path = None
        self._experiment_id = None

        if 'experiment' in self._parameters:
            self._use_mlflow = True

            mlflow_url = os.environ.get('MLFLOW_URL')
            if not mlflow_url:
                raise Exception('MLFLOW_URL environment variable not found.')

            import mlflow
            mlflow.set_tracking_uri(mlflow_url)
            self._experiment = mlflow.get_experiment_by_name(
                self._parameters['experiment'])
            # 없으면 생성
            if self._experiment is None:
                mlflow.create_experiment(self._parameters['experiment'])
                self._experiment = mlflow.get_experiment_by_name(
                    self._parameters['experiment'])

            self._experiment_id = self._experiment.experiment_id
            self._artifact_path = artifact_path
            if (not self._experiment_id) or (not self._artifact_path):
                raise Exception('Experiment Id or Artifact Path is None.')

        else:
            self._use_mlflow = False

    def start(self) -> None:
        """파이프라인 시작"""
        if self._use_mlflow:
            import mlflow
            with mlflow.start_run(
                run_name=_validate(self._parameters, 'run_name'),
                experiment_id=self._experiment.experiment_id):

                mlflow.set_tag('mlflow.note.content',
                                _validate(self._parameters, 'run_name'))

                try:
                    self.run_experiment()

                except BaseException as ex_message:
                    TraceLog().info(f"Error occurred. Message:{ex_message}")

                finally:
                    self.mlflow_logging_file()
        else:
            self.run_experiment()

    def find_child_runs(self, active_run_id) -> Union[List, Any]:
        """현재 run에서 nested child run을 찾기위한 method

        Args:
            active_run_id (mlflow.entities.RunInfo.run_id): mlflow run_id

        Returns:
            (List[Run] | pandas.DataFrame): run object 리스트나 DataFrame
        """
        if not self._use_mlflow:
            return None

        if not active_run_id:
            return None

        import mlflow
        filter_child_runs = f"tags.mlflow.parentRunId = '{active_run_id}'"
        return mlflow.search_runs(
                    experiment_ids=[self._experiment_id],
                    filter_string=filter_child_runs,
                    order_by=['attribute.start_time ASC'])

    def run_experiment(self) -> None:
        """실험 시작"""
        TraceLog().info('- Parameter List -')
        for k, v in self._parameters.items():
            TraceLog().info(f"key=({k}) value=({v})")

        TraceLog().info('- Start PipeLine -')

        # parent run 에 log_param 세팅.
        self.mlflow_logging_pipeline_parameters()

        # 데이터 분리.
        model = Model.prepare_data(
            data=_validate(self._parameters, 'data'),
            table=_validate(self._parameters, 'table'),
            split_ratio=_validate(options, 'split_ratio'),
            seed=_validate(self._parameters, 'seed'))

        # Classifier를 통한 모델 초기화 (명시적 호출 필요).
        model.init_model(
            _validate(self._parameters, 'classification_file_name'))

        # 분리된 데이터 출력.
        if 'show_data' in self._parameters:
            TraceLog().info('- Split Data Information -')
            model.show_data()

        # 모델 로드.
        if 'load_model' in self._parameters:
            TraceLog().info('- Start Load Model -')
            model.load_model(
                model=_validate(self._parameters, 'load_model'),
                table=_validate(self._parameters, 'table'))

        for name in self.mlflow_start_child_run(model.get_classifier_names()):

            self.mlflow_logging_pipeline_parameters()
            TraceLog().info(f"============== Model:{name} ==============")
            # 학습.
            if 'train' in self._parameters:
                TraceLog().info('- Start Train -')
                model.train_by_name(name=name)

            # 평가.
            if 'evaluate' in self._parameters:
                TraceLog().info('- Start Evaluate -')
                model.evaluate_by_name(name=name)

            # 평가 지표 출력.
            if 'show_metric_by_thresholds' in self._parameters:
                TraceLog().info('- Show Metrics to Evaluation by Threshold -')
                threshold_list = _validate(
                    self._parameters, 'show_metric_by_thresholds')
                for data in model.get_evaluate_metrics_by_name(
                    name=name, thresholds=threshold_list):
                    if data.verify():
                        TraceLog().info(f"Threshold:({data.threshold})")
                        PipeLine.trace_metric(data.metric)

            # 모델의 f1-score가 가장 높은 임계값과 평가 지표를 출력.
            if 'show_optimal_metric' in self._parameters:
                TraceLog().info('- Show Optimal Metric -')
                for metric_score_info in model.get_optimal_metrics(
                    name=name, metric_name_list=[MetricData.str_f1()]):
                    PipeLine.trace_optimal_metric(
                        metric_score_info=metric_score_info)
                    self.mlflow_logging_optimal_metric(
                        metric_score_info=metric_score_info)

            # Hyper Parameter 찾기.
            if 'grid_search' in self._parameters:
                TraceLog().info('- Start Grid Search -')
                best_param = model.grid_search(name=name)
                TraceLog().info(f'best param={best_param}')
                self.mlflow_logging_hyper_parameters(best_param)

            if 'bayesian_optimizer' in self._parameters:
                TraceLog().info('- Start Bayesian Optimizer -')
                max_value = model.get_hyper_parameters_by_bo(name=name)
                TraceLog().info(f'best params={max_value}')
                self.mlflow_logging_hyper_parameters(max_value)

            # 개별 모델 저장.
            if 'save_model' in self._parameters:
                TraceLog().info('- Start Save Model -')
                # mlflow로 모델 저장 시 자체 로컬 경로를 생성하여 저장함.
                local_model_path = self.mlflow_save_model(
                    pipeline_tuple=model.get_pipeline_with_name(name=name))
                # 리턴 받은 실제 경로를 model 클래스에 다시 세팅함.
                model.set_save_model_file(
                    name=name,
                    model_file=local_model_path)

        # model 객체에 개별 저장된 경로를 Database에 저장.
        # (개별 모델의 루프가 종료된 후 다시 저장 해야 함)
        if 'save_model' in self._parameters:
            model.save_model_information_to_database(
                model=_validate(self._parameters, 'save_model'),
                data=_validate(self._parameters, 'data'),
                table=_validate(self._parameters, 'table'),
                save_path_list=model.get_save_model_path_list())

        # 평가지표가 가장 높은 알고리즘을 찾아 임계값과 평가 지표를 출력.
        if 'find_best_model' in self._parameters:
            TraceLog().info('- Find Highest Score For Model -')
            score_info = model.get_highest_score_for_model()

            PipeLine.trace_highest_score(score_info)
            self.mlflow_logging_highest_score_for_model(
                score_info=score_info,
                local_model_paths=model.get_local_model_path_with_name())

        return

    def mlflow_start_child_run(
        self, classifier_names:Iterator[str]) -> Iterator[str]:
        """mlflow를 사용할때 nested child run 을 생성하는 제너레이터

        Args:
            classifier_names (Iterator[str]): Classifier 이름 제너레이터

        Yields:
            Iterator[str]: Classifier 이름 제너레이터
        """
        for name in classifier_names:
            if self._use_mlflow:
                import mlflow
                with mlflow.start_run(
                            run_name=name,
                            experiment_id=self._experiment.experiment_id,
                            nested=True):
                    yield name
            else:
                yield name

    def mlflow_logging_file(self) -> None:
        """mlflow 에 trace.log 파일 로깅
           현재는 parent run 에만 기록
        """
        if not self._use_mlflow:
            return

        trace_log_path = os.path.join(
            self._artifact_path, 'trace', 'Trace.log')

        import mlflow
        if os.path.exists(trace_log_path):
            # 현재 run 에서 로그 파일 로깅함
            mlflow.log_artifact(trace_log_path)

            # nested child run 에 기록하면 duration 값이 갱신되므로
            # 똑같은 로그를 남기지 않도록 함.
            # runs = self.find_child_runs(mlflow.active_run().info.run_id)
            # for (_, row) in runs.iterrows():
            #     with mlflow.start_run(
            #         run_id=row['run_id'], nested=True):
            #         mlflow.log_artifact(trace_log_path)

        yaml_file = Classifier.get_classification_path_from_file_name(
            _validate(self._parameters, 'classification_file_name'))
        if os.path.exists(yaml_file):
            mlflow.log_artifact(yaml_file)

    def mlflow_logging_pipeline_parameters(self) -> None:
        """mlflow에 실행 파라미터를 파라미터 항목에 로깅"""
        if not self._use_mlflow:
            return

        import mlflow
        for key_value_list in _get_dicts(self._parameters):
            if not key_value_list:
                continue
            key_string = ''
            value = key_value_list[-1]
            # 리스트 값이 너무 길어질 수 있어 특정 문자열만 로깅.
            if isinstance(value, list):
                value = 'String List Type'
            for child_key in key_value_list[:-1]:
                key_string += f"{child_key}_"
            key_string = key_string[:-1] # 마지막 _ 제거.

            mlflow.log_param(key_string, value)

    def mlflow_logging_classifier_parameter(
        self, name: str, classifier_dict: dict) -> None:
        """mlflow에 classifier 파라미터를 파라미터 항목에 로깅"""
        if not self._use_mlflow:
            return

        if (isinstance(name, str) is False) or (not name):
            return

        def remove_prefix(origin: dict):
            """'clf__'가 있으면 제거함"""
            for key, value in origin.items():
                if key.startswith('clf__'):
                    key = key.replace('clf__', '')
                yield {key: value}

        import mlflow
        hyper_param = classifier_dict[name][Classifier.Index.HYPER_PARAMETER]
        for remove_dict in remove_prefix(hyper_param):
            mlflow.log_params(remove_dict)

    def mlflow_logging_metric(self, metric: MetricData) -> None:
        """mlflow에 평가 메트릭 로깅

        Args:
            metric (MetricData): 메트릭 데이터 클래스
        """
        if not self._use_mlflow:
            return

        if (isinstance(metric, MetricData) is False) and (not metric):
            return

        import mlflow
        if metric.accuracy:
            mlflow.log_metric(metric.str_accuracy(), metric.accuracy)
        if metric.precision:
            mlflow.log_metric(metric.str_precision(), metric.precision)
        if metric.recall:
            mlflow.log_metric(metric.str_recall(), metric.recall)
        if metric.f1:
            mlflow.log_metric(metric.str_f1(), metric.f1)
        if metric.roc_auc:
            if isinstance(metric.roc_auc, dict) is False:
                mlflow.log_metric(metric.str_roc_auc(), metric.roc_auc)

    def mlflow_logging_optimal_metric(
        self, metric_score_info: MetricScoreInfo) -> None:
        """모델의 f1-score를 로깅함

        Args:
            metric_score_info (MetricScoreInfo): MetricScoreInfo 데이터 클래스
        """

        if self._use_mlflow is False:
            return

        if ((isinstance(metric_score_info, MetricScoreInfo) is False) and
            (not metric_score_info)):
            return

        if metric_score_info.verify():
            if metric_score_info.metric_name == MetricData.str_f1():
                import mlflow
                mlflow.set_tag('Best F1-Score Threshold',
                               str(metric_score_info.score_info.threshold))
                self.mlflow_logging_metric(
                    metric=metric_score_info.score_info.metric)

    def mlflow_logging_hyper_parameters(self, best_param: dict) -> None:
        """찾은 Hyper Parameter를 mlflow tag에 로깅

        Args:
            best_param (dict): hyper parameter 정보
        """
        if not self._use_mlflow:
            return

        if (not best_param) or (isinstance(best_param, dict) is False):
            return

        import mlflow

        def _replace_prefix(origin: dict):
            """'clf__'가 있으면 변경함"""
            for key, value in origin.items():
                if key.startswith('clf__'):
                    key = key.replace('clf__', 'Best Param ')
                yield {key: value}

        for value in _replace_prefix(best_param):
            mlflow.set_tags(value)

    @staticmethod
    def trace_metric(metric: MetricData) -> None:
        """콘솔 및 로깅에 메트릭 값 전달

        Args:
            metric (MetricData): MetricData 데이터 클래스
        """
        if (isinstance(metric, MetricData) is False) and (not metric):
            return

        if metric.confusion.size:
            TraceLog().info("-Confusion Matrix-")
            for matrix in metric.confusion:
                TraceLog().info(matrix)

        message = ""
        if metric.accuracy:
            message += f"{metric.str_accuracy()}=({metric.accuracy:.4f}) "
        if metric.precision:
            message += f"{metric.str_precision()}=({metric.precision:.4f}) "
        if metric.recall:
            message += f"{metric.str_recall()}=({metric.recall:.4f}) "
        if metric.f1:
            message += f"{metric.str_f1()}=({metric.f1:.4f}) "
        if metric.roc_auc:
            if isinstance(metric.roc_auc, dict):
                message += (f"{metric.str_roc_auc()}=({str(metric.roc_auc)})")
            else:
                message += (f"{metric.str_roc_auc()}=({metric.roc_auc:.4f})")
        TraceLog().info(message)

    @staticmethod
    def trace_optimal_metric(metric_score_info: MetricScoreInfo) -> None:
        """f1-score의 최적의 메트릭 로깅

        Args:
            metric_score_info (MetricScoreInfo): MetricScoreInfo 데이터 클래스
        """
        if ((isinstance(metric_score_info, MetricScoreInfo) is False) and
            (not metric_score_info)):
            return

        message = None
        if metric_score_info.metric_name == MetricData.str_f1():
            message = '= Optimal Threshold F1-Score Metric ='
        if not message:
            return

        if metric_score_info.verify():
            TraceLog().info(message)
            TraceLog().info(
                f"Threshold:{metric_score_info.score_info.threshold}")
            PipeLine.trace_metric(metric_score_info.score_info.metric)

    @staticmethod
    def trace_highest_score(score_info:BestModelScoreInfo) -> None:
        """전체 모델 중 f1-score 가 가장 높게 나온 모델의\
           임계값과 메트릭 출력

        Args:
            score_info (BestModelScoreInfo): BestModelScoreInfo 데이터 클래스
        """
        if ((isinstance(score_info, BestModelScoreInfo) is False) or
            (not score_info)):
            return

        if score_info.highest_f1.verify():
            TraceLog().info('============== Highest F1-Score ==============')
            TraceLog().info(f"Model:{score_info.highest_f1.name}")
            TraceLog().info(f"Threshold:{score_info.highest_f1.threshold}")
            PipeLine.trace_metric(score_info.highest_f1.metric)

    def mlflow_save_model(self, pipeline_tuple: Tuple[str, Any]) -> str:
        """mlflow에 학습한 개별 모델을 업로드

        Args:
            pipeline_tuple (Tuple): (알고리즘 이름, 학습된 파이프라인 객체)

        Returns:
            str: mlflow API를 통해 저장된 실제 경로
        """
        if not self._use_mlflow:
            return

        import mlflow
        import json
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import Schema, ColSpec

        def make_nested_json_from_parameters(key_list:list):
            """파라미터에서 {root key: {child key: value}} 구조를
                yield 하는 함수
            """
            for key in key_list:
                for nested_data in _get_dicts(
                    _validate(self._parameters, key)):
                    if not nested_data:
                        continue
                    yield json.dumps(
                        {key: {nested_data[0]: nested_data[1]}})

        schema_list = []
        for schema in make_nested_json_from_parameters(
            ['data', 'table', 'save_model']):
            schema_list.append(ColSpec('string', schema))

        mlflow.sklearn.log_model(
            sk_model=pipeline_tuple,
            artifact_path='model',
            signature=ModelSignature(inputs=Schema(schema_list)))

        # mlflow에서 지정된 모델명
        local_model_path = os.path.join(
            mlflow.get_artifact_uri(), 'model', 'model.pkl')
        return local_model_path

    def mlflow_logging_highest_score_for_model(
        self, score_info:BestModelScoreInfo,
        local_model_paths:Iterator[Tuple[str, str]]) -> None:
        """mlflow root run 에 가장 높은 점수의 모델을 기록

        Args:
            score_info (BestModelScoreInfo): BestModelScoreInfo 데이터 클래스
            local_model_paths (Iterator[Tuple[str, str]]):\
                Tuple(모델명, 저장된 모델 경로) 제너레이터
        """

        if not self._use_mlflow:
            return

        if ((isinstance(score_info, BestModelScoreInfo) is False) or
            (not score_info)):
            return

        # f1-score가 가장 높은 모델을 부모 experiment에 기록함
        if score_info.highest_f1.verify() is False:
            return

        import mlflow
        # 모델 경로 제너레이터에서 mlflow에 의해 저장된 로컬 경로명 로드
        model_path = None
        for (name, path) in local_model_paths:
            if name == score_info.highest_f1.name:
                if os.path.exists(path):
                    model_path = path
                break

        if model_path:
            # active run 은 parent run 이어야 함
            active_run_id = mlflow.active_run().info.run_id
            run_object = mlflow.get_run(run_id=active_run_id)

            # 이미 해당 run에 notes가 있으면 append
            note_content = None
            if 'mlflow.note.content' in run_object.data.tags:
                note_content=\
                    run_object.data.tags['mlflow.note.content'] + '\n\n'

            # mlflow에서 저장된 로컬 경로에서 해당 run id 추출
            model_run_id = None
            # /mlruns 찾음
            if model_path.startswith(self._experiment.artifact_location):
                model_run_id = model_path.replace(
                    self._experiment.artifact_location, '').split(os.sep)[1]

            # 바로 접근할 수 있는 url 생성
            model_url = (f'{mlflow.get_tracking_uri()}/#/'
                            f'experiments/{self._experiment_id}/'
                            f'runs/{model_run_id}/artifactPath/models\n')
            note_content += f'Best Model URL\n{model_url}'
            mlflow.set_tag('mlflow.note.content', note_content)

        mlflow.set_tag('Best F1-Score Model Name', score_info.highest_f1.name)
        mlflow.set_tag('Best F1-Score Threshold',
                       str(score_info.highest_f1.threshold))
        self.mlflow_logging_metric(metric=score_info.highest_f1.metric)

if __name__ == "__main__":

    try:
        log = init_log_object(log_file_name=get_log_file_name(__file__))
        log.info('=========== Start ===========')

        options = parse_commandline_to_pipeline_dict()

        from common.function import TempDir
        with TempDir(prefix='mlflow_') as tmp:
            trace_path = tmp.path('trace')
            os.mkdir(trace_path)
            TraceLog().initialize(
                log_file_name=os.path.join(trace_path, 'Trace.log'))

            p = PipeLine(parameters=options, artifact_path=tmp.path())
            p.start()

    except BaseException as e:
        log.info(f"Error occurred. Message:{e}")

    finally:
        log.info('===========  End  ===========')
