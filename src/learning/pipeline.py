import os
import inspect
import argparse
import json
import textwrap
from typing import Iterator, List, Tuple, Any
from common.container.mongo import MongoDBConnection

from common.data_type import MetricData, MetricScoreInfo, BestModelScoreInfo
from common.function import get_code_line, TempDir
from common.trace_log import TraceLog, init_log_object, get_log_file_name

from learning.classifier import Classifier
from learning.model import Model

# VSCODE launch.json parameter example.
# "args" : [
#   "--experiment=test", "--run_name=test",
#   "--_data={\"database\" : \"test\", \"collection\" : \"iris.data\",
#                  \"start_date\" : \"202501\", \"end\" : \"202501\"}",
#   "--table={\"database\":\"test\",\"collection\":\"table\",
#             \"start_date\" : \"202501\",\"end\" : \"202501\"}",
#   "--seed=123", "--show_data", "--classification_file_name=iris.yaml",
#   "--show_metric_by_thresholds=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
#   "--train", "--train_with_tuning", "--evaluate", "--show_optimal_metric",
#   "--load_model={\"database\" : \"test\", \"collection\" : \"model\",
#                  \"start_date\" : \"202501\", \"end_date\" : \"202501\"}",
#   "--find_best_model",
#   "--save_model={\"database\" : \"test\", \"collection\" : \"model\"}",
# ]

def parse_commandline() -> dict:
    """"커맨드라인 파싱"""
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
                  "start_date": "202501", "end_date": "202501"}'
         => Query 202501 Data

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
                       "start_date": "202501","end_date": "202501"}"

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
        '--train_with_tuning', action='store_true',
        help="Train With Hyper Parameter Tuning.")
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
        self._artifact_path = ''
        self._experiment_id = ''

        if 'experiment' in self._parameters:
            self._use_mlflow = True

            mlflow_url = os.environ.get('MLFLOW_URL')
            if not mlflow_url:
                raise Exception('MLFLOW_URL environment variable not found.')

            import mlflow
            mlflow.set_tracking_uri(uri=mlflow_url)
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
                run_name=self._parameters.get('run_name', ''),
                experiment_id=self._experiment.experiment_id):
                mlflow.set_tag(
                    'mlflow.note.content',
                    self._parameters.get('run_name', ''))
                try:
                    self.run_experiment()

                except BaseException as ex_message:
                    TraceLog().info(f"Error occurred. Message:{ex_message}")

                finally:
                    self.mlflow_logging_file()
        else:
            self.run_experiment()

    def find_child_runs(self, active_run_id) -> List | Any:
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
            data=self._parameters.get('data', None),
            table=self._parameters.get('table', None),
            split_ratio=self._parameters.get('split_ratio', None),
            seed=self._parameters.get('seed', None))

        # Classifier를 통한 모델 초기화 (명시적 호출 필요).
        classifier_file_name = self._parameters.get(
            'classification_file_name', None)
        model.init_model(file_name=classifier_file_name)

        # 분리된 데이터 출력.
        if 'show_data' in self._parameters:
            TraceLog().info('- Split Data Information -')
            model.show_data()

        # 이미 학습된 모델 로드.
        if 'load_model' in self._parameters:
            TraceLog().info('- Start Load Model -')
            model.load_model(
                model=self._parameters.get('load_model', None),
                table=self._parameters.get('table', None))

        for name in self.mlflow_start_child_run(model.get_classifier_names()):

            TraceLog().info(f"============== Model:{name} ==============")
            self.mlflow_logging_pipeline_parameters()
            # 학습.
            if 'train' in self._parameters:
                TraceLog().info('- Start Train -')
                model.train(name=name)

            # 파라미터 튜닝 및 학습.
            if 'train_with_tuning' in self._parameters:
                TraceLog().info('- Start Train With Tuning -')
                parameters = model.tuning(
                    name=name, update_file=classifier_file_name)
                if parameters:
                    TraceLog().info(f'Best Paramters=({parameters})')
                model.train(name=name)

            # 평가.
            if 'evaluate' in self._parameters:
                TraceLog().info('- Start Evaluate -')
                model.evaluate_by_classifier_name(name=name)

            # 평가 지표 출력.
            if 'show_metric_by_thresholds' in self._parameters:
                TraceLog().info('- Show Metrics to Evaluation by Threshold -')
                threshold_list = self._parameters.get(
                    'show_metric_by_thresholds', None)
                for data in model.get_evaluate_metrics(
                    name=name, thresholds=threshold_list):
                    TraceLog().info(f"Threshold:({data.threshold})")
                    PipeLine.trace_metric(data.metric)

            # 모델의 지정한 Score가 가장 높은 임계값과 평가 지표를 출력.
            if 'show_optimal_metric' in self._parameters:
                TraceLog().info('- Show Optimal Metric -')
                metric_name_list = [MetricData.str_f1()]
                for metric_score_info in model.get_optimal_metrics(
                    name=name, metric_name_list=metric_name_list):
                    PipeLine.trace_optimal_metric(
                        metric_score_info=metric_score_info)
                    self.mlflow_logging_optimal_metric(
                        metric_score_info=metric_score_info)

            # 개별 모델 저장.
            if 'save_model' in self._parameters:
                TraceLog().info('- Start Save Model -')
                # mlflow로 모델 저장 시 자체 로컬 경로를 생성하여 저장함.
                local_model_path = self.mlflow_save_model(
                    pipeline_tuple=model.get_pipeline_with_name(name=name))
                # mlflow artifact 경로에 모델을 저장하므로 mlflow 설정없이는
                # 로컬에 모델을 저장할 수 없음
                if not local_model_path:
                    raise ValueError('MLFlow parameter are required '
                                  'to create model local path.')
                # 리턴 받은 실제 경로를 model 클래스에 다시 세팅함.
                model.set_save_model_file(
                    name=name, model_file=local_model_path)

        # model 객체에 개별 저장된 경로를 Database에 저장.
        # (개별 모델의 루프가 종료된 후 다시 저장 해야 함)
        if 'save_model' in self._parameters:
            model.save_model_information_to_database(
                model=self._parameters.get('save_model', None),
                data=self._parameters.get('data', None),
                table=self._parameters.get('table', None),
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

        # 파라미터 파일 로깅함
        yaml_file = Classifier.get_yml_file_path(
            self._parameters.get('classification_file_name', None))
        if os.path.exists(yaml_file):
            mlflow.log_artifact(yaml_file)

    def mlflow_logging_pipeline_parameters(self) -> None:
        """mlflow에 실행 파라미터를 파라미터 항목에 로깅"""
        if not self._use_mlflow:
            return

        import mlflow
        for k, v in self._parameters.items():
            mlflow.log_param(k, str(v).replace("'", "\""))

    def mlflow_logging_classifier_parameter(
        self, name: str, parameter_file_name: str) -> None:
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

        classifier = Classifier(file_name=parameter_file_name)
        hyper_param = classifier.get_hyper_parameter(name=name)
        if not hyper_param:
            return

        import mlflow
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
        # 모든 metric은 0.0일 경우도 있어 유효성 체크하지 않고 기록함
        if isinstance(metric.accuracy, float):
            mlflow.log_metric(metric.str_accuracy(), round(metric.accuracy, 4))
        if isinstance(metric.precision, float):
            mlflow.log_metric(metric.str_precision(),round(metric.precision, 4))
        if isinstance(metric.recall, float):
            mlflow.log_metric(metric.str_recall(), round(metric.recall, 4))
        if isinstance(metric.f1, float):
            mlflow.log_metric(metric.str_f1(), round(metric.f1, 4))
        if isinstance(metric.roc_auc, float):
            mlflow.log_metric(metric.str_roc_auc(), metric.roc_auc, 4)
        if isinstance(metric.miss_rate, float):
            mlflow.log_metric(metric.str_miss_rate(), round(metric.miss_rate, 4))
        if isinstance(metric.fall_out, float):
            mlflow.log_metric(metric.str_fall_out(), round(metric.fall_out, 4))

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

        # 현재 평가지표로 사용하는 메트릭은 f1-score
        if metric_score_info.metric_name == MetricData.str_f1():
            import mlflow
            mlflow.set_tag('best_threshold',
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
        # 미탐, 과탐률은 0.0일 경우도 있어 유효성 체크하지 않고 기록함
        message = ""
        message = f"{metric.str_miss_rate()}=({metric.miss_rate:.4f}) "
        message += f"{metric.str_fall_out()}=({metric.fall_out:.4f})"

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

        TraceLog().info(message)
        TraceLog().info(
            f"Threshold:{metric_score_info.score_info.threshold}")
        PipeLine.trace_metric(metric_score_info.score_info.metric)

    @staticmethod
    def trace_highest_score(score_info:BestModelScoreInfo) -> None:
        """전체 모델 중 f1-score가 가장 높게 나온 모델의\
           임계값과 메트릭 출력

        Args:
            score_info (BestModelScoreInfo): BestModelScoreInfo 데이터 클래스
        """
        if ((isinstance(score_info, BestModelScoreInfo) is False) or
            (not score_info)):
            return

        TraceLog().info(f"{'='*17} Highest F1-Score {'='*17}")
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
            return ''

        import mlflow
        import mlflow.sklearn
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import Schema, ColSpec

        schema_list = []
        for key in ['train_data', 'test_data', 'table', 'save_model']:
            if self._parameters.get(key, None):
                schema = json.dumps({key: self._parameters[key]})
                schema_list.append(ColSpec('string', schema))

        # 저장하려는 모델의 package 버전을 추가함
        requirements_list = []
        requirements_list.append(
            Classifier.get_pip_package_string(Classifier.RANDOMFOREST))
        if pipeline_tuple[0] != Classifier.RANDOMFOREST:
            # sklearn을 중복해서 추가할 필요 없음
            requirements_list.append(
                Classifier.get_pip_package_string(pipeline_tuple[0]))

        # pip_requirements나 conda_env에 pip 값을 추가하면 자동으로
        # mlflow 버전이 추가됨
        mlflow.sklearn.log_model(
            sk_model=pipeline_tuple,
            artifact_path='model',
            signature=ModelSignature(inputs=Schema(schema_list)),
            pip_requirements=requirements_list,)

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

        if ((not isinstance(score_info, BestModelScoreInfo)) or
            (not score_info)):
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

            # 이미 해당 run에 notes가 있으면 append 해야함
            note_content = None
            if 'mlflow.note.content' in run_object.data.tags:
                note_content = run_object.data.tags\
                                    ['mlflow.note.content'] + '\n\n'

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

        mlflow.set_tag('best_model', score_info.highest_f1.name)
        mlflow.set_tag('best_threshold', str(score_info.highest_f1.threshold))

        # f1-score를 로깅함
        self.mlflow_logging_metric(metric=score_info.highest_f1.metric)

if __name__ == "__main__":

    # mlflow logging 과 따로 남기기 위해 로그 객체 새로 생성
    log = init_log_object(log_file_name=get_log_file_name(__file__))

    try:
        log.info('========================= Start ========================')
        MongoDBConnection().initialize()
        options = parse_commandline_to_pipeline_dict()
        with TempDir(prefix='mlflow_') as tmp:
            trace_path = tmp.path('trace')
            os.mkdir(trace_path)
            TraceLog().initialize(
                log_file_name=os.path.join(trace_path, 'Trace.log'))

            p = PipeLine(parameters=options, artifact_path=tmp.path())
            p.start()

    except BaseException as e:
        import traceback
        log.info(f"Error occurred. Message:{e} {traceback.format_exc()}")

    finally:
        MongoDBConnection().close()
        log.info('========================== End =========================')
