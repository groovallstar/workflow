import os

from common.function import _validate
from common.data_type import ThresholdMetricData, BestModelScoreInfo
from common.trace_log import TraceLog

from learning.classifier import Classifier
from learning.model import Model

def pipeline(experiment, options: dict) -> None:
    """pipeline"""
    TraceLog().info('- Parameter List -')
    for _key, _value in options.items():
        TraceLog().info(f'key=({_key}) value=({_value})')

    TraceLog().info('- Start pipeline -')

    # 데이터 분리.
    model = Model.prepare_data(
        data=_validate(options, 'data'),
        table=_validate(options, 'table'),
        split_ratio=_validate(options, 'split_ratio'),
        sampling=_validate(options, 'sampling'),
        seed=_validate(options, 'seed'))

    # Classifier를 통한 모델 초기화 (명시적 호출 필요).
    model.init_model(_validate(options, 'classification_file_name'))

    # 분리된 데이터 출력.
    if 'show_data' in options:
        TraceLog().info('- Split Data Information -')
        model.show_data()

    # 학습.
    if 'train' in options:
        TraceLog().info('- Start Train -')
        model.train()

    # 모델 로드.
    if 'load_model' in options:
        TraceLog().info('- Start Load Pre-Trained Model -')
        model.load_model(
            model=_validate(options, 'load_model'),
            table=_validate(options, 'table'))

    # 교차검증.
    if 'cross_validation' in options:
        TraceLog().info('- Start Cross Validation -')
        for name in model.get_classifier_names():
            TraceLog().info(f"= Model:{name} =")
            cross_val_score = model.cross_validation(name)
            TraceLog().info(f'cross validation score=({cross_val_score})')

    # Hyper Parameter 찾기.
    if 'grid_search' in options:
        TraceLog().info('- Start Grid Search -')
        for name in model.get_classifier_names():
            TraceLog().info(f"= Model:{name} =")
            best_params = model.grid_search(name)
            TraceLog().info(f'best param={best_params}')

    if 'bayesian_optimizer' in options:
        TraceLog().info('- Start Bayesian Optimizer -')
        for name in model.get_classifier_names():
            TraceLog().info(f"= Model:{name} =")
            max_value = model.get_hyper_parameters_by_bo(name)
            TraceLog().info(f'best param={max_value}')

    # 평가.
    if 'evaluate' in options:
        TraceLog().info('- Start Evaluate -')
        model.evaluate()

    # Threshold 에 의한 평가 지표 출력.
    if 'show_metric_by_thresholds' in options:
        TraceLog().info('- Show Metrics to Evaluation by Threshold -')
        for name in model.get_classifier_names():
            TraceLog().info(f"= Model:{name} =")
            for data in model.get_evaluate_metrics_by_name(
                name=name,
                thresholds=_validate(options, 'show_metric_by_thresholds')):
                PipeLineWrapper.trace_evaluate_metric(data)

    # 평가지표가 가장 높은 알고리즘을 찾아 임계값과 평가 지표를 출력.
    if 'show_best_model' in options:
        TraceLog().info('- Show Best Metric Model-')
        best_model_info = model.get_highest_score_for_model()
        PipeLineWrapper.trace_best_model(best_model_info)

    # 학습된 모델 저장.
    if 'save_model' in options:
        TraceLog().info('- Start Save Model -')
        # 로컬에 모델 저장.
        model.dump_model(model=_validate(options, 'save_model'))
        # 데이터베이스에 정보 저장.
        model.save_model_information_to_database(
            model=_validate(options, 'save_model'),
            data=_validate(options, 'data'),
            table=_validate(options, 'table'),
            save_path_list=model.get_save_model_path_list())

    # 평가 메트릭 OmniBoard 에 저장.
    if experiment:
        for name in model.get_classifier_names():
            for metric_data in model.get_evaluate_metrics_by_name(
                name=name, thresholds=[0.5]):
                if metric_data.metric.accuracy:
                    experiment.log_scalar(
                        'accuracy', round(metric_data.metric.accuracy, 4))
                if metric_data.metric.precision:
                    experiment.log_scalar(
                        'precision', round(metric_data.metric.precision, 4))
                if metric_data.metric.recall:
                    experiment.log_scalar(
                        'recall', round(metric_data.metric.recall, 4))
                if metric_data.metric.f1:
                    experiment.log_scalar(
                        'f1', round(metric_data.metric.f1, 4))
                if metric_data.metric.roc_auc:
                    if isinstance(metric_data.metric.roc_auc, dict) is False:
                        experiment.log_scalar(
                            'roc_auc', round(metric_data.metric.roc_auc, 4))

    return

# VSCODE launch.json parameter example.
# "args" : ["-n=Experiment Name", "-c=Comment",
#     "--omniboard=test",
#     "--data={\"database\" : \"test\", \"collection\" : \"iris.data\",
#              \"start_date\" : \"202201\", \"end\" : \"202201\"}",
#     "--table={\"database\" : \"test\",\"collection\" : \"iris.table\",
#               \"start_date\" : \"202201\",\"end\" : \"202201\"}",
#     "--sampling=0.5", "--seed=123", "--show_data",
#     "--classification_file_name=iris.yaml",
#     "--show_metric_by_thresholds", "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
#     "--train",
#     "--load_model={\"database\" : \"test\", \"collection\" : \"model\",
#                    \"start_date\" : \"202201\", \"end_date\" : \"202201\"}",
#     "--evaluate", "--show_metric", "--cross_validation",
#     "--grid_search", "--bayesian_optimizer",
#     "--show_best_model",
#     "--save_model={\"database\" : \"test\", \"collection\" : \"model\",
#                    \"path\" : \"tmp/iris/model/\"}",]

def parse_commandline():
    """"커맨드라인 파싱"""

    import argparse
    import json
    import textwrap

    description = textwrap.dedent("""
    ===========================================================================
    --data, --table Example
    Format: Dictionary Format
    Key: "database", "collection", "start_date", "end_date"
          "database": Database Name
          "collection": Collection Name
          "start_date": Gather Start Date(str)
          "end_date": Gather End Date(str)
    e.g. --data='{"database":"database", "collection":"collection",
                  "start_date":"202201", "end_date":"202202"}'
         => Query 202201 ~ 202202 Data
    
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
    e.g. 1) --show_metric_by_thresholds 0.2 0.3 0.6 
            => String With Separator Space 
               (Evaluate by Threshold 0.2, 0.3, 0.6 Value)
    --classification_file_name: Classification File Name (YAML Format)
    
    --load_model: Load Model Dictionary
    Format: Dictionary Format
    Key: "database": Database Name
         "collection": Collection Name
         "start_date": Pre-Trained Model Start Date
         "end_date": Pre-Trained Model End Date
    e.g. --load_model={"database": "test","collection": "model",
                       "start_date": "202201","end_date": "202201"}"
    --save_model: Save Model Dictionary
    Format: Dictionary Format
    Key: "database": Database Name
         "collection": Collection Name
         "path": Model Save *Root Folder* (Directory)
    e.g. "--save_model={"database": "test", "collection": "model", 
                        "path": "/tmp/iris/model/202201"}",
    ===========================================================================
    """)

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '--ex_name', '-n', default=None, type=str, help='Experiment Name.')
    parser.add_argument(
        '--comments', '-c', default=None, type=str, help='Comment.')
    parser.add_argument(
        '--omniboard', default=None, type=str, help="OmniBoard Database Name.")
    parser.add_argument(
        '--data', type=json.loads, help='Data Dictionary.')
    parser.add_argument(
        '--table', type=json.loads, required=True, help='Table Dictionary.')
    parser.add_argument(
        '--split_ratio', default=None, type=json.loads,
        help="Split Ratio Dictionary.")
    parser.add_argument(
        '--sampling', default=None, type=float, help="Data Sampling.")
    parser.add_argument(
        '--seed', default=None, type=int, help="Seed.")
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
        '--show_metric', action='store_true',
        help="Show Metrics to Evaluation.")
    parser.add_argument(
        '--show_metric_by_thresholds', default=None, type=str,
        help="Apply Threshold Value to Evaluation. String(Separator=',')")
    parser.add_argument(
        '--show_best_model', action='store_true',
        help="Show Best Metric Model.")
    parser.add_argument(
        '--cross_validation', action='store_true', help="Cross Validation.")
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


class PipeLineWrapper():
    """pipeline wrapper class"""
    def __init__(self, parameters: dict):

        options = {}
        options = PipeLineWrapper.convert_parameters_to_pipeline_dict(
            parameters)

        if 'omniboard' in options:
            from sacred.observers import MongoObserver
            from sacred import Experiment
            # 콘솔모드가 아닌 상태에서 captured out 항목에 출력 하기 위해서 추가.
            from sacred.settings import SETTINGS
            SETTINGS.CAPTURE_MODE = 'sys'

            ex = Experiment()
            # omniboard config에 classification 정보 추가
            if _validate(options, 'classification_file_name'):
                yaml_file = Classifier.get_classification_path_from_file_name(
                    options['classification_file_name'])
                if os.path.exists(yaml_file):
                    ex.add_config(yaml_file)

            # ex.run 보다 상위에 선언 해야 run 함수에서 ex.main 호출 가능.
            @ex.main
            def ex_main(_run):
                try:
                    return pipeline(experiment=_run, options=options)
                except Exception as e:
                    TraceLog().info('Error occurred. Message:', e)
                    return False

            sacred_database_url = os.environ.get('SACRED_DATABASE_URL')
            if not sacred_database_url:
                raise Exception('SACRED_DATABASE_URL environment not found.')
            ex.observers.append(MongoObserver.create(
                url=sacred_database_url,
                db_name=options['omniboard']))

            ex.run(options={'--name': _validate(options, 'ex_name'),
                            '--comment': _validate(options, 'comments')})

        else:
            pipeline(experiment=None, options=options)

    @classmethod
    def convert_parameters_to_pipeline_dict(cls, parameters: dict):
        """파싱된 커맨드라인을 pipeline에서 사용할 dictionary 값으로 다시 파싱"""

        if isinstance(parameters, dict) is False:
            import inspect
            from common.function import get_code_line
            raise Exception('Parameters must be dict.',
                            get_code_line(inspect.currentframe()))

        return_value = {}

        # 설정값이 있을 경우만 filtering 하기 위해 체크.
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
            # 구분자 컴마(,)를 파싱 해야 함.
            thresholds_list = return_value['show_metric_by_thresholds']
            list_data = list(thresholds_list.split(','))
            return_value['show_metric_by_thresholds'] = [
                float(x.strip()) for x in list_data if x]

        if len(return_value) == 0:
            import inspect
            from common.function import get_code_line
            raise Exception('argparse convert failed.',
                            get_code_line(inspect.currentframe()))

        return return_value

    @classmethod
    def trace_evaluate_metric(cls, metric_data: ThresholdMetricData):
        """print evaluate metric

        Args:
            metric_data (ThresholdMetricData): ThresholdMetricData
        """
        if isinstance(metric_data, ThresholdMetricData) is False:
            return

        TraceLog().info(f'thresdhold:({metric_data.threshold})')
        if metric_data.metric.confusion.size:
            TraceLog().info("-Confusion Matrix-")
            for matrix in metric_data.metric.confusion:
                TraceLog().info(matrix)

        message = ""
        if metric_data.metric.accuracy:
            message += (f"{metric_data.metric.str_accuracy()}="
                        f"({metric_data.metric.accuracy:.4f}) ")
        if metric_data.metric.precision:
            message += (f"{metric_data.metric.str_precision()}="
                        f"({metric_data.metric.precision:.4f}) ")
        if metric_data.metric.recall:
            message += (f"{metric_data.metric.str_recall()}="
                        f"({metric_data.metric.recall:.4f}) ")
        if metric_data.metric.f1:
            message += (f"{metric_data.metric.str_f1()}="
                    f"({metric_data.metric.f1:.4f}) ")
        if metric_data.metric.roc_auc:
            if isinstance(metric_data.metric.roc_auc, dict):
                message += (f"{metric_data.metric.str_roc_auc()}="
                        f"({str(metric_data.metric.roc_auc)})")
            else:
                message += (f"{metric_data.metric.str_roc_auc()}="
                            f"({metric_data.metric.roc_auc:.4f})")
        TraceLog().info(message)

    @classmethod
    def trace_best_model(cls, score_info:BestModelScoreInfo) -> None:
        """print best model params

        Args:
            best_param (dict): best parameters
        """
        if isinstance(score_info, BestModelScoreInfo) is False:
            return

        if score_info.highest_f1.verify():
            TraceLog().info('= Highest F1-Score =')
            TraceLog().info(f"Model:{score_info.highest_f1.name}")
            TraceLog().info(f"Threshold:{score_info.highest_f1.threshold}")

        return
    
if __name__ == "__main__":

    try:
        TraceLog().info('=========== Start ===========')
        args = parse_commandline()
        PipeLineWrapper(parameters=args)

    except BaseException as e:
        TraceLog().info(f"Error occurred. Message:{e}")

    finally:
        TraceLog().info('===========  End  ===========')
