import datetime
import logging
import logging.handlers

from common.function import singleton

def init_log_object(log_file_name: str, logger_name: str = __name__):
    """Initialize logging object.
    각각 파일별로 로깅할 경우 name 지정 필요함.

    Args:
        log_file_name (str): Write Log File Path.
    """
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_file_name,
        when='W0', backupCount=10,
        encoding='utf-8', atTime=datetime.time(0, 0, 0))
    file_handler.setFormatter(formatter)
    file_handler.suffix = '%Y%m%d'
    file_handler.setLevel(logging.INFO)

    log = logging.getLogger(logger_name)
    log.setLevel(logging.INFO)

    import sys
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    # 동일 객체로 다시 초기화 할 때 기존 핸들러 설정 초기화.
    log.handlers.clear()
    log.addHandler(file_handler)
    log.addHandler(stream_handler)

    return log

def get_log_file_name(file_name: str = ''):
    """파일 로그 경로 생성.
       PYTHONPATH 환경 변수가 없을 경우 assert

    Args:
        file_name (str, optional): 생성할 파일 이름. Defaults to __file__.

    Returns:
        str: 로그 파일 경로명.
    """
    if file_name == '':
        file_name = __file__

    import os
    import tempfile
    log_file = os.path.join(tempfile.gettempdir(), os.path.splitext(
        os.path.basename(file_name))[0] + '.log')
    return log_file

@singleton
class TraceLog:
    """TraceLog Class"""
    log_object = None

    @classmethod
    def initialize(
        cls, log_file_name: str = "", logger_name: str = __name__) -> None:
        """명시적으로 초기화 할 경우 file logging 및 stream logging 기능 활성.
        Args:
            log_file_name (str, optional): 로그 파일명. Defaults to "".
            logger_name (str, optional): 파일별로 따로 로깅하기 위해서는
                                         이름을 지정 해야함.
        """
        # 파일명 없이 초기화 할 경우 script.log 파일로 지정됨.
        if log_file_name == "":
            log_file_name = get_log_file_name()
        cls.initialize_with_log_file_name(
            log_file_name=log_file_name, logger_name=logger_name)

    @classmethod
    def initialize_with_log_file_name(
        cls, log_file_name: str, logger_name: str=__name__) -> None:
        """로그 파일명을 지정하여 초기화.

        Args:
            log_file_name (str): 로그 파일명.
            logger_name (str, optional): 파일별로 따로 로깅하기 위해서는
                                         이름을 지정 해야함.
        """
        cls.log_object = init_log_object(log_file_name, logger_name)

    @classmethod
    def info(cls, message: str):
        """Write INFO message or print message.

        Args:
            message (str): Log Message.
        """
        if cls.log_object:
            cls.log_object.info(message)
        else:
            print(message)

if __name__ == '__main__':

    '''sample'''
    # TraceLog().initialize(log_file_name='test1.log', logger_name='test1')
    # TraceLog().info('test1')
    # test2_log = init_log_object(log_file_name='test2.log', logger_name='test2')
    # test2_log.info('test2')
