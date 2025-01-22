from enum import Enum
from typing import TypeVar, Callable, Iterator, Any

T = TypeVar('F', bound=Callable[..., Any])
def singleton(class_: T) -> T:
    """singleton pattern."""
    class class_w(class_):
        _instance = None
        def __new__(class_, *args, **kwargs):
            """new."""
            if class_w._instance is None:
                class_w._instance = super(
                    class_w, class_).__new__(class_)
                class_w._instance._sealed = False
            return class_w._instance

        def __init__(self, *args, **kwargs) -> None:
            """init."""
            if self._sealed:
                return
            super(class_w, self).__init__(*args, **kwargs)
            self._sealed = True
    class_w.__name__ = class_.__name__
    return class_w

def timeit(method):
    """함수 종료 시간 측정"""
    import functools
    @functools.wraps(method)
    def timed(*args, **kwargs):
        import time
        start_time = time.time()
        result = method(*args, **kwargs)
        end_time = time.time()

        from common.trace_log import TraceLog
        TraceLog().info(f"'{method.__name__}' "
                        f"Time : {(end_time - start_time):.4f} seconds")
        return result
    return timed

def iter_files_in_folder(folder_path: str) -> Iterator[str]:
    """폴더 하위 전체 파일 iteration

    Args:
        folder_path (str): iteration할 폴더 경로

    Returns:
        Iterator: file path
    """
    import os
    for root, _, files in os.walk(folder_path):
        for file in files:
            yield os.path.join(root, file)

class conditional_decorator():
    """조건부 데코레이터 클래스"""
    def __init__(self, decorator_method, condition):
        """생성자

        Args:
            decorator_method (Method Type): 호출할 함수
            condition (Boolean): 실제로 함수를 호출할지 말지 결정할 Flag
        """
        self.decorator = decorator_method
        self.condition = condition

    def __call__(self, func):
        """데코레이터 함수 호출"""
        if not self.condition:
            # Return the function unchanged, not decorated.
            return func
        return self.decorator(func)

def get_code_line(frame):
    """해당 frame의 파일명, 코드라인 얻기

    Args:
        frame (inspect.currentframe): inspect 객체의 currentframe() 호출 결과

    Returns:
        string: 코드 위치, 코드 라인
    """
    # __FILE__
    file_name = frame.f_code.co_filename
    # __LINE__
    file_line = frame.f_lineno

    return file_name, file_line

class StrEnum(str, Enum):
    """string enum class"""
    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def __next__(self):
        return self.name

class TempDir:
    """임시폴더 클래스"""
    def __init__(
        self, prefix:str='', chdr:bool=False, remove_on_exit:bool=True):
        """초기화.

        Args:
            prefix (str, optional): 임시폴더 prefix.
            chdr (bool, optional): SetCurrentDirectory 지정.
            remove_on_exit (bool, optional): with scope 빠져나올 때 폴더 삭제.
        """
        self._dir = None
        self._path = None
        self._prefix = prefix
        self._chdr = chdr
        self._remove = remove_on_exit

    def __enter__(self):
        """with scope 진입 시"""
        import tempfile
        import os
        self._path = os.path.abspath(tempfile.mkdtemp(prefix=self._prefix))
        assert os.path.exists(self._path)
        if self._chdr:
            self._dir = os.path.abspath(os.getcwd())
            os.chdir(self._path)
        return self

    def __exit__(self, tp, val, traceback):
        """with scope 빠져나올 때"""
        import os
        if self._chdr and self._dir:
            os.chdir(self._dir)
            self._dir = None
        if self._remove and os.path.exists(self._path):
            import shutil
            shutil.rmtree(self._path)

        assert not self._remove or not os.path.exists(self._path)
        assert os.path.exists(os.getcwd())

    def path(self, *path):
        """경로"""
        import os
        return os.path.join("./", *path)\
            if self._chdr else os.path.join(self._path, *path)
