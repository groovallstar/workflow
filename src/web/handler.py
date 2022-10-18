import json
from typing import Callable, Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from starlette.requests import Request as StarLetteRequest
from starlette.responses import Response
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp

from fastapi import FastAPI, Request, APIRouter, status
from fastapi import Header
from fastapi.types import DecoratedCallable
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder

from common.trace_log import TraceLog

class PrettyJSONResponse(Response):
    """JSON Pretty Print Class"""
    media_type = "application/json"
    def render(self, content: Any) -> bytes:
        """pretty json 포멧으로 변환"""
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            separators=(", ", ": "),
        ).encode("utf-8")

class JsonException(Exception):
    """HTTP Exception을 json 형식으로 변환 클래스"""
    def __init__(self, content: dict, status_code: status):
        """initialize"""
        self.content = content
        self.status_code = status_code

class LimitUploadSize(BaseHTTPMiddleware):
    """Check Content Length Before Request"""
    def __init__(self, app: ASGIApp, max_upload_size: int) -> None:
        super().__init__(app)
        self.max_upload_size = max_upload_size

    async def dispatch(
        self, request: StarLetteRequest, call_next: RequestResponseEndpoint
    ) -> Response:
        """Dispath Method"""
        # post method의 content-length 제한.
        if ((request.method == 'POST') and
            ('content-length' in request.headers)):
            content_length = int(request.headers['content-length'])
            if content_length > self.max_upload_size:
                return PrettyJSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content=jsonable_encoder({
                        'status_code': status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        'status': 'File too large'}))
        return await call_next(request)

class EndSlashRemoveRouter(APIRouter):
    """'/'가 path에 있을경우 제거 후 route 하는 클래스"""
    def api_route(
        self, path: str, *, include_in_schema: bool = True, **kwargs: Any
    ) -> Callable[[DecoratedCallable], DecoratedCallable]:
        """path dispatch"""
        if path.endswith("/"):
            path = path[:-1]

        add_path = super().api_route(
            path, include_in_schema=include_in_schema, **kwargs)

        alternate_path = path + "/"
        add_alternate_path = super().api_route(
            alternate_path, include_in_schema=False, **kwargs)

        def decorator(func: DecoratedCallable) -> DecoratedCallable:
            add_alternate_path(func)
            return add_path(func)

        return decorator

async def content_length_limit(
    content_length: int=Header(...)):
    """Content-Length 사이즈 체크 함수
       Dependency Injection 으로 사용 가능

    Args:
        content_length (int, optional): limit 사이즈.

    Raises:
        JsonException: Depends 를 method 파라미터로 사용 시에는 예외를 일으켜야함

    Returns:
        int: Content-Length
    """
    if content_length > 10 * 1024 * 1024:
        raise JsonException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content=jsonable_encoder({
                'status_code': status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                'status': 'File too large'}))
    return content_length

def add_exception_handler(app: FastAPI) -> None:
    """예외 핸들러 등록.

    Args:
        app (FastAPI): FastAPI app 객체
    """
    @app.exception_handler(JsonException)
    async def json_exception_handler(request: Request, ex: JsonException):
        """Json Exception 핸들러"""
        return PrettyJSONResponse(
            status_code=ex.status_code,
            content=jsonable_encoder(ex.content))

    @app.exception_handler(StarletteHTTPException)
    async def starlette_http_exception_handler(
        request: Request, exc: StarletteHTTPException):
        """Starlette HTTP Exception 핸들러"""
        TraceLog().info(
            f"http_exception "
            f"status_code=({exc.status_code}), status=({exc.detail})")
        return PrettyJSONResponse(
            status_code=exc.status_code,
            content=jsonable_encoder({
                    'status_code': exc.status_code,
                    'status': exc.detail}))

    @app.exception_handler(RequestValidationError)
    async def request_validation_exception_handler(
        request: Request, exc: RequestValidationError):
        """Pydantic Type Validation Error 핸들러"""
        TraceLog().info(
            f"validation_exception exc=({str(exc)})")
        return PrettyJSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=jsonable_encoder({
                'status_code': status.HTTP_500_INTERNAL_SERVER_ERROR,
                'status': 'Internal Server Error'}))
