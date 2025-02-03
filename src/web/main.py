from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from common.trace_log import TraceLog, get_log_file_name
from common.container.mongo import MongoDBConnection

from web.plugins.redis._redis import redis_plugin, RedisSettings
from web.settings import Settings, get_settings
from web.controller import controller_router
from web.view import view_router
from web.handler import add_exception_handler

TraceLog().initialize(log_file_name=get_log_file_name('web'))

@asynccontextmanager
async def lifespan(app: FastAPI, settings: Settings = get_settings()):
    """start/end"""
    TraceLog().info('Start.')

    if not settings.redis_url:
        TraceLog().info('CELERY_BROKER_URL environment variable not found.')
        raise Exception('CELERY_BROKER_URL environment variable not found.')

    MongoDBConnection().initialize()

    config = RedisSettings(redis_url=settings.redis_url)
    await redis_plugin.init_app(app, config)
    await redis_plugin.init()

    yield

    await redis_plugin.terminate()
    MongoDBConnection().close()

    TraceLog().info('End.')

app = FastAPI(lifespan=lifespan)
# path 끝에 '/' 가 포함될 경우 제거하는 apirouter 적용
app.router.redirect_slashes = False
app.include_router(controller_router)

app.mount(
    path="/static",
    app=StaticFiles(directory="static", html=True),
    name="static")
app.mount('', view_router) # html 처리 관련 path router.

# 커스텀 예외 핸들러 추가
add_exception_handler(app)

if __name__ == "__main__":
    import uvicorn
    port = get_settings().port_number
    uvicorn.run('main:app', host='0.0.0.0', port=int(port), reload=True)
