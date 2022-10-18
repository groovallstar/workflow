import os
from functools import lru_cache

from pydantic import BaseSettings

class Settings(BaseSettings):
    """global settings"""
    redis_subscribe_channel: str=os.environ.get('REDIS_SUBSCRIBE_CHANNEL', '')
    redis_url: str=os.environ.get('CELERY_BROKER_URL', '')
    port_number: str=os.environ.get('FASTAPI_PORT', '')

@lru_cache()
def get_settings():
    """return settings object"""
    return Settings()
