__all__ = ["app"]

from fastapi import FastAPI

from app.core import Config

from .route import setup_routing
from .util import lifespan

app = FastAPI(
    title=Config.server.title,
    openapi_tags=Config.server.openapi_tags,
    lifespan=lifespan,
)

setup_routing(app)
