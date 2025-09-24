__all__ = ["app"]

from fastapi import FastAPI

from app.core import config

from .route import setup_routing
from .util import lifespan

app = FastAPI(title=config.server.title, lifespan=lifespan)

setup_routing(app)
