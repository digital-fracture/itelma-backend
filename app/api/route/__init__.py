__all__ = ["setup_routing"]

from fastapi import APIRouter, FastAPI
from fastapi.responses import RedirectResponse

from .exceptions import register_exception_handlers
from .middlewares import register_middlewares
from .v1 import v1_router

root_router = APIRouter()

root_router.include_router(v1_router)


@root_router.get("/", include_in_schema=False)
async def docs_redirect() -> RedirectResponse:
    return RedirectResponse(url="/docs")


def setup_routing(app: FastAPI) -> None:
    register_exception_handlers(app)
    register_middlewares(app)
    app.include_router(root_router)
