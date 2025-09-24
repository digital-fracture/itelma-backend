__all__ = ["v1_router"]

from fastapi import APIRouter

from .emulation import emulation_router

v1_router = APIRouter(prefix="/v1")

v1_router.include_router(emulation_router)
