__all__ = ["Lock", "LockManager"]

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import ClassVar

from app.util import AsyncRWLock


class LockManager:
    """Manages ``AsyncRWLock`` objects per key. Safe to call concurrently."""

    _locks: ClassVar[dict[str, AsyncRWLock]] = {}
    _global_lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    @classmethod
    async def _get_lock(cls, key: str) -> AsyncRWLock:
        async with cls._global_lock:
            if key not in cls._locks:
                cls._locks[key] = AsyncRWLock()
            return cls._locks[key]

    @classmethod
    @asynccontextmanager
    async def read(cls, key: str) -> AsyncGenerator[None]:
        lock = await cls._get_lock(key)
        async with lock.read():
            yield

    @classmethod
    @asynccontextmanager
    async def write(cls, key: str) -> AsyncGenerator[None]:
        lock = await cls._get_lock(key)
        async with lock.write():
            yield

    @classmethod
    @asynccontextmanager
    async def read_many(cls, keys: list[str]) -> AsyncGenerator[None]:
        keys_sorted = sorted(set(keys))
        locks = [await cls._get_lock(key) for key in keys_sorted]

        ctxs = [lock.read() for lock in locks]
        await asyncio.gather(*(ctx.__aenter__() for ctx in ctxs))

        try:
            yield

        finally:
            await asyncio.gather(*(ctx.__aexit__(None, None, None) for ctx in reversed(ctxs)))

    @classmethod
    @asynccontextmanager
    async def write_many(cls, keys: list[str]) -> AsyncGenerator[None]:
        keys_sorted = sorted(set(keys))
        locks = [await cls._get_lock(key) for key in keys_sorted]

        ctxs = [lock.write() for lock in locks]
        # writer locks are serializing so acquire sequentially to avoid weird ordering
        for ctx in ctxs:
            await ctx.__aenter__()

        try:
            yield

        finally:
            for ctx in reversed(ctxs):
                await ctx.__aexit__(None, None, None)


class Lock:
    patient_list = "patients"

    @staticmethod
    @lru_cache
    def patient(patient_id: int) -> str:
        return f"patients:{patient_id}"

    @staticmethod
    @lru_cache
    def patient_examination_list(patient_id: int) -> str:
        return f"patients:{patient_id}:examinations"

    @staticmethod
    @lru_cache
    def patient_examination(patient_id: int, examination_id: int) -> str:
        return f"patients:{patient_id}:examinations:{examination_id}"
