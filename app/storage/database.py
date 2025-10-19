from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import logfire
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core import Paths

_engine = create_async_engine(Paths.storage.patients_db_url)
logfire.instrument_sqlalchemy(_engine)


async def initialize_database() -> None:
    async with _engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)


async def dispose_database() -> None:
    await _engine.dispose()


async def spawn_session() -> AsyncGenerator[AsyncSession]:
    async with AsyncSession(_engine) as session:
        yield session
        await session.commit()


async def spawn_readonly_session() -> AsyncGenerator[AsyncSession]:
    async with AsyncSession(_engine) as session:
        yield session
        await session.rollback()


async def spawn_session_with_transaction() -> AsyncGenerator[AsyncSession]:
    async with AsyncSession(_engine) as session, session.begin():
        yield session


start_session = asynccontextmanager(spawn_session)
start_readonly_session = asynccontextmanager(spawn_readonly_session)
start_transaction = asynccontextmanager(spawn_session_with_transaction)
