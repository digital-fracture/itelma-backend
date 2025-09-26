import asyncio
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel


class Point(BaseModel):
    time: float
    value: float


class Plot(BaseModel):
    index: int
    point: Point


class Predictions(BaseModel):  # WIP
    ...


class QueueMessage(BaseModel):
    plot: Plot | None = None
    predictions: Predictions | None = None


QueueType = asyncio.Queue[QueueMessage | None]


@dataclass
class Session:
    session_id: str
    files: list[Path]

    queue: QueueType | None = None
    emulation_task_gather: asyncio.Future[list[None]] | None = None
