import asyncio
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel


class GraphPoint(BaseModel):
    time: float
    value: float


class Predictions(BaseModel):  # WIP
    ...


class QueueMessage(BaseModel):
    graph_point: GraphPoint
    predictions: Predictions | None = None


QueueType = asyncio.Queue[QueueMessage | None]


@dataclass
class Session:
    session_id: str
    file_path: Path
    queue: QueueType | None = None
    emulation_task: asyncio.Task[None] | None = None
