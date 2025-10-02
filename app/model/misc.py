from enum import StrEnum
from typing import ClassVar, Self

from pydantic import BaseModel

PlotPoint = tuple[float, float]


class Channel(StrEnum):
    BPM = "bpm"
    UTERUS = "uterus"


class MessageModel(BaseModel):
    message_priority: ClassVar[int] = 0

    def __lt__(self, other: Self) -> bool:
        return self.message_priority < other.message_priority
