import asyncio
from enum import StrEnum

from pydantic import BaseModel

from .analysis import EmulationPrediction, ExaminationPartInterval, ExaminationPartStats
from .examination import ExaminationPartData
from .misc import Channel, MessageModel, PlotPoint


class EmulationStatus(StrEnum):
    SENDING = "sending"
    WAITING_FOR_NEXT_COMMAND = "waiting-for-next-command"
    FINISHED = "finished"
    ABORTED = "aborted"


class EmulationPlot(BaseModel):
    channel: Channel
    point: PlotPoint


class EmulationState(BaseModel):
    status: EmulationStatus
    part_index: int
    sent_part_data: ExaminationPartData

    sent_predictions: list[EmulationPrediction]
    sent_intervals: list[ExaminationPartInterval]
    last_stats: ExaminationPartStats | None = None


class EmulationMessageState(MessageModel):
    message_priority = 0
    state: EmulationState


class EmulationMessageStatus(MessageModel):
    message_priority = 1
    status: EmulationStatus


class EmulationMessageClose(MessageModel):  # signals the end of emulation
    message_priority = 2


class EmulationMessageInterval(MessageModel):
    message_priority = 3
    interval: ExaminationPartInterval


class EmulationMessagePrediction(MessageModel):
    message_priority = 3
    prediction: EmulationPrediction


class EmulationMessageStats(MessageModel):
    message_priority = 3
    stats: ExaminationPartStats


class EmulationMessagePlot(MessageModel):
    message_priority = 4
    plot: EmulationPlot


class EmulationCommand(StrEnum):
    ABORT = "abort"
    NEXT_PART = "next-part"


class EmulationMessageCommand(MessageModel):
    command: EmulationCommand


EmulationMessageOut = (
    EmulationMessageState
    | EmulationMessageStatus
    | EmulationMessageClose
    | EmulationMessageInterval
    | EmulationMessagePrediction
    | EmulationMessageStats
    | EmulationMessagePlot
)
EmulationMessageIn = EmulationMessageCommand


EmulationQueueOut = asyncio.PriorityQueue[EmulationMessageOut]
EmulationQueueIn = asyncio.PriorityQueue[EmulationMessageIn]
