import asyncio
from enum import StrEnum

from pydantic import BaseModel, Field

from .analysis import EmulationPrediction, ExaminationPartInterval, ExaminationStats
from .examination import ExaminationPlot
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
    sent_part_data: ExaminationPlot = Field(default_factory=ExaminationPlot)

    sent_predictions: list[EmulationPrediction] = Field(default_factory=list)
    sent_intervals: list[ExaminationPartInterval] = Field(default_factory=list)
    last_stats: ExaminationStats | None = None

    part_data_log: ExaminationPlot = Field(default_factory=ExaminationPlot, exclude=True)

    def rotate(self) -> None:
        self.part_data_log = self.total_sent_data()

        self.part_index += 1
        self.sent_part_data = ExaminationPlot()
        self.sent_predictions = []
        self.sent_intervals = []
        self.last_stats = None

    def total_sent_data(self) -> ExaminationPlot:
        return ExaminationPlot.concatenate(self.part_data_log, self.sent_part_data)


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
    stats: ExaminationStats


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
