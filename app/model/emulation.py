import asyncio
from enum import StrEnum

from pydantic import BaseModel, Field

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
    sent_part_data: ExaminationPartData = Field(default_factory=ExaminationPartData)

    sent_predictions: list[EmulationPrediction] = Field(default_factory=list)
    sent_intervals: list[ExaminationPartInterval] = Field(default_factory=list)
    last_stats: ExaminationPartStats | None = None

    part_data_log: ExaminationPartData = Field(default_factory=ExaminationPartData, exclude=True)
    log_last_timestamp: float = Field(default=0, exclude=True)

    def rotate(self) -> None:
        for channel in Channel:
            self.part_data_log.by_channel(channel).extend(
                self.shifted_plot(self.sent_part_data.by_channel(channel))
            )
        self.log_last_timestamp = max(
            self.part_data_log.by_channel(channel)[-1][0] for channel in Channel
        )

        self.part_index += 1
        self.sent_part_data = ExaminationPartData()
        self.sent_predictions = []
        self.sent_intervals = []
        self.last_stats = None

    def shifted_plot(self, plot: list[PlotPoint]) -> list[PlotPoint]:
        return [(self.log_last_timestamp + time, value) for (time, value) in plot]


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
