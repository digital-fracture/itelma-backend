from enum import StrEnum
from types import NoneType
from typing import Self

from pydantic import BaseModel, model_validator

from app.util import CustomAsyncioPriorityQueue

from .examination import ExaminationPartData
from .misc import Channel, PlotPoint

MultipleMessageTypesError = ValueError("Exactly one message type must be present")


class EmulationPlot(BaseModel):
    channel: Channel
    point: PlotPoint


class EmulationPrediction(BaseModel):  # TODO: WIP
    dummy: str = "placeholder"


class EmulationMessageOutStatus(StrEnum):
    SENDING = "sending"
    WAITING_FOR_NEXT_COMMAND = "waiting-for-next-command"
    FINISHED = "finished"
    ABORTED = "aborted"


class EmulationMessageInitial(BaseModel):
    last_status: EmulationMessageOutStatus
    current_part_index: int
    sent_part_data: ExaminationPartData
    sent_predictions: list[EmulationPrediction]


class EmulationMessageOut(BaseModel):
    status: EmulationMessageOutStatus | None = None
    plot: EmulationPlot | None = None
    prediction: EmulationPrediction | None = None

    @model_validator(mode="after")
    def ensure_one_field(self) -> Self:
        if len(self.model_fields_set) == 1:
            return self

        raise MultipleMessageTypesError


class EmulationMessageInCommand(StrEnum):
    ABORT = "abort"  # abort the emulation
    NEXT_PART = "next-part"  # move to the next part


class EmulationMessageIn(BaseModel):
    command: EmulationMessageInCommand


EmulationMessageOutUnion = EmulationMessageInitial | EmulationMessageOut | None


class EmulationQueueOut(CustomAsyncioPriorityQueue[EmulationMessageOutUnion]):
    type_priority_order = (
        EmulationMessageInitial,
        (EmulationMessageIn, NoneType),
    )


class EmulationQueueIn(CustomAsyncioPriorityQueue[EmulationMessageIn]):
    pass
