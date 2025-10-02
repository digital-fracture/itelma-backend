import datetime

from pydantic import BaseModel, Field

from .analysis import ExaminationPartInterval, ExaminationPartStats
from .misc import Channel, PlotPoint


class ExaminationMetadata(BaseModel):
    date: datetime.date
    part_count: int


class ExaminationPredictions(BaseModel):
    pass


class ExaminationBrief(BaseModel):
    id: int
    metadata: ExaminationMetadata


class Examination(ExaminationBrief):
    predictions: ExaminationPredictions = Field(default_factory=ExaminationPredictions)


class ExaminationPartData(BaseModel):
    bpm: list[PlotPoint] = Field(default_factory=list)
    uterus: list[PlotPoint] = Field(default_factory=list)

    def by_channel(self, channel: Channel) -> list[PlotPoint]:
        match channel:
            case Channel.BPM:
                return self.bpm
            case Channel.UTERUS:
                return self.uterus


class ExaminationPart(BaseModel):
    index: int
    data: ExaminationPartData

    stats: ExaminationPartStats = Field(default_factory=ExaminationPartStats)
    # intervals: list[ExaminationPartInterval] = Field(default_factory=list)
    intervals: list[ExaminationPartInterval] = [
        ExaminationPartInterval(start=20, end=25, message="Example 1"),
        ExaminationPartInterval(start=40, end=50, message="Example 2"),
    ]
