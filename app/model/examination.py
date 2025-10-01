import datetime

from pydantic import BaseModel, Field

from .misc import Channel, PlotPoint


class ExaminationMetadata(BaseModel):
    date: datetime.date
    part_count: int


class ExaminationPredictions(BaseModel):  # TODO: WIP
    dummy: str = "placeholder"


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
