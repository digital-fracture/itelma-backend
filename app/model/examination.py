import datetime
from functools import cached_property
from typing import Self

from pydantic import BaseModel, Field

from .analysis import ExaminationPartInterval, ExaminationStats, ExaminationVerdict
from .misc import Channel, PlotPoint


class ExaminationMetadata(BaseModel):
    date: datetime.date
    part_count: int


class ExaminationBrief(BaseModel):
    id: int
    metadata: ExaminationMetadata


class Examination(ExaminationBrief):
    stats: ExaminationStats = Field(default_factory=ExaminationStats)
    verdict: ExaminationVerdict = Field(default_factory=ExaminationVerdict)


class ExaminationPlot(BaseModel):
    bpm: list[PlotPoint] = Field(default_factory=list)
    uterus: list[PlotPoint] = Field(default_factory=list)

    @cached_property
    def last_timestamp(self) -> float:
        last_bpm = self.bpm[-1][0] if self.bpm else 0
        last_uterus = self.uterus[-1][0] if self.uterus else 0
        return max(last_bpm, last_uterus)

    def by_channel(self, channel: Channel) -> list[PlotPoint]:
        match channel:
            case Channel.BPM:
                return self.bpm
            case Channel.UTERUS:
                return self.uterus

    @classmethod
    def concatenate(cls, *plots: Self) -> Self:
        result_bpm: list[PlotPoint] = []
        result_uterus: list[PlotPoint] = []
        last_timestamp = 0.0

        for plot in plots:
            shifted_bpm = [(last_timestamp + time, value) for (time, value) in plot.bpm]
            shifted_uterus = [(last_timestamp + time, value) for (time, value) in plot.uterus]

            result_bpm.extend(shifted_bpm)
            result_uterus.extend(shifted_uterus)
            last_timestamp += plot.last_timestamp

        return cls(bpm=result_bpm, uterus=result_uterus)


class ExaminationPart(BaseModel):
    index: int
    data: ExaminationPlot

    intervals: list[ExaminationPartInterval] = Field(default_factory=list)
