from enum import StrEnum

PlotPoint = tuple[float, float]


class Channel(StrEnum):
    BPM = "bpm"
    UTERUS = "uterus"
