from enum import StrEnum

from pydantic import BaseModel, Field


class OverallState(StrEnum):
    STABLE = "стабильное состояние"
    ATTENTION = "требуется внимание"
    CRITICAL = "критическое состояние"


class ExaminationStats(BaseModel):
    bpm_average: float = 0
    uterus_average: float = 0

    acceleration_count: int = 0
    deceleration_count: int = 0
    late_deceleration_count: int = 0
    early_deceleration_count: int = 0
    variable_deceleration_count: int = 0

    mild_tachycardia_seconds: float = 0
    severe_tachycardia_seconds: float = 0
    mild_bradycardia_seconds: float = 0
    severe_bradycardia_seconds: float = 0

    condition: str = "Нормальное состояние"


class ExaminationPartInterval(BaseModel):
    start: float
    end: float
    message: str


class EmulationPrediction(BaseModel):
    messages: list[str]
    bpm_average: float
    bpm_min: float


class ExaminationVerdict(BaseModel):
    overall_status: OverallState = OverallState.STABLE
    recommendations: list[str] = Field(default_factory=list)
    attention_zones: list[str] = Field(default_factory=list)
    risk_zones: list[str] = Field(default_factory=list)


class PipelineResult(BaseModel):
    prediction: EmulationPrediction | None
    intervals: list[ExaminationPartInterval]
    stats: ExaminationStats
    verdict: ExaminationVerdict | None
