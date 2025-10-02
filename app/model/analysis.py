from pydantic import BaseModel


class ExaminationPartStats(BaseModel):
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

    condition: str = "Норма"


class ExaminationPartInterval(BaseModel):
    start: float
    end: float
    message: str


class EmulationPrediction(BaseModel):
    messages: list[str]
    bpm_average: float
    bpm_min: float
