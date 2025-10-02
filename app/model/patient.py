import datetime
from enum import StrEnum
from functools import cached_property

from pydantic import BaseModel, Field, computed_field

from app.core import Config

from .examination import ExaminationBrief


class Normal(StrEnum):
    BELOW = "ниже нормы"
    NORMAL = "в норме"
    ABOVE = "выше нормы"


class BloodGasItem(BaseModel):
    name: str
    value: float

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def unit(self) -> str:
        return (
            Config.medical.blood_gas[self.name.lower()].unit
            if self.name.lower() in Config.medical.blood_gas
            else ""
        )

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def normal(self) -> Normal:
        if self.name not in Config.medical.blood_gas:
            return Normal.NORMAL

        gas_config = Config.medical.blood_gas[self.name]

        if self.value < gas_config.normal_min:
            return Normal.BELOW
        if self.value > gas_config.normal_max:
            return Normal.ABOVE
        return Normal.NORMAL


class PatientInfo(BaseModel):
    """Clinical information about a patient."""

    parity: str = ""
    pregnancy_course: str = ""
    last_menstrual_period: datetime.date = Field(default_factory=datetime.date.today)
    somatic_diseases: str = ""
    blood_gas: list[BloodGasItem] = Field(default_factory=list)


class PatientPredictions(BaseModel):
    pass


class PatientBrief(BaseModel):
    id: int
    name: str = ""
    ongoing_examination_id: int | None = None


class Patient(PatientBrief):
    info: PatientInfo = Field(default_factory=PatientInfo)
    predictions: PatientPredictions = Field(default_factory=PatientPredictions)

    examinations: list[ExaminationBrief] = Field(default_factory=list)


class PatientCreate(BaseModel):
    name: str = ""
    info: PatientInfo = Field(default_factory=PatientInfo)


class PatientUpdate(BaseModel):
    name: str | None = None
    info: PatientInfo | None = None
