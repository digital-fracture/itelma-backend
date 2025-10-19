import datetime
from enum import StrEnum
from functools import cached_property

from pydantic import BaseModel, Field, computed_field
from sqlmodel import Field as SQLModelField
from sqlmodel import SQLModel

from app.core import Config

from .analysis import ExaminationVerdict, OverallState
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


class PatientMetadata(BaseModel):
    name: str = ""
    unread: bool = False
    overall_state: OverallState = OverallState.STABLE


class PatientInfo(BaseModel):
    """Clinical information about a patient."""

    parity: str = ""
    pregnancy_course: str = ""
    last_menstrual_period: datetime.date = Field(default_factory=datetime.date.today)
    somatic_diseases: str = ""
    blood_gas: list[BloodGasItem] = Field(default_factory=list)


class PatientBrief(BaseModel):
    id: int
    metadata: PatientMetadata

    ongoing_examination_id: int | None = None


class Patient(PatientBrief):
    info: PatientInfo
    comment: str

    examinations: list[ExaminationBrief] = Field(default_factory=list)
    last_verdict: ExaminationVerdict | None = None


class PatientCreate(BaseModel):
    metadata: PatientMetadata = Field(default_factory=PatientMetadata)
    info: PatientInfo = Field(default_factory=PatientInfo)
    comment: str = ""


class PatientUpdate(BaseModel):
    metadata: PatientMetadata | None = None
    info: PatientInfo | None = None
    comment: str | None = None


class _PatientIdModel(SQLModel):
    id: int | None = SQLModelField(default=None, primary_key=True)


class PatientDb(PatientMetadata, _PatientIdModel, table=True):
    __tablename__ = "patient"
