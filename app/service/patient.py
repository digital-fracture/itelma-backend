from collections.abc import Sequence

import logfire

# noinspection PyProtectedMember
from sqlmodel.sql._expression_select_cls import SelectBase

from app.model import (
    Patient,
    PatientBrief,
    PatientCreate,
    PatientDb,
    PatientMetadata,
    PatientUpdate,
)
from app.storage import ExaminationStorage, PatientStorage

from .emulation import EmulationService


class PatientService:
    @staticmethod
    async def create(patient: PatientCreate, patient_id: int | None = None) -> Patient:
        return await PatientStorage.create(patient, patient_id)

    @staticmethod
    def get_all_statement() -> SelectBase[PatientDb]:
        return PatientStorage.get_all_statement()

    @staticmethod
    async def transform_all_for_pagination(patient_dbs: Sequence[PatientDb]) -> list[PatientBrief]:
        """Return all patients sorted by ID descendingly."""
        patients = [
            PatientBrief(id=patient_db.id, metadata=PatientMetadata.model_validate(patient_db))
            for patient_db in patient_dbs
        ]
        EmulationService.fill_ongoing_examination(patients)

        return patients

    @staticmethod
    @logfire.instrument(record_return=True)
    async def get_by_id(patient_id: int) -> Patient:
        """Return patient with examinations sorted by ID descendingly."""
        patient = await PatientStorage.get_by_id(patient_id)
        patient.examinations = sorted(
            await ExaminationStorage.get_all_by_patient_id(patient_id),
            key=lambda examination: examination.id,
            reverse=True,
        )

        if patient.examinations:
            last_examination_brief = patient.examinations[0]
            last_examination_full = await ExaminationStorage.get_by_id(
                patient_id, last_examination_brief.id
            )
            patient.last_verdict = last_examination_full.verdict

        EmulationService.fill_ongoing_examination([patient])

        return patient

    @staticmethod
    async def update_by_id(patient_id: int, patient_update: PatientUpdate) -> None:
        await PatientStorage.update_by_id(patient_id, patient_update)
