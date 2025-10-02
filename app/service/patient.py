import logfire

from app.model import Patient, PatientBrief, PatientCreate, PatientUpdate
from app.storage import ExaminationStorage, PatientStorage

from .emulation import EmulationService


class PatientService:
    @staticmethod
    async def create(patient: PatientCreate, patient_id: int | None = None) -> Patient:
        return await PatientStorage.create(patient, patient_id)

    @staticmethod
    async def get_all() -> list[PatientBrief]:
        patients = await PatientStorage.get_all()
        EmulationService.fill_ongoing_examination(patients)

        return patients

    @staticmethod
    @logfire.instrument(record_return=True)
    async def get_by_id(patient_id: int) -> Patient:
        patient = await PatientStorage.get_by_id(patient_id)
        patient.examinations = await ExaminationStorage.get_all_by_patient_id(patient_id)
        EmulationService.fill_ongoing_examination([patient])

        return patient

    @staticmethod
    async def update_by_id(patient_id: int, patient_update: PatientUpdate) -> None:
        await PatientStorage.update_by_id(patient_id, patient_update)
