import logfire
from fastapi import UploadFile

from app.model import Examination, ExaminationPart, PatientCreate
from app.storage import ExaminationStorage, PatientStorage


class ExaminationService:
    @staticmethod
    @logfire.instrument(record_return=True)
    async def create(patient_id: int, examination_upload_file: UploadFile) -> Examination:
        if not (await PatientStorage.check_exists(patient_id, throw=False)):
            await PatientStorage.create(PatientCreate(), patient_id=patient_id)

        return await ExaminationStorage.save_uploaded(patient_id, examination_upload_file)

    @staticmethod
    async def get_by_id(patient_id: int, examination_id: int) -> Examination:
        return await ExaminationStorage.get_by_id(patient_id, examination_id)

    @staticmethod
    @logfire.instrument
    async def get_part_data(
        patient_id: int, examination_id: int, part_index: int
    ) -> ExaminationPart:
        return ExaminationPart(
            index=part_index,
            data=await ExaminationStorage.read_part_data(patient_id, examination_id, part_index),
        )
