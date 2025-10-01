from typing import ClassVar

import aiofiles
import aiofiles.os
import logfire

from app.core import Paths
from app.core.exceptions import PatientNotFoundError
from app.model import (
    Patient,
    PatientBrief,
    PatientCreate,
    PatientInfo,
    PatientPredictions,
    PatientUpdate,
)

from . import util
from .locking import Lock, LockManager


class PatientStorage:
    _id_to_name: ClassVar[dict[int, str]]

    @classmethod
    @logfire.instrument
    async def initialize(cls) -> None:
        async with LockManager.read(Lock.patient_list):
            cls._id_to_name = await util.load_yaml(Paths.patient_names_file)

    @classmethod
    @logfire.instrument(record_return=True)
    async def check_exists(cls, patient_id: int, *, throw: bool = True) -> bool:
        async with LockManager.read(Lock.patient_list):
            if patient_id not in cls._id_to_name:
                if throw:
                    raise PatientNotFoundError(patient_id)
                return False

        return True

    @classmethod
    @logfire.instrument(record_return=True)
    async def create(cls, patient: PatientCreate, patient_id: int | None = None) -> Patient:
        if patient_id is None:
            async with LockManager.read(Lock.patient_list):
                patient_id = max(cls._id_to_name.keys(), default=0) + 1

        async with LockManager.write(Lock.patient_list):
            cls._id_to_name[patient_id] = patient.name
            await util.dump_yaml(cls._id_to_name, Paths.patient_names_file)

        async with LockManager.write(Lock.patient(patient_id)):
            await aiofiles.os.makedirs(Paths.patient_dir(patient_id))
            await aiofiles.os.makedirs(Paths.all_examinations_dir(patient_id))

            await util.dump_yaml(patient.info.model_dump(), Paths.patient_info_file(patient_id))
            await util.create_empty(Paths.patient_predictions_file(patient_id))

        return Patient(id=patient_id, name=patient.name, info=patient.info)

    @classmethod
    @logfire.instrument(record_return=True)
    async def get_all(cls) -> list[PatientBrief]:
        async with LockManager.read(Lock.patient_list):
            return [PatientBrief(id=p_id, name=name) for p_id, name in cls._id_to_name.items()]

    @classmethod
    @logfire.instrument(record_return=True)
    async def get_by_id(cls, patient_id: int) -> Patient:
        await cls.check_exists(patient_id)

        async with LockManager.read(Lock.patient_list):
            name = cls._id_to_name[patient_id]

        async with LockManager.read(Lock.patient(patient_id)):
            raw_info = await util.load_yaml(Paths.patient_info_file(patient_id))
            raw_predictions = await util.load_yaml(Paths.patient_predictions_file(patient_id))

        return Patient(
            id=patient_id,
            name=name,
            info=PatientInfo.model_validate(raw_info),
            predictions=PatientPredictions.model_validate(raw_predictions),
        )

    @classmethod
    @logfire.instrument
    async def update_by_id(cls, patient_id: int, patient_update: PatientUpdate) -> None:
        await cls.check_exists(patient_id)

        if patient_update.name is not None:
            async with LockManager.write(Lock.patient_list):
                cls._id_to_name[patient_id] = patient_update.name
                await util.dump_yaml(cls._id_to_name, Paths.patient_names_file)

        if patient_update.info is not None:
            async with LockManager.write(Lock.patient(patient_id)):
                info_path = Paths.patient_info_file(patient_id)

                old_info = PatientInfo.model_validate(await util.load_yaml(info_path))
                updated_info = old_info.model_copy(
                    update=patient_update.info.model_dump(exclude_unset=True)
                )

                await util.dump_yaml(updated_info.model_dump(), info_path)
