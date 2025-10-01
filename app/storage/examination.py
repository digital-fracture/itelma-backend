import asyncio
import datetime
from pathlib import Path

import aiofiles
import aiofiles.os
import logfire
from async_unzip.unzipper import unzip  # type: ignore
from fastapi import UploadFile

from app.core import Paths
from app.core.exceptions import ExaminationNotFoundError, ExaminationPartNotFoundError
from app.model import (
    Examination,
    ExaminationBrief,
    ExaminationMetadata,
    ExaminationPartData,
    ExaminationPredictions,
    PlotPoint,
)

from . import util
from .locking import Lock, LockManager
from .patient import PatientStorage


class ExaminationStorage:
    @staticmethod
    @logfire.instrument(record_return=True)
    async def check_exists(patient_id: int, examination_id: int, *, throw: bool = True) -> bool:
        if not (await PatientStorage.check_exists(patient_id, throw=throw)):
            return False

        async with LockManager.read(Lock.patient_examination_list(patient_id)):
            if str(examination_id) not in (
                await aiofiles.os.listdir(Paths.all_examinations_dir(patient_id))
            ):
                if throw:
                    raise ExaminationNotFoundError(patient_id, examination_id)
                return False

        return True

    @staticmethod
    @logfire.instrument(record_return=True)
    async def save_uploaded(patient_id: int, examination_upload_file: UploadFile) -> Examination:
        async with LockManager.write(Lock.patient_examination_list(patient_id)):
            examination_id = max(
                map(int, await aiofiles.os.listdir(Paths.all_examinations_dir(patient_id)))
            )
            await aiofiles.os.makedirs(
                examination_dir := Paths.examination_dir(patient_id, examination_id)
            )

        zip_path = await util.save_temp_file(examination_upload_file)

        async with LockManager.write(Lock.patient_examination(patient_id, examination_id)):
            await unzip(zip_path, examination_dir)

            metadata = ExaminationMetadata(
                date=datetime.datetime.now(tz=datetime.UTC).astimezone(),
                part_count=len(
                    await aiofiles.os.listdir(Paths.examination_bpm_dir(patient_id, examination_id))
                ),
            )
            await util.dump_yaml(
                metadata.model_dump(), Paths.examination_metadata_file(patient_id, examination_id)
            )
            await util.create_empty(Paths.examination_predictions_file(patient_id, examination_id))

        return Examination(id=examination_id, metadata=metadata)

    @classmethod
    @logfire.instrument(record_return=True)
    async def get_all_by_patient_id(cls, patient_id: int) -> list[ExaminationBrief]:
        await PatientStorage.check_exists(patient_id)

        async with LockManager.read(Lock.patient_examination_list(patient_id)):
            examinations_dir = Paths.all_examinations_dir(patient_id)

            return await asyncio.gather(
                *(
                    cls._get_brief(Path(dir_entry))
                    for dir_entry in await aiofiles.os.scandir(examinations_dir)
                    if dir_entry.is_dir() and dir_entry.name.isdigit()
                )
            )

    @staticmethod
    async def _get_brief(examination_dir: Path) -> ExaminationBrief:
        examination_id = int(examination_dir.name)
        metadata_path = examination_dir / Paths.metadata_file_name

        raw_metadata = await util.load_yaml(metadata_path)
        return ExaminationBrief(
            id=examination_id, metadata=ExaminationMetadata.model_validate(raw_metadata)
        )

    @classmethod
    @logfire.instrument(record_return=True)
    async def get_by_id(cls, patient_id: int, examination_id: int) -> Examination:
        await cls.check_exists(patient_id, examination_id)

        async with LockManager.read(Lock.patient_examination(patient_id, examination_id)):
            raw_metadata = await util.load_yaml(
                Paths.examination_metadata_file(patient_id, examination_id)
            )
            raw_predictions = await util.load_yaml(
                Paths.examination_predictions_file(patient_id, examination_id)
            )

        return Examination(
            id=patient_id,
            metadata=ExaminationMetadata.model_validate(raw_metadata),
            predictions=ExaminationPredictions.model_validate(raw_predictions),
        )

    @classmethod
    @logfire.instrument
    async def read_part_data(
        cls, patient_id: int, examination_id: int, part_index: int
    ) -> ExaminationPartData:
        await cls.check_exists(patient_id, examination_id)

        bpm_path = Paths.examination_part_bpm_file(patient_id, examination_id, part_index)
        uterus_path = Paths.examination_part_uterus_file(patient_id, examination_id, part_index)

        if not all(
            await asyncio.gather(
                aiofiles.os.path.exists(bpm_path), aiofiles.os.path.exists(uterus_path)
            )
        ):
            raise ExaminationPartNotFoundError(patient_id, examination_id, part_index)

        async with asyncio.TaskGroup() as tg:
            bpm_task = tg.create_task(cls._read_csv(bpm_path))
            uterus_task = tg.create_task(cls._read_csv(uterus_path))

        return ExaminationPartData(bpm=bpm_task.result(), uterus=uterus_task.result())

    @staticmethod
    async def _read_csv(path: Path) -> list[PlotPoint]:
        async with aiofiles.open(path) as file:
            await file.readline()  # skip csv header

            points = []
            async for line in file:
                timestamp, value = map(float, line.strip().split(","))
                points.append((timestamp, value))

        return points
