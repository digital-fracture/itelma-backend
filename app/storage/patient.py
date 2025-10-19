import aiofiles.os
import logfire
from sqlmodel import select

# noinspection PyProtectedMember
from sqlmodel.sql._expression_select_cls import SelectBase

from app.core import Paths
from app.core.exceptions import PatientNotFoundError
from app.model import (
    Patient,
    PatientCreate,
    PatientDb,
    PatientInfo,
    PatientMetadata,
    PatientUpdate,
)

from . import util
from .database import start_readonly_session, start_transaction
from .locking import Lock, LockManager


# noinspection Pydantic
class PatientStorage:
    @classmethod
    def get_all_statement(cls) -> SelectBase[PatientDb]:
        return select(PatientDb).order_by(PatientDb.id.desc())  # type: ignore

    @classmethod
    @logfire.instrument(record_return=True)
    async def check_exists(cls, patient_id: int, *, raise_exception: bool = True) -> bool:
        async with start_readonly_session() as session:
            patient_db = await session.get(PatientDb, patient_id)
            exists = patient_db is not None

        if not exists:
            if raise_exception:
                raise PatientNotFoundError(patient_id)
            return False

        return True

    @classmethod
    @logfire.instrument(record_return=True)
    async def create(cls, patient_create: PatientCreate, patient_id: int | None = None) -> Patient:
        patient_db = PatientDb.model_validate(patient_create.metadata)
        if patient_id is not None:
            patient_db.id = patient_id

        async with start_transaction() as session:
            session.add(patient_db)
            await session.flush()
            patient_id = patient_db.id

        async with LockManager.write(Lock.patient(patient_id)):
            await aiofiles.os.makedirs(Paths.storage.patient_dir(patient_id))
            await aiofiles.os.makedirs(Paths.storage.all_examinations_dir(patient_id))

            await util.dump_yaml(
                patient_create.info.model_dump(exclude_unset=True),
                Paths.storage.patient_info_file(patient_id),
            )
            await util.write_text(
                patient_create.comment,
                Paths.storage.patient_comment_file(patient_id),
            )

        return Patient(
            id=patient_id,
            metadata=patient_create.metadata,
            info=patient_create.info,
            comment=patient_create.comment,
        )

    @classmethod
    @logfire.instrument(record_return=True)
    async def get_by_id(cls, patient_id: int) -> Patient:
        async with start_readonly_session() as session:
            patient_db = await session.get(PatientDb, patient_id)
            if patient_db is None:
                raise PatientNotFoundError(patient_id)

            metadata = PatientMetadata(**patient_db.model_dump())

        async with LockManager.read(Lock.patient(patient_id)):
            raw_info = await util.load_yaml(Paths.storage.patient_info_file(patient_id))
            comment = await util.read_text(Paths.storage.patient_comment_file(patient_id))

        return Patient(
            id=patient_id,
            metadata=metadata,
            info=PatientInfo.model_validate(raw_info),
            comment=comment,
        )

    @classmethod
    @logfire.instrument
    async def update_by_id(
        cls, patient_id: int, patient_update: PatientUpdate, *, _check: bool = True
    ) -> None:
        if _check:
            await cls.check_exists(patient_id)

        if patient_update.metadata is not None:
            async with start_transaction() as session:
                patient_db = await session.get(PatientDb, patient_id)

                if patient_db is None:
                    raise PatientNotFoundError(patient_id)

                patient_db.sqlmodel_update(patient_update.metadata.model_dump(exclude_unset=True))
                session.add(patient_db)

        if patient_update.info is not None:
            async with LockManager.write(Lock.patient(patient_id)):
                info_path = Paths.storage.patient_info_file(patient_id)

                old_info = PatientInfo.model_validate(await util.load_yaml(info_path))
                updated_info = old_info.model_copy(
                    update=patient_update.info.model_dump(exclude_unset=True)
                )

                await util.dump_yaml(updated_info.model_dump(exclude_unset=True), info_path)
