import asyncio
import datetime
from pathlib import Path
from typing import ClassVar

import aiofiles
import aiofiles.os
import logfire
from async_lru import alru_cache
from fastapi import UploadFile

from app import analysis
from app.core import Config, Paths
from app.core.exceptions import (
    ExaminationNotFoundError,
    ExaminationPartNotFoundError,
    UnsupportedFileTypeError,
)
from app.model import (
    Examination,
    ExaminationBrief,
    ExaminationMetadata,
    ExaminationPartInterval,
    ExaminationPlot,
    ExaminationStats,
    ExaminationVerdict,
    OverallState,
    PatientMetadata,
    PatientUpdate,
    PlotPoint,
)

from . import util
from .locking import Lock, LockManager
from .patient import PatientStorage


class ExaminationStorage:
    _background_tasks: ClassVar[set[asyncio.Task[None]]] = set()

    @staticmethod
    @logfire.instrument(record_return=True)
    async def check_exists(
        patient_id: int, examination_id: int, *, raise_exception: bool = True
    ) -> bool:
        if not (await PatientStorage.check_exists(patient_id, raise_exception=raise_exception)):
            return False

        async with LockManager.read(Lock.patient_examination_list(patient_id)):
            if str(examination_id) not in (
                await aiofiles.os.listdir(Paths.storage.all_examinations_dir(patient_id))
            ):
                if raise_exception:
                    raise ExaminationNotFoundError(patient_id, examination_id)
                return False

        return True

    @classmethod
    @logfire.instrument(record_return=True)
    async def save_uploaded(
        cls, patient_id: int, examination_upload_file: UploadFile
    ) -> Examination:
        async with LockManager.write(Lock.patient_examination_list(patient_id)):
            examination_id = (
                max(
                    map(
                        int,
                        await aiofiles.os.listdir(Paths.storage.all_examinations_dir(patient_id)),
                    ),
                    default=0,
                )
                + 1
            )

        zip_path = await util.save_temp_file(examination_upload_file)
        examination_dir = Paths.storage.examination_dir(patient_id, examination_id)

        async with LockManager.write(Lock.patient_examination(patient_id, examination_id)):
            try:
                await util.unzip(zip_path, examination_dir)
                await util.fix_extracted_examination(examination_dir)
            except UnsupportedFileTypeError:
                await util.rmtree(examination_dir)
                raise
            except Exception as e:
                await util.rmtree(examination_dir)
                raise UnsupportedFileTypeError from e

            await asyncio.gather(
                util.rename_to_numbers(
                    Paths.storage.examination_bpm_dir(patient_id, examination_id)
                ),
                util.rename_to_numbers(
                    Paths.storage.examination_uterus_dir(patient_id, examination_id)
                ),
            )

            metadata = ExaminationMetadata(
                date=datetime.datetime.now(tz=datetime.UTC).astimezone().date(),
                part_count=len(
                    await aiofiles.os.listdir(
                        Paths.storage.examination_bpm_dir(patient_id, examination_id)
                    )
                ),
            )
            await util.dump_yaml(
                metadata.model_dump(exclude_unset=True),
                Paths.storage.examination_metadata_file(patient_id, examination_id),
            )

            # creating empty files to not break get_by_id method
            await aiofiles.os.makedirs(
                Paths.storage.examination_intervals_dir(patient_id, examination_id)
            )
            await asyncio.gather(
                *(
                    util.create_empty(path)
                    for path in (
                        [
                            Paths.storage.examination_stats_file(patient_id, examination_id),
                            Paths.storage.examination_verdict_file(patient_id, examination_id),
                        ]
                        + [
                            Paths.storage.examination_part_intervals_file(
                                patient_id, examination_id, part_index
                            )
                            for part_index in range(1, metadata.part_count + 1)
                        ]
                    )
                ),
            )

        cls._background_tasks.add(
            asyncio.create_task(
                cls._analyze_and_save(patient_id, examination_id, metadata.part_count)
            )
        )

        return Examination(id=examination_id, metadata=metadata)

    @classmethod
    @logfire.instrument(record_return=True)
    async def get_all_by_patient_id(cls, patient_id: int) -> list[ExaminationBrief]:
        await PatientStorage.check_exists(patient_id)

        async with LockManager.read(Lock.patient_examination_list(patient_id)):
            examinations_dir = Paths.storage.all_examinations_dir(patient_id)

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
        metadata_path = examination_dir / Paths.storage.metadata_file_name

        raw_metadata = await util.load_yaml(metadata_path)
        return ExaminationBrief(
            id=examination_id, metadata=ExaminationMetadata.model_validate(raw_metadata)
        )

    @classmethod
    @logfire.instrument(record_return=True)
    async def get_by_id(cls, patient_id: int, examination_id: int) -> Examination:
        await cls.check_exists(patient_id, examination_id)

        async with (
            LockManager.read(Lock.patient_examination(patient_id, examination_id)),
            asyncio.TaskGroup() as tg,
        ):
            metadata_task = tg.create_task(
                util.load_yaml(Paths.storage.examination_metadata_file(patient_id, examination_id))
            )
            stats_task = tg.create_task(
                util.load_yaml(Paths.storage.examination_stats_file(patient_id, examination_id))
            )
            verdict_task = tg.create_task(
                util.load_yaml(Paths.storage.examination_verdict_file(patient_id, examination_id))
            )

        return Examination(
            id=patient_id,
            metadata=ExaminationMetadata.model_validate(metadata_task.result()),
            stats=ExaminationStats.model_validate(stats_task.result()),
            verdict=ExaminationVerdict.model_validate(verdict_task.result()),
        )

    @classmethod
    @alru_cache(maxsize=Config.server.part_cache_size)
    @logfire.instrument
    async def read_part_plot(
        cls, patient_id: int, examination_id: int, part_index: int, *, _check: bool = True
    ) -> ExaminationPlot:
        if _check:
            await cls.check_exists(patient_id, examination_id)

        bpm_path = Paths.storage.examination_part_bpm_file(patient_id, examination_id, part_index)
        uterus_path = Paths.storage.examination_part_uterus_file(
            patient_id, examination_id, part_index
        )

        if not all(
            await asyncio.gather(
                aiofiles.os.path.exists(bpm_path), aiofiles.os.path.exists(uterus_path)
            )
        ):
            raise ExaminationPartNotFoundError(patient_id, examination_id, part_index)

        async with asyncio.TaskGroup() as tg:
            bpm_task = tg.create_task(cls._read_csv(bpm_path))
            uterus_task = tg.create_task(cls._read_csv(uterus_path))

        return ExaminationPlot(bpm=bpm_task.result(), uterus=uterus_task.result())

    @staticmethod
    async def _read_csv(path: Path) -> list[PlotPoint]:
        async with aiofiles.open(path) as file:
            await file.readline()  # skip csv header

            points = []
            async for line in file:
                timestamp, value = map(float, line.strip().split(","))
                points.append((timestamp, value))

        return points

    @classmethod
    async def _analyze_and_save(cls, patient_id: int, examination_id: int, part_count: int) -> None:
        all_plots: list[ExaminationPlot] = await asyncio.gather(
            *(
                cls.read_part_plot(patient_id, examination_id, part_index, _check=False)
                for part_index in range(1, part_count + 1)
            )
        )
        concatenated_plot: ExaminationPlot = ExaminationPlot.concatenate(*all_plots)

        intervals_per_part: list[list[ExaminationPartInterval]] = [  # TODO: here would be gather
            [
                ExaminationPartInterval(start=5, end=10, message="интервал 1"),
                ExaminationPartInterval(start=15, end=20, message="интервал 2"),
            ]
            for index, _ in enumerate(all_plots, start=1)
        ]

        stats = ExaminationStats(  # TODO: calculate real stats
            bpm_average=concatenated_plot.bpm[0][1],
            uterus_average=concatenated_plot.uterus[0][1],
        )
        predictions = await analysis.predict(concatenated_plot)  # noqa: F841

        verdict = ExaminationVerdict(
            overall_status=OverallState.ATTENTION,
            recommendations=["рекомендация 1"],
            attention_zones=["что-то в норме 1"],
            risk_zones=["зона риска 1"],
        )

        async with (
            LockManager.write(Lock.patient_examination(patient_id, examination_id)),
            asyncio.TaskGroup() as tg,
        ):
            tg.create_task(
                util.dump_yaml(
                    stats.model_dump(),
                    Paths.storage.examination_stats_file(patient_id, examination_id),
                )
            )
            tg.create_task(
                util.dump_yaml(
                    verdict.model_dump(),
                    Paths.storage.examination_verdict_file(patient_id, examination_id),
                )
            )
            for part_index, intervals in enumerate(intervals_per_part, start=1):
                tg.create_task(
                    util.dump_yaml(
                        [interval.model_dump() for interval in intervals],
                        Paths.storage.examination_part_intervals_file(
                            patient_id, examination_id, part_index
                        ),
                    )
                )

        await PatientStorage.update_by_id(
            patient_id,
            PatientUpdate(metadata=PatientMetadata(overall_state=verdict.overall_status)),
        )
