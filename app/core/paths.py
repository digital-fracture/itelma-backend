import tempfile
from functools import lru_cache
from pathlib import Path


class InternalPaths:
    resource_dir = Path("resources")
    config = resource_dir / "config.yml"


class MLPaths:
    root_dir = InternalPaths.resource_dir / "ml"

    predictor_model = root_dir / "predictor.pt"


class StoragePaths:
    root_dir = Path("storage")

    all_patients_dir = root_dir / "patients"
    patients_db_file = all_patients_dir / "patients.db"
    patients_db_url = f"sqlite+aiosqlite:///{patients_db_file}"

    examinations_dir_name = "examinations"
    bpm_dir_name = "bpm"
    uterus_dir_name = "uterus"
    intervals_dir_name = "intervals"

    information_file_name = "information.yml"
    comment_file_name = "comment.txt"
    metadata_file_name = "metadata.yml"
    stats_file_name = "stats.yml"
    verdict_file_name = "verdict.yml"

    @classmethod
    @lru_cache
    def patient_dir(cls, patient_id: int) -> Path:
        return cls.all_patients_dir / str(patient_id)

    @classmethod
    @lru_cache
    def patient_info_file(cls, patient_id: int) -> Path:
        return cls.patient_dir(patient_id) / cls.information_file_name

    @classmethod
    @lru_cache
    def patient_comment_file(cls, patient_id: int) -> Path:
        return cls.patient_dir(patient_id) / cls.comment_file_name

    @classmethod
    @lru_cache
    def all_examinations_dir(cls, patient_id: int) -> Path:
        return cls.patient_dir(patient_id) / cls.examinations_dir_name

    @classmethod
    @lru_cache
    def examination_dir(cls, patient_id: int, examination_id: int) -> Path:
        return cls.all_examinations_dir(patient_id) / str(examination_id)

    @classmethod
    @lru_cache
    def examination_metadata_file(cls, patient_id: int, examination_id: int) -> Path:
        return cls.examination_dir(patient_id, examination_id) / cls.metadata_file_name

    @classmethod
    @lru_cache
    def examination_stats_file(cls, patient_id: int, examination_id: int) -> Path:
        return cls.examination_dir(patient_id, examination_id) / cls.stats_file_name

    @classmethod
    @lru_cache
    def examination_verdict_file(cls, patient_id: int, examination_id: int) -> Path:
        return cls.examination_dir(patient_id, examination_id) / cls.verdict_file_name

    @classmethod
    @lru_cache
    def examination_bpm_dir(cls, patient_id: int, examination_id: int) -> Path:
        return cls.examination_dir(patient_id, examination_id) / cls.bpm_dir_name

    @classmethod
    @lru_cache
    def examination_uterus_dir(cls, patient_id: int, examination_id: int) -> Path:
        return cls.examination_dir(patient_id, examination_id) / cls.uterus_dir_name

    @classmethod
    @lru_cache
    def examination_intervals_dir(cls, patient_id: int, examination_id: int) -> Path:
        return cls.examination_dir(patient_id, examination_id) / cls.intervals_dir_name

    @classmethod
    @lru_cache
    def examination_part_bpm_file(
        cls, patient_id: int, examination_id: int, part_index: int
    ) -> Path:
        return cls.examination_bpm_dir(patient_id, examination_id) / f"{part_index}.csv"

    @classmethod
    @lru_cache
    def examination_part_uterus_file(
        cls, patient_id: int, examination_id: int, part_index: int
    ) -> Path:
        return cls.examination_uterus_dir(patient_id, examination_id) / f"{part_index}.csv"

    @classmethod
    @lru_cache
    def examination_part_intervals_file(
        cls, patient_id: int, examination_id: int, part_index: int
    ) -> Path:
        return cls.examination_intervals_dir(patient_id, examination_id) / f"{part_index}.yml"


class Paths:
    temp_dir = Path(tempfile.gettempdir()) / "itelma"

    internal = InternalPaths
    ml = MLPaths
    storage = StoragePaths
