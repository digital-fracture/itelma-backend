import tempfile
from functools import lru_cache
from pathlib import Path

from .config import RESOURCE_DIR


class MLPaths:
    root_dir = RESOURCE_DIR / "ml"

    predictor_model = root_dir / "predictor.pt"


class StoragePaths:
    root_dir = Path("storage")

    all_patients_dir = root_dir / "patients"
    patient_names_file = all_patients_dir / "names.yml"

    examinations_dir_name = "examinations"
    bpm_dir_name = "bpm"
    uterus_dir_name = "uterus"

    information_file_name = "information.yml"
    metadata_file_name = "metadata.yml"
    predictions_file_name = "predictions.yml"

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
    def patient_predictions_file(cls, patient_id: int) -> Path:
        return cls.patient_dir(patient_id) / cls.predictions_file_name

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
    def examination_predictions_file(cls, patient_id: int, examination_id: int) -> Path:
        return cls.examination_dir(patient_id, examination_id) / cls.predictions_file_name

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


class Paths:
    temp_dir = Path(tempfile.gettempdir()) / "itelma"

    ml = MLPaths
    storage = StoragePaths
