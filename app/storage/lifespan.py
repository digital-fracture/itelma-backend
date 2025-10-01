from app.core import Paths


def build_storage() -> None:
    Paths.all_patients_dir.mkdir(parents=True, exist_ok=True)
    Paths.patient_names_file.touch()
