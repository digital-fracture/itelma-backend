from app.core import Paths

from .patient import PatientStorage


async def build_storage() -> None:
    Paths.temp_dir.mkdir(parents=True, exist_ok=True)

    Paths.all_patients_dir.mkdir(parents=True, exist_ok=True)
    Paths.patient_names_file.touch()

    await PatientStorage.initialize()
