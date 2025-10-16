from app.core import Paths

from .patient import PatientStorage


async def build_storage() -> None:
    Paths.storage.all_patients_dir.mkdir(parents=True, exist_ok=True)
    Paths.storage.patient_names_file.touch()

    await PatientStorage.initialize()
