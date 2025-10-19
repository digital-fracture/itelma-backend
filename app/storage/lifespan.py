from app.core import Paths

from .database import dispose_database, initialize_database


async def initialize_storage() -> None:
    Paths.storage.all_patients_dir.mkdir(parents=True, exist_ok=True)
    Paths.storage.patients_db_file.touch()

    await initialize_database()


async def dispose_storage() -> None:
    await dispose_database()
