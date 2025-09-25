from functools import cached_property
from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from pydantic_config import SettingsConfig, SettingsModel
from sqlalchemy import URL

ENV_PREFIX = "ITELMA__"
ENV_SEPARATOR = "__"

RESOURCE_DIR = Path("resources")
CONFIG_PATH = RESOURCE_DIR / "config.yml"


class AppConfig(BaseModel):
    file_storage_dir: Path
    allowed_file_types: dict[str, str]

    @field_validator("file_storage_dir", mode="after")
    @classmethod
    def create_storage_dir(cls, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path


class ServerConfig(BaseModel):
    title: str
    allow_origins_raw: str = Field(validation_alias="allow_origins")
    ws_stop_message: str

    @cached_property
    def allow_origins(self) -> list[str]:
        return [origin.strip() for origin in self.allow_origins_raw.split(",") if origin.strip()]


class LogfireConfig(BaseModel):
    service_name: str
    environment: str


class PostgresConfig(BaseModel):
    host: str
    port: int
    db: str
    user: str
    password: str

    pool_size: int
    max_overflow: int

    @cached_property
    def url(self) -> URL:
        return URL.create(
            drivername="postgresql+asyncpg",
            username=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.db,
        )


class DBConfig(BaseModel):
    postgres: PostgresConfig


class Config(SettingsModel):
    app: AppConfig
    server: ServerConfig
    logfire: LogfireConfig
    db: DBConfig

    model_config = SettingsConfig(
        config_merge=True,
        nested_model_default_partial_update=True,
        enable_decoding=True,
        case_sensitive=False,
        env_prefix=ENV_PREFIX,
        env_nested_delimiter=ENV_SEPARATOR,
        config_file=CONFIG_PATH,
        extra="allow",
    )
