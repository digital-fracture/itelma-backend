from functools import cached_property
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_config import SettingsConfig, SettingsModel

ENV_PREFIX = "ITELMA__"
ENV_SEPARATOR = "__"

RESOURCE_DIR = Path("resources")
CONFIG_PATH = RESOURCE_DIR / "config.yml"


class ServerConfig(BaseModel):
    title: str
    allow_origins_raw: str = Field(validation_alias="allow_origins")
    openapi_tag_list: list[str] = Field(validation_alias="openapi_tags")

    @cached_property
    def allow_origins(self) -> list[str]:
        return [origin.strip() for origin in self.allow_origins_raw.split(",") if origin.strip()]

    @cached_property
    def openapi_tags(self) -> list[dict[str, str]]:
        return [{"name": tag} for tag in self.openapi_tag_list]


class LogfireConfig(BaseModel):
    service_name: str
    environment: str


class BloodGasConfig(BaseModel):
    unit: str
    normal_min: float
    normal_max: float


class MedicalConfig(BaseModel):
    blood_gas: dict[str, BloodGasConfig]


class ConfigModel(SettingsModel):
    server: ServerConfig
    logfire: LogfireConfig
    medical: MedicalConfig

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


Config = ConfigModel()
