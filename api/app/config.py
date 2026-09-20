from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Segmentation API"
    app_version: str = "1.0.0"

    model_mode: Literal["local", "huggingface", "mock"] = "local"
    model_name: Literal[
        "unet_mobilenetv2_sans_augmentation", "segformer_sans_augmentation"
    ] = "unet_mobilenetv2_sans_augmentation"
    hf_repo_id: str | None = None
    hf_revision: str = "main"
    hf_cache_dir: str | None = None

    max_upload_mb: int = 10

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
