"""Typed, environment-driven configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Device = Literal["auto", "cpu", "cuda", "mps"]


class Settings(BaseSettings):
    """Runtime settings, overridable via ``LATENTLENS_*`` env vars or a ``.env`` file."""

    model_config = SettingsConfigDict(
        env_prefix="LATENTLENS_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    data_dir: Path = Path("data")

    # Encoder
    encoder_model_id: str = "facebook/vjepa2-vitl-fpc64-256"
    device: Device = "auto"
    batch_size: int = Field(default=4, ge=1)

    # Temporal segmentation
    frames_per_clip: int = Field(default=16, ge=2)
    window_sec: float = Field(default=4.0, gt=0)
    stride_sec: float = Field(default=2.0, gt=0)

    # Decoding
    resize_short_side: int | None = Field(default=256, ge=32)

    # Language bridge (Phase 2) — Gemini free tier
    gemini_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "LATENTLENS_GEMINI_API_KEY"),
    )
    gemini_model: str = "gemini-2.5-flash"
    gemini_rpm: int = Field(default=10, ge=1, description="Requests/minute budget (free tier)")
    keyframes_per_clip: int = Field(default=3, ge=1, le=8)

    @field_validator("frames_per_clip")
    @classmethod
    def _even_frames(cls, value: int) -> int:
        if value % 2:
            raise ValueError("frames_per_clip must be even (V-JEPA tubelet size is 2)")
        return value

    @model_validator(mode="after")
    def _stride_within_window(self) -> Settings:
        if self.stride_sec > self.window_sec:
            raise ValueError("stride_sec must not exceed window_sec (would leave gaps)")
        return self

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def captions_path(self) -> Path:
        return self.data_dir / "captions.json"

    def require_gemini_key(self) -> str:
        """Return the Gemini key or fail with an actionable message."""
        if self.gemini_api_key is None or not self.gemini_api_key.get_secret_value():
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/apikey and add it to your .env file."
            )
        return self.gemini_api_key.get_secret_value()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
