from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from latentlens.config import Settings


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def test_rejects_odd_frame_count() -> None:
    with pytest.raises(ValidationError, match="even"):
        _settings(frames_per_clip=7)


def test_rejects_stride_larger_than_window() -> None:
    with pytest.raises(ValidationError, match="stride_sec"):
        _settings(window_sec=2.0, stride_sec=3.0)


def test_derived_directories(tmp_path: Path) -> None:
    settings = _settings(data_dir=tmp_path)
    assert settings.index_dir == tmp_path / "index"
    assert settings.cache_dir == tmp_path / "cache"


def test_reads_standard_gemini_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret-123")
    settings = _settings()

    assert settings.require_gemini_key() == "secret-123"
    assert "secret-123" not in repr(settings)  # never leak keys into logs


def test_missing_gemini_key_fails_with_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LATENTLENS_GEMINI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match=r"aistudio\.google\.com"):
        _settings().require_gemini_key()
