"""Shared fixtures: a deterministic fake encoder and synthetic test videos."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest

from latentlens.config import Settings
from latentlens.encoders.base import l2_normalize
from latentlens.index import VectorStore

FPS = 10
SIZE = (64, 48)  # (width, height)

RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)


class ColorEncoder:
    """Embeds a clip as its mean RGB colour — trivially predictable similarity."""

    name = "fake:color"
    dim = 3

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, clips: Sequence[np.ndarray]) -> np.ndarray:
        self.calls += 1
        if not clips:
            return np.empty((0, self.dim), dtype=np.float32)
        means = np.stack([c.reshape(-1, 3).mean(axis=0) for c in clips]) + 1.0
        return l2_normalize(means)


def write_video(path: Path, segments: Sequence[tuple[tuple[int, int, int], float]]) -> Path:
    """Write an MJPG AVI made of solid-colour segments ``[(rgb, seconds), ...]``."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, SIZE)
    assert writer.isOpened(), "OpenCV could not open a video writer"
    for rgb, seconds in segments:
        frame = np.full((SIZE[1], SIZE[0], 3), rgb[::-1], dtype=np.uint8)  # RGB → BGR
        for _ in range(int(seconds * FPS)):
            writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        frames_per_clip=4,
        window_sec=2.0,
        stride_sec=1.0,
        batch_size=3,
        resize_short_side=None,
        _env_file=None,  # type: ignore[call-arg]
    )


@pytest.fixture
def encoder() -> ColorEncoder:
    return ColorEncoder()


@pytest.fixture
def store(encoder: ColorEncoder) -> VectorStore:
    return VectorStore(dim=encoder.dim, encoder_name=encoder.name)


@pytest.fixture
def rgb_video(tmp_path: Path) -> Path:
    """10 s: red 0–4 s, green 4–7 s, blue 7–10 s."""
    return write_video(tmp_path / "rgb.avi", [(RED, 4), (GREEN, 3), (BLUE, 3)])


@pytest.fixture
def blue_video(tmp_path: Path) -> Path:
    return write_video(tmp_path / "blue.avi", [(BLUE, 4)])
