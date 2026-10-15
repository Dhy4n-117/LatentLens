from __future__ import annotations

import cv2
import numpy as np
import pytest

from latentlens.video import encode_jpeg, keyframe_indices, select_keyframes


@pytest.mark.parametrize(
    ("num_frames", "count", "expected"),
    [
        (16, 3, [2, 8, 13]),
        (16, 1, [8]),
        (4, 4, [0, 1, 2, 3]),
        (2, 5, [0, 1]),  # clamps instead of repeating frames
    ],
)
def test_keyframe_indices(num_frames: int, count: int, expected: list[int]) -> None:
    assert keyframe_indices(num_frames, count) == expected


def test_select_keyframes_returns_requested_frames() -> None:
    frames = np.arange(8, dtype=np.uint8).reshape(8, 1, 1, 1).repeat(3, axis=-1)
    picked = select_keyframes(frames, count=2)
    assert picked.shape == (2, 1, 1, 3)
    assert picked[:, 0, 0, 0].tolist() == [2, 6]


def test_select_keyframes_validates_shape() -> None:
    with pytest.raises(ValueError):
        select_keyframes(np.zeros((4, 8, 8), dtype=np.uint8))


def test_encode_jpeg_downscales_and_preserves_colour() -> None:
    frame = np.zeros((600, 1200, 3), dtype=np.uint8)
    frame[..., 0] = 220  # red in RGB

    data = encode_jpeg(frame, max_side=400)
    decoded = cv2.cvtColor(
        cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
    )

    assert data[:2] == b"\xff\xd8"  # JPEG magic number
    assert decoded.shape == (200, 400, 3)
    r, g, b = decoded.reshape(-1, 3).mean(axis=0)
    assert r > 200 and g < 30 and b < 30


def test_encode_jpeg_keeps_small_frames_unscaled() -> None:
    data = encode_jpeg(np.full((48, 64, 3), 128, dtype=np.uint8), max_side=512)
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (48, 64, 3)


@pytest.mark.parametrize("quality", [0, 101])
def test_encode_jpeg_rejects_bad_quality(quality: int) -> None:
    with pytest.raises(ValueError):
        encode_jpeg(np.zeros((8, 8, 3), dtype=np.uint8), quality=quality)
