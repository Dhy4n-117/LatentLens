"""Pick and compress the few frames per clip that get sent to Gemini for captioning.

Images are the most expensive part of a multimodal request, so we send a small,
evenly spread set of frames, downscaled and JPEG-compressed.
"""

from __future__ import annotations

import cv2
import numpy as np


def keyframe_indices(num_frames: int, count: int) -> list[int]:
    """Indices of ``count`` frames at the centres of equal temporal segments.

    Centres (not edges) avoid boundary frames, which are the most likely to
    belong to a neighbouring shot. Never repeats an index.
    """
    if num_frames < 1 or count < 1:
        raise ValueError("num_frames and count must be >= 1")
    count = min(count, num_frames)
    return [int((i + 0.5) * num_frames / count) for i in range(count)]


def select_keyframes(frames: np.ndarray, count: int = 3) -> np.ndarray:
    """Return a ``(k, H, W, 3)`` subset of a ``(T, H, W, 3)`` clip, ``k = min(count, T)``."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected (T, H, W, 3) frames, got shape {frames.shape}")
    return frames[keyframe_indices(len(frames), count)]


def encode_jpeg(frame: np.ndarray, *, max_side: int = 512, quality: int = 85) -> bytes:
    """Downscale an RGB frame so its long side is at most ``max_side``, then JPEG-encode it."""
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"Expected (H, W, 3) frame, got shape {frame.shape}")
    if not 1 <= quality <= 100:
        raise ValueError("quality must be in [1, 100]")

    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )

    bgr = cv2.cvtColor(np.ascontiguousarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buffer.tobytes()
