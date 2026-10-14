"""Plan which frames make up each clip.

Pure functions only — no I/O — so the segmentation logic is trivially testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-6


@dataclass(frozen=True, slots=True)
class FrameWindow:
    """A clip expressed as a time range plus the frame indices sampled from it."""

    start_sec: float
    end_sec: float
    frame_indices: tuple[int, ...]

    @property
    def last_frame(self) -> int:
        return self.frame_indices[-1]


def make_window(
    start_sec: float,
    end_sec: float,
    *,
    fps: float,
    frame_count: int,
    num_frames: int,
) -> FrameWindow:
    """Uniformly sample ``num_frames`` frame indices from ``[start_sec, end_sec)``.

    Indices repeat when the range holds fewer frames than requested, which keeps
    the encoder's expected temporal length fixed for very short videos.
    """
    if fps <= 0 or frame_count <= 0:
        raise ValueError("fps and frame_count must be positive")
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")

    first = min(round(start_sec * fps), frame_count - 1)
    last = min(round(end_sec * fps) - 1, frame_count - 1)
    last = max(first, last)

    indices = np.linspace(first, last, num_frames).round().astype(int)
    return FrameWindow(
        start_sec=round(start_sec, 3),
        end_sec=round(end_sec, 3),
        frame_indices=tuple(int(i) for i in indices),
    )


def plan_windows(
    *,
    fps: float,
    frame_count: int,
    window_sec: float,
    stride_sec: float,
    num_frames: int,
) -> list[FrameWindow]:
    """Tile a video with sliding windows, guaranteeing the tail is covered.

    Windows are returned in chronological order with monotonically
    non-decreasing ``last_frame`` — a property the streaming reader relies on.
    """
    if window_sec <= 0 or stride_sec <= 0:
        raise ValueError("window_sec and stride_sec must be positive")

    duration = frame_count / fps
    if duration <= window_sec + _EPS:
        starts = [0.0]
    else:
        last_start = duration - window_sec
        starts = [float(s) for s in np.arange(0.0, last_start + _EPS, stride_sec)]
        if last_start - starts[-1] > _EPS:
            starts.append(last_start)

    return [
        make_window(
            start,
            min(start + window_sec, duration),
            fps=fps,
            frame_count=frame_count,
            num_frames=num_frames,
        )
        for start in starts
    ]
