"""Streaming video decoding built on OpenCV."""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np

from latentlens.video.segmenter import FrameWindow

logger = logging.getLogger(__name__)

Frames = np.ndarray  # (T, H, W, 3) uint8 RGB

_FINGERPRINT_CHUNK = 1 << 20  # 1 MiB


class VideoDecodeError(RuntimeError):
    """Raised when a video cannot be opened or decoded."""


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps


def fingerprint(path: str | Path) -> str:
    """Fast, content-based video ID: hash of size + head + tail bytes.

    Stable across renames/moves, and avoids hashing multi-GB files end to end.
    """
    path = Path(path)
    size = path.stat().st_size
    digest = hashlib.blake2b(str(size).encode(), digest_size=8)
    with path.open("rb") as fh:
        digest.update(fh.read(_FINGERPRINT_CHUNK))
        if size > _FINGERPRINT_CHUNK:
            fh.seek(max(size - _FINGERPRINT_CHUNK, _FINGERPRINT_CHUNK))
            digest.update(fh.read(_FINGERPRINT_CHUNK))
    return digest.hexdigest()


class VideoReader:
    """Decode clips from a video in a single sequential pass.

    Seeking is slow and unreliable for many codecs, so frames are read in order
    and routed to every window that needs them. Memory stays bounded because a
    window is yielded (and released) as soon as its last frame is decoded.
    """

    def __init__(self, path: str | Path, *, resize_short_side: int | None = 256) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._resize_short_side = resize_short_side
        self._cap: cv2.VideoCapture | None = None
        self._metadata: VideoMetadata | None = None

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> VideoReader:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def open(self) -> None:
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise VideoDecodeError(f"Cannot open video: {self.path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or frame_count <= 0:
            cap.release()
            raise VideoDecodeError(f"Video reports no frames or invalid FPS: {self.path}")

        self._cap = cap
        self._metadata = VideoMetadata(
            path=self.path,
            fps=float(fps),
            frame_count=frame_count,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def metadata(self) -> VideoMetadata:
        if self._metadata is None:
            raise RuntimeError("VideoReader is not open")
        return self._metadata

    # -- decoding ----------------------------------------------------------

    def read_windows(self, windows: Sequence[FrameWindow]) -> Iterator[tuple[FrameWindow, Frames]]:
        """Yield ``(window, frames)`` pairs in order. Windows must come from the segmenter."""
        if self._cap is None:
            raise RuntimeError("VideoReader is not open")
        if not windows:
            return

        routes: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for w_idx, window in enumerate(windows):
            for slot, frame_idx in enumerate(window.frame_indices):
                routes[frame_idx].append((w_idx, slot))

        buffers: list[list[np.ndarray | None]] = [
            [None] * len(w.frame_indices) for w in windows
        ]
        last_needed = max(routes)
        next_emit = 0
        frame_idx = 0

        while frame_idx <= last_needed:
            if frame_idx in routes:
                ok, bgr = self._cap.read()
                if not ok:
                    break
                rgb = self._postprocess(bgr)
                for w_idx, slot in routes[frame_idx]:
                    buffers[w_idx][slot] = rgb
            elif not self._cap.grab():
                break

            while next_emit < len(windows) and windows[next_emit].last_frame <= frame_idx:
                yield windows[next_emit], _stack(buffers[next_emit])
                buffers[next_emit] = []
                next_emit += 1
            frame_idx += 1

        # Container metadata often over-reports frame counts; salvage partial windows.
        for w_idx in range(next_emit, len(windows)):
            frames = buffers[w_idx]
            if any(f is not None for f in frames):
                logger.debug("Padding truncated window %s in %s", windows[w_idx], self.path.name)
                yield windows[w_idx], _stack(_fill_gaps(frames))
            else:
                logger.warning(
                    "Dropping window %.2fs–%.2fs of %s: stream ended early",
                    windows[w_idx].start_sec,
                    windows[w_idx].end_sec,
                    self.path.name,
                )

    def _postprocess(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        target = self._resize_short_side
        if target is None:
            return rgb
        h, w = rgb.shape[:2]
        scale = target / min(h, w)
        if scale >= 1.0:
            return rgb
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)


def _fill_gaps(frames: list[np.ndarray | None]) -> list[np.ndarray | None]:
    """Forward-fill missing frames, back-filling any leading gap."""
    first = next(f for f in frames if f is not None)
    filled: list[np.ndarray | None] = []
    previous = first
    for frame in frames:
        previous = frame if frame is not None else previous
        filled.append(previous)
    return filled


def _stack(frames: list[np.ndarray | None]) -> Frames:
    if any(f is None for f in frames):
        frames = _fill_gaps(frames)
    return np.stack(frames)  # type: ignore[arg-type]
