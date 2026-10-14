"""Video → clips → embeddings → index."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from latentlens.config import Settings
from latentlens.encoders.base import VideoEncoder
from latentlens.index.store import VectorStore
from latentlens.types import ClipSpan
from latentlens.utils import batched
from latentlens.video import VideoReader, fingerprint, plan_windows

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]  # (clips_done, clips_total)


@dataclass(frozen=True, slots=True)
class IngestReport:
    video_id: str
    path: Path
    num_clips: int
    duration_sec: float
    elapsed_sec: float
    skipped: bool = False


class IngestionPipeline:
    """Idempotently index videos: re-ingesting an unchanged file is a no-op."""

    def __init__(self, encoder: VideoEncoder, store: VectorStore, settings: Settings) -> None:
        self._encoder = encoder
        self._store = store
        self._settings = settings

    def ingest(
        self,
        path: str | Path,
        *,
        force: bool = False,
        on_progress: ProgressCallback | None = None,
    ) -> IngestReport:
        path = Path(path).resolve()
        video_id = fingerprint(path)
        started = time.perf_counter()

        if self._store.has_video(video_id):
            if not force:
                logger.info("Skipping %s (already indexed)", path.name)
                return IngestReport(video_id, path, 0, 0.0, 0.0, skipped=True)
            removed = self._store.remove_video(video_id)
            logger.info("Re-indexing %s (removed %d stale clips)", path.name, removed)

        cfg = self._settings
        with VideoReader(path, resize_short_side=cfg.resize_short_side) as reader:
            meta = reader.metadata
            windows = plan_windows(
                fps=meta.fps,
                frame_count=meta.frame_count,
                window_sec=cfg.window_sec,
                stride_sec=cfg.stride_sec,
                num_frames=cfg.frames_per_clip,
            )

            done = 0
            for batch in batched(reader.read_windows(windows), cfg.batch_size):
                spans = [
                    ClipSpan(video_id, str(path), window.start_sec, window.end_sec)
                    for window, _ in batch
                ]
                vectors = self._encoder.encode([frames for _, frames in batch])
                self._store.add(spans, vectors)

                done += len(batch)
                if on_progress:
                    on_progress(done, len(windows))

        elapsed = time.perf_counter() - started
        logger.info("Indexed %s: %d clips in %.1fs", path.name, done, elapsed)
        return IngestReport(video_id, path, done, meta.duration, elapsed)
