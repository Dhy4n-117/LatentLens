"""Indexed clips → keyframes → Gemini captions."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from latentlens.config import Settings
from latentlens.index.captions import CaptionStore
from latentlens.llm.captioner import Captioner, CaptionError
from latentlens.types import ClipSpan
from latentlens.video import VideoReader, encode_jpeg, keyframe_window

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True, slots=True)
class CaptionReport:
    captioned: int = 0
    already_done: int = 0
    failed: int = 0
    missing_video: int = 0


class CaptioningPipeline:
    """Caption every clip that doesn't have a caption yet.

    Resumable by design: captions are saved after each video, so an interrupted
    run (or an exhausted daily quota) loses at most one video's progress.
    """

    def __init__(self, captioner: Captioner, captions: CaptionStore, settings: Settings) -> None:
        self._captioner = captioner
        self._captions = captions
        self._settings = settings

    def run(
        self,
        spans: Iterable[ClipSpan],
        *,
        limit: int | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> CaptionReport:
        spans = list(spans)
        pending = [s for s in spans if s not in self._captions]
        already_done = len(spans) - len(pending)
        if limit is not None:
            pending = pending[:limit]

        by_video: dict[str, list[ClipSpan]] = defaultdict(list)
        for span in pending:
            by_video[span.video_id].append(span)

        captioned = failed = missing = done = 0
        for group in by_video.values():
            path = Path(group[0].video_path)
            if not path.is_file():
                logger.warning("Skipping %d clips: video not found at %s", len(group), path)
                missing += len(group)
                done += len(group)
                if on_progress:
                    on_progress(done, len(pending))
                continue

            ok, bad = self._caption_video(path, group, done, len(pending), on_progress)
            captioned, failed, done = captioned + ok, failed + bad, done + len(group)
            self._captions.save()

        return CaptionReport(captioned, already_done, failed, missing)

    def _caption_video(
        self,
        path: Path,
        spans: list[ClipSpan],
        done: int,
        total: int,
        on_progress: ProgressCallback | None,
    ) -> tuple[int, int]:
        count = self._settings.keyframes_per_clip
        ok = bad = 0
        # Full-resolution decode; encode_jpeg handles downscaling for the API.
        with VideoReader(path, resize_short_side=None) as reader:
            meta = reader.metadata
            windows = {
                keyframe_window(
                    s.start_sec, s.end_sec, fps=meta.fps, frame_count=meta.frame_count, count=count
                ): s
                for s in spans
            }
            ordered = sorted(windows, key=lambda w: (w.last_frame, w.start_sec))

            for window, frames in reader.read_windows(ordered):
                span = windows[window]
                try:
                    text = self._captioner.caption([encode_jpeg(f) for f in frames])
                except CaptionError as exc:
                    logger.warning("Caption failed for %s %s: %s", path.name, span.label(), exc)
                    bad += 1
                else:
                    self._captions.add(span, text, self._captioner.name)
                    ok += 1
                if on_progress:
                    on_progress(done + ok + bad, total)
        return ok, bad
