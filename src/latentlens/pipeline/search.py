"""Query the visual memory."""

from __future__ import annotations

from pathlib import Path

from latentlens.config import Settings
from latentlens.encoders.base import VideoEncoder
from latentlens.index.store import VectorStore
from latentlens.types import ClipSpan, SearchHit
from latentlens.video import VideoDecodeError, VideoReader, fingerprint, make_window


class SearchService:
    """Phase 1 search: *video moment → similar moments*.

    Phase 2 adds text queries by projecting language into the same latent space.
    """

    def __init__(self, encoder: VideoEncoder, store: VectorStore, settings: Settings) -> None:
        self._encoder = encoder
        self._store = store
        self._settings = settings

    def search_by_moment(
        self,
        path: str | Path,
        at_sec: float,
        *,
        k: int = 5,
        include_same_video: bool = True,
    ) -> list[SearchHit]:
        """Find clips that look and move like the window starting at ``at_sec`` in ``path``.

        The query window itself (and anything overlapping it) is always excluded,
        so results are genuinely *other* moments.
        """
        path = Path(path).resolve()
        cfg = self._settings

        with VideoReader(path, resize_short_side=cfg.resize_short_side) as reader:
            meta = reader.metadata
            start = min(max(at_sec, 0.0), max(meta.duration - cfg.window_sec, 0.0))
            end = min(start + cfg.window_sec, meta.duration)
            window = make_window(
                start,
                end,
                fps=meta.fps,
                frame_count=meta.frame_count,
                num_frames=cfg.frames_per_clip,
            )
            decoded = next(reader.read_windows([window]), None)

        if decoded is None:
            raise VideoDecodeError(f"Could not decode {start:.2f}s–{end:.2f}s of {path.name}")

        query = self._encoder.encode([decoded[1]])[0]
        query_id = fingerprint(path)

        def exclude(span: ClipSpan) -> bool:
            if span.video_id != query_id:
                return False
            return not include_same_video or span.overlaps(start, end)

        return self._store.search(query, k, exclude=exclude)
