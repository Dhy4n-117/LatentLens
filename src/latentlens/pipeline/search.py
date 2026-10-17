"""Query the visual memory."""

from __future__ import annotations

from pathlib import Path

from latentlens.config import Settings
from latentlens.encoders.base import TextEncoder, VideoEncoder
from latentlens.index.captions import CaptionStore
from latentlens.index.store import VectorStore
from latentlens.types import ClipSpan, SearchHit
from latentlens.video import VideoDecodeError, VideoReader, fingerprint, make_window


class SearchService:
    """Multi-modal search service: search by visual moment or text queries."""

    def __init__(
        self,
        encoder: VideoEncoder,
        store: VectorStore,
        settings: Settings,
        *,
        text_encoder: TextEncoder | None = None,
        captions: CaptionStore | None = None,
        text_store: VectorStore | None = None,
    ) -> None:
        self._encoder = encoder
        self._store = store
        self._settings = settings
        self._text_encoder = text_encoder
        self._captions = captions
        self._text_store = text_store

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

        hits = self._store.search(query, k, exclude=exclude)
        if self._captions is not None:
            hits = [
                SearchHit(
                    rank=h.rank,
                    score=h.score,
                    span=h.span,
                    caption=(
                        self._captions.get(h.span).text if self._captions.get(h.span) else None
                    ),
                )
                for h in hits
            ]
        return hits

    def search_by_text(
        self,
        query_text: str,
        *,
        k: int = 5,
    ) -> list[SearchHit]:
        """Search clips by textual query using local text embeddings over clip captions."""
        if not query_text.strip():
            raise ValueError("query_text cannot be empty")
        if self._text_encoder is None or self._captions is None:
            raise RuntimeError("search_by_text requires text_encoder and captions store")

        if len(self._captions) == 0:
            return []

        # If text_store is provided, search directly
        if self._text_store is not None and len(self._text_store) > 0:
            query_vec = self._text_encoder.encode([query_text])[0]
            hits = self._text_store.search(query_vec, k=k)
            return [
                SearchHit(
                    rank=h.rank,
                    score=h.score,
                    span=h.span,
                    caption=(
                        self._captions.get(h.span).text if self._captions.get(h.span) else None
                    ),
                )
                for h in hits
            ]

        # On-the-fly caption similarity if text_store is not populated
        captions_list = list(self._captions)
        texts = [c.text for c in captions_list]
        query_vec = self._text_encoder.encode([query_text])[0]
        text_vecs = self._text_encoder.encode(texts)

        scores = text_vecs @ query_vec
        top_indices = scores.argsort()[::-1][:k]

        return [
            SearchHit(
                rank=i + 1,
                score=float(scores[idx]),
                span=captions_list[idx].span,
                caption=captions_list[idx].text,
            )
            for i, idx in enumerate(top_indices)
        ]
