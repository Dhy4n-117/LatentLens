"""Persistent FAISS store that keeps vectors and clip metadata in lock-step."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from latentlens.types import ClipSpan, SearchHit
from latentlens.utils import atomic_write

SCHEMA_VERSION = 1
INDEX_FILE = "index.faiss"
META_FILE = "metadata.json"


class IncompatibleIndexError(RuntimeError):
    """Raised when an on-disk index was built with a different encoder."""


class VectorStore:
    """Exact cosine-similarity search over unit-norm clip embeddings.

    ``IndexFlatIP`` on normalised vectors == cosine similarity, and is exact.
    That is the right default up to ~1M clips; swap in HNSW/IVF later without
    touching callers.
    """

    def __init__(self, *, dim: int, encoder_name: str) -> None:
        self.dim = dim
        self.encoder_name = encoder_name
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self._spans: dict[int, ClipSpan] = {}
        self._next_id = 0

    def __len__(self) -> int:
        return len(self._spans)

    # -- mutation ----------------------------------------------------------

    def add(self, spans: Sequence[ClipSpan], vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape != (len(spans), self.dim):
            raise ValueError(
                f"Expected vectors of shape ({len(spans)}, {self.dim}), got {vectors.shape}"
            )
        if not spans:
            return

        ids = np.arange(self._next_id, self._next_id + len(spans), dtype=np.int64)
        self._index.add_with_ids(vectors, ids)
        self._spans.update(zip(ids.tolist(), spans, strict=True))
        self._next_id += len(spans)

    def remove_video(self, video_id: str) -> int:
        ids = [i for i, span in self._spans.items() if span.video_id == video_id]
        if ids:
            self._index.remove_ids(np.asarray(ids, dtype=np.int64))
            for i in ids:
                del self._spans[i]
        return len(ids)

    # -- queries -----------------------------------------------------------

    def has_video(self, video_id: str) -> bool:
        return any(span.video_id == video_id for span in self._spans.values())

    def spans(self) -> list[ClipSpan]:
        """All indexed spans in insertion order."""
        return [self._spans[i] for i in sorted(self._spans)]

    def videos(self) -> dict[str, tuple[str, int]]:
        """Map ``video_id -> (path, clip_count)``."""
        summary: dict[str, tuple[str, int]] = {}
        for span in self._spans.values():
            path, count = summary.get(span.video_id, (span.video_path, 0))
            summary[span.video_id] = (path, count + 1)
        return summary

    def search(
        self,
        query: np.ndarray,
        k: int = 5,
        *,
        exclude: Callable[[ClipSpan], bool] | None = None,
    ) -> list[SearchHit]:
        if len(self) == 0 or k < 1:
            return []
        query = np.ascontiguousarray(query, dtype=np.float32).reshape(1, self.dim)

        # Over-fetch when filtering so we can still return k results.
        fetch = min(len(self), k if exclude is None else max(k * 4, k + 32))
        scores, ids = self._index.search(query, fetch)

        hits: list[SearchHit] = []
        for score, idx in zip(scores[0].tolist(), ids[0].tolist(), strict=True):
            if idx < 0:
                continue
            span = self._spans[idx]
            if exclude is not None and exclude(span):
                continue
            hits.append(SearchHit(rank=len(hits) + 1, score=float(score), span=span))
            if len(hits) == k:
                break
        return hits

    # -- persistence -------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        meta: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "encoder": self.encoder_name,
            "dim": self.dim,
            "next_id": self._next_id,
            "clips": [{"id": i, **asdict(s)} for i, s in sorted(self._spans.items())],
        }
        # Serialise via bytes: faiss.write_index can't handle non-ASCII paths on Windows.
        atomic_write(directory / INDEX_FILE, faiss.serialize_index(self._index).tobytes())
        atomic_write(directory / META_FILE, json.dumps(meta, indent=2).encode("utf-8"))

    @classmethod
    def load(cls, directory: str | Path) -> VectorStore:
        directory = Path(directory)
        meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise IncompatibleIndexError(f"Unsupported schema version in {directory}")

        store = cls(dim=int(meta["dim"]), encoder_name=meta["encoder"])
        raw = np.frombuffer((directory / INDEX_FILE).read_bytes(), dtype=np.uint8)
        store._index = faiss.deserialize_index(raw)
        store._next_id = int(meta["next_id"])
        store._spans = {int(c.pop("id")): ClipSpan(**c) for c in meta["clips"]}
        if store._index.ntotal != len(store._spans):
            raise IncompatibleIndexError(f"Index and metadata are out of sync in {directory}")
        return store

    @classmethod
    def open(cls, directory: str | Path, *, dim: int, encoder_name: str) -> VectorStore:
        """Load an existing store or create an empty one, verifying encoder compatibility."""
        directory = Path(directory)
        if not (directory / META_FILE).exists():
            return cls(dim=dim, encoder_name=encoder_name)

        store = cls.load(directory)
        if store.encoder_name != encoder_name or store.dim != dim:
            raise IncompatibleIndexError(
                f"Index at {directory} was built with {store.encoder_name} (dim={store.dim}); "
                f"current encoder is {encoder_name} (dim={dim}). Use a different data dir."
            )
        return store
