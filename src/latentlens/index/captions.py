"""Persistent clip captions, stored next to the vector index."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from latentlens.types import ClipSpan
from latentlens.utils import atomic_write

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class Caption:
    span: ClipSpan
    text: str
    model: str


class CaptionStore:
    """Caption-per-clip mapping persisted as a single JSON file.

    Captions cost free-tier quota, so the store is saved frequently and
    atomically by its callers and is keyed independently of FAISS row IDs.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._items: dict[str, Caption] = {}

    @staticmethod
    def key(span: ClipSpan) -> str:
        return f"{span.video_id}:{span.start_sec:.3f}:{span.end_sec:.3f}"

    # -- access ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, span: object) -> bool:
        return isinstance(span, ClipSpan) and self.key(span) in self._items

    def __iter__(self) -> Iterator[Caption]:
        return iter(
            sorted(self._items.values(), key=lambda c: (c.span.video_path, c.span.start_sec))
        )

    def get(self, span: ClipSpan) -> Caption | None:
        return self._items.get(self.key(span))

    # -- mutation ----------------------------------------------------------

    def add(self, span: ClipSpan, text: str, model: str) -> Caption:
        caption = Caption(span=span, text=text, model=model)
        self._items[self.key(span)] = caption
        return caption

    def remove_video(self, video_id: str) -> int:
        stale = [k for k, c in self._items.items() if c.span.video_id == video_id]
        for k in stale:
            del self._items[k]
        return len(stale)

    # -- persistence -------------------------------------------------------

    def save(self) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "captions": [{**asdict(c.span), "text": c.text, "model": c.model} for c in self],
        }
        atomic_write(self.path, json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8"))

    @classmethod
    def load(cls, path: str | Path) -> CaptionStore:
        """Load captions, or return an empty store if the file doesn't exist yet."""
        store = cls(path)
        if not store.path.exists():
            return store
        payload = json.loads(store.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported captions schema in {store.path}")
        for record in payload["captions"]:
            text, model = record.pop("text"), record.pop("model")
            store.add(ClipSpan(**record), text, model)
        return store
