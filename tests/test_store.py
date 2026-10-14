from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from latentlens.index import IncompatibleIndexError, VectorStore
from latentlens.types import ClipSpan


def _span(video: str, start: float) -> ClipSpan:
    return ClipSpan(video, f"/videos/{video}.mp4", start, start + 2)


def _unit(*xs: float) -> np.ndarray:
    v = np.asarray(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def filled() -> VectorStore:
    store = VectorStore(dim=3, encoder_name="test")
    store.add(
        [_span("a", 0), _span("a", 2), _span("b", 0)],
        np.stack([_unit(1, 0, 0), _unit(0, 1, 0), _unit(0.9, 0.1, 0)]),
    )
    return store


def test_search_ranks_by_cosine_similarity(filled: VectorStore) -> None:
    hits = filled.search(_unit(1, 0, 0), k=3)

    assert [(h.span.video_id, h.span.start_sec) for h in hits] == [("a", 0), ("b", 0), ("a", 2)]
    assert [h.rank for h in hits] == [1, 2, 3]
    assert hits[0].score == pytest.approx(1.0)


def test_search_respects_exclusion_filter(filled: VectorStore) -> None:
    hits = filled.search(_unit(1, 0, 0), k=2, exclude=lambda s: s.video_id == "a")
    assert [h.span.video_id for h in hits] == ["b"]


def test_remove_video(filled: VectorStore) -> None:
    assert filled.remove_video("a") == 2
    assert len(filled) == 1
    assert not filled.has_video("a")
    assert filled.search(_unit(0, 1, 0), k=5)[0].span.video_id == "b"


def test_add_rejects_wrong_shape() -> None:
    store = VectorStore(dim=3, encoder_name="test")
    with pytest.raises(ValueError):
        store.add([_span("a", 0)], np.zeros((1, 4), dtype=np.float32))


def test_roundtrip_persistence(filled: VectorStore, tmp_path: Path) -> None:
    filled.remove_video("b")
    filled.save(tmp_path)
    loaded = VectorStore.load(tmp_path)

    assert len(loaded) == 2
    assert loaded.videos() == filled.videos()
    # IDs keep increasing after reload — no collisions with removed entries.
    loaded.add([_span("c", 0)], _unit(0, 0, 1)[None])
    assert loaded.search(_unit(0, 0, 1), k=1)[0].span.video_id == "c"


def test_open_refuses_mismatched_encoder(filled: VectorStore, tmp_path: Path) -> None:
    filled.save(tmp_path)
    with pytest.raises(IncompatibleIndexError):
        VectorStore.open(tmp_path, dim=3, encoder_name="other")


def test_open_creates_empty_store_when_missing(tmp_path: Path) -> None:
    store = VectorStore.open(tmp_path / "nope", dim=8, encoder_name="x")
    assert len(store) == 0
    assert store.search(np.ones(8, dtype=np.float32), k=3) == []
