from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from latentlens.config import Settings
from latentlens.encoders.base import l2_normalize
from latentlens.index import CaptionStore, VectorStore
from latentlens.pipeline.search import SearchService
from latentlens.types import ClipSpan
from tests.conftest import ColorEncoder


class FakeTextEncoder:
    name = "fake:text"
    dim = 3

    def encode(self, texts: list[str]) -> np.ndarray:
        # Map word content to simple 3D vector
        vecs = []
        for t in texts:
            t_lower = t.lower()
            r = 1.0 if "red" in t_lower or "apple" in t_lower else 0.1
            g = 1.0 if "green" in t_lower or "grass" in t_lower else 0.1
            b = 1.0 if "blue" in t_lower or "sky" in t_lower else 0.1
            vecs.append([r, g, b])
        return l2_normalize(np.array(vecs, dtype=np.float32))


def test_search_by_text_ranks_captions(tmp_path: Path, settings: Settings) -> None:
    captions = CaptionStore(tmp_path / "captions.json")
    span_red = ClipSpan("vid_red", "/vids/red.mp4", 0.0, 4.0)
    span_green = ClipSpan("vid_green", "/vids/green.mp4", 0.0, 4.0)
    span_blue = ClipSpan("vid_blue", "/vids/blue.mp4", 0.0, 4.0)

    captions.add(span_red, "A shiny red apple on a desk", "gemini-flash")
    captions.add(span_green, "Lush green grass in a field", "gemini-flash")
    captions.add(span_blue, "Deep blue sky with clouds", "gemini-flash")

    dummy_video_encoder = ColorEncoder()
    dummy_vector_store = VectorStore(dim=3, encoder_name=dummy_video_encoder.name)
    text_encoder = FakeTextEncoder()

    service = SearchService(
        encoder=dummy_video_encoder,
        store=dummy_vector_store,
        settings=settings,
        text_encoder=text_encoder,
        captions=captions,
    )

    hits = service.search_by_text("red apple", k=2)
    assert len(hits) == 2
    assert hits[0].span.video_id == "vid_red"
    assert hits[0].caption == "A shiny red apple on a desk"
    assert hits[0].score > hits[1].score


def test_search_by_text_empty_captions(tmp_path: Path, settings: Settings) -> None:
    captions = CaptionStore(tmp_path / "captions.json")
    service = SearchService(
        encoder=ColorEncoder(),
        store=VectorStore(dim=3, encoder_name="fake"),
        settings=settings,
        text_encoder=FakeTextEncoder(),
        captions=captions,
    )
    assert service.search_by_text("anything") == []


def test_search_by_text_requires_query(tmp_path: Path, settings: Settings) -> None:
    captions = CaptionStore(tmp_path / "captions.json")
    service = SearchService(
        encoder=ColorEncoder(),
        store=VectorStore(dim=3, encoder_name="fake"),
        settings=settings,
        text_encoder=FakeTextEncoder(),
        captions=captions,
    )
    with pytest.raises(ValueError):
        service.search_by_text("   ")
