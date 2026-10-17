from __future__ import annotations

from pathlib import Path

import pytest

from latentlens.config import Settings
from latentlens.index import CaptionStore
from latentlens.pipeline.caption import CaptioningPipeline
from latentlens.types import ClipSpan
from tests.conftest import BLUE, GREEN, RED, write_video


class DummyCaptioner:
    name = "dummy:test"

    def __init__(self) -> None:
        self.called = 0

    def caption(self, images: list[bytes]) -> str:
        self.called += 1
        return f"Scene with {len(images)} keyframes"


@pytest.fixture
def caption_store(tmp_path: Path) -> CaptionStore:
    return CaptionStore(tmp_path / "captions.json")


def test_caption_store_roundtrip(caption_store: CaptionStore) -> None:
    span = ClipSpan("vid1", "/path/vid1.mp4", 0.0, 4.0)
    caption_store.add(span, "A person dancing", "model-x")
    assert span in caption_store
    assert caption_store.get(span) is not None
    assert caption_store.get(span).text == "A person dancing"  # type: ignore[union-attr]

    caption_store.save()

    loaded = CaptionStore.load(caption_store.path)
    assert len(loaded) == 1
    assert span in loaded
    assert loaded.get(span).text == "A person dancing"  # type: ignore[union-attr]


def test_caption_store_remove_video(caption_store: CaptionStore) -> None:
    s1 = ClipSpan("v1", "/path/v1.mp4", 0.0, 4.0)
    s2 = ClipSpan("v1", "/path/v1.mp4", 4.0, 8.0)
    s3 = ClipSpan("v2", "/path/v2.mp4", 0.0, 4.0)

    caption_store.add(s1, "c1", "m")
    caption_store.add(s2, "c2", "m")
    caption_store.add(s3, "c3", "m")

    removed = caption_store.remove_video("v1")
    assert removed == 2
    assert len(caption_store) == 1
    assert s1 not in caption_store
    assert s3 in caption_store


def test_captioning_pipeline_e2e(tmp_path: Path, caption_store: CaptionStore) -> None:
    vid = write_video(tmp_path / "test.avi", [(RED, 2), (GREEN, 2), (BLUE, 2)])
    settings = Settings(
        data_dir=tmp_path / "data",
        keyframes_per_clip=3,
        _env_file=None,  # type: ignore[call-arg]
    )

    captioner = DummyCaptioner()
    pipeline = CaptioningPipeline(captioner, caption_store, settings)

    spans = [
        ClipSpan("test_id", str(vid), 0.0, 2.0),
        ClipSpan("test_id", str(vid), 2.0, 4.0),
    ]

    report = pipeline.run(spans)
    assert report.captioned == 2
    assert report.already_done == 0
    assert report.failed == 0
    assert report.missing_video == 0
    assert len(caption_store) == 2

    # Running again is idempotent
    second_report = pipeline.run(spans)
    assert second_report.already_done == 2
    assert second_report.captioned == 0
    assert captioner.called == 2
