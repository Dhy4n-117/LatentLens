"""End-to-end: real decoding + segmentation + indexing, with a fake encoder."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from latentlens.config import Settings
from latentlens.index import VectorStore
from latentlens.pipeline import IngestionPipeline, SearchService
from latentlens.video import VideoReader, plan_windows
from tests.conftest import FPS, SIZE, ColorEncoder


def test_reader_streams_every_planned_window(rgb_video: Path) -> None:
    with VideoReader(rgb_video, resize_short_side=None) as reader:
        meta = reader.metadata
        windows = plan_windows(
            fps=meta.fps, frame_count=meta.frame_count, window_sec=2, stride_sec=1, num_frames=4
        )
        results = list(reader.read_windows(windows))

    assert meta.fps == FPS
    assert len(results) == len(windows) == 9
    for _, frames in results:
        assert frames.shape == (4, SIZE[1], SIZE[0], 3)
        assert frames.dtype == np.uint8


def test_reader_decodes_colours_as_rgb(rgb_video: Path) -> None:
    with VideoReader(rgb_video, resize_short_side=None) as reader:
        windows = plan_windows(fps=FPS, frame_count=100, window_sec=2, stride_sec=1, num_frames=4)
        _, first = next(reader.read_windows(windows))

    mean = first.reshape(-1, 3).mean(axis=0)
    assert mean[0] > 200 and mean[1] < 50 and mean[2] < 50  # red, not blue


def test_reader_downscales_to_short_side(rgb_video: Path) -> None:
    with VideoReader(rgb_video, resize_short_side=24) as reader:
        windows = plan_windows(fps=FPS, frame_count=100, window_sec=2, stride_sec=1, num_frames=4)
        _, frames = next(reader.read_windows(windows))
    assert frames.shape[1:3] == (24, 32)


def test_ingest_is_idempotent(
    rgb_video: Path, encoder: ColorEncoder, store: VectorStore, settings: Settings
) -> None:
    pipeline = IngestionPipeline(encoder, store, settings)
    progress: list[tuple[int, int]] = []

    first = pipeline.ingest(rgb_video, on_progress=lambda d, t: progress.append((d, t)))
    second = pipeline.ingest(rgb_video)

    assert first.num_clips == 9 and not first.skipped
    assert second.skipped and len(store) == 9
    assert progress[-1] == (9, 9)
    assert encoder.calls == 3  # 9 clips / batch_size 3


def test_force_reingest_replaces_clips(
    rgb_video: Path, encoder: ColorEncoder, store: VectorStore, settings: Settings
) -> None:
    pipeline = IngestionPipeline(encoder, store, settings)
    pipeline.ingest(rgb_video)
    report = pipeline.ingest(rgb_video, force=True)

    assert report.num_clips == 9
    assert len(store) == 9


def test_search_by_moment_finds_matching_scenes(
    rgb_video: Path,
    blue_video: Path,
    encoder: ColorEncoder,
    store: VectorStore,
    settings: Settings,
) -> None:
    pipeline = IngestionPipeline(encoder, store, settings)
    pipeline.ingest(rgb_video)
    pipeline.ingest(blue_video)
    service = SearchService(encoder, store, settings)

    # Query the blue tail of rgb_video: the best matches must be blue clips elsewhere.
    hits = service.search_by_moment(rgb_video, at_sec=8.0, k=3, include_same_video=False)
    assert hits and all(Path(h.span.video_path).name == "blue.avi" for h in hits)
    assert hits[0].score > 0.99

    # Including the same video: overlapping windows of the query are never returned.
    hits = service.search_by_moment(rgb_video, at_sec=0.0, k=5)
    for hit in hits:
        if hit.span.video_path.endswith("rgb.avi"):
            assert not hit.span.overlaps(0.0, 2.0)
    assert hits[0].span.start_sec == 2.0  # still inside the red segment
