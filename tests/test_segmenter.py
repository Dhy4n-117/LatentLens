from __future__ import annotations

import pytest

from latentlens.video import make_window, plan_windows


def test_windows_cover_whole_video_including_tail() -> None:
    windows = plan_windows(fps=10, frame_count=105, window_sec=4, stride_sec=2, num_frames=8)

    assert windows[0].start_sec == 0.0
    assert windows[-1].end_sec == pytest.approx(10.5)
    assert [w.start_sec for w in windows] == [0.0, 2.0, 4.0, 6.0, 6.5]


def test_last_frames_are_monotonic_and_in_bounds() -> None:
    windows = plan_windows(fps=30, frame_count=1000, window_sec=3, stride_sec=1.5, num_frames=16)

    last_frames = [w.last_frame for w in windows]
    assert last_frames == sorted(last_frames)
    assert all(0 <= i < 1000 for w in windows for i in w.frame_indices)
    assert all(len(w.frame_indices) == 16 for w in windows)


def test_short_video_yields_single_padded_window() -> None:
    windows = plan_windows(fps=10, frame_count=5, window_sec=4, stride_sec=2, num_frames=8)

    assert len(windows) == 1
    assert len(windows[0].frame_indices) == 8
    assert max(windows[0].frame_indices) == 4


def test_make_window_samples_uniformly() -> None:
    window = make_window(1.0, 2.0, fps=10, frame_count=100, num_frames=4)
    assert window.frame_indices == (10, 13, 16, 19)


@pytest.mark.parametrize("bad", [{"fps": 0}, {"frame_count": 0}, {"num_frames": 0}])
def test_make_window_rejects_invalid_input(bad: dict[str, int]) -> None:
    kwargs = {"fps": 10, "frame_count": 10, "num_frames": 4} | bad
    with pytest.raises(ValueError):
        make_window(0, 1, **kwargs)
