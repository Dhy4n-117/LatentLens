"""Video decoding and temporal segmentation."""

from latentlens.video.keyframes import (
    encode_jpeg,
    keyframe_indices,
    keyframe_window,
    select_keyframes,
)
from latentlens.video.reader import VideoDecodeError, VideoMetadata, VideoReader, fingerprint
from latentlens.video.segmenter import FrameWindow, make_window, plan_windows

__all__ = [
    "FrameWindow",
    "VideoDecodeError",
    "VideoMetadata",
    "VideoReader",
    "encode_jpeg",
    "fingerprint",
    "keyframe_indices",
    "keyframe_window",
    "make_window",
    "plan_windows",
    "select_keyframes",
]
