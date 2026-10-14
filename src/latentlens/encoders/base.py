"""The encoder contract every backbone (V-JEPA, CLIP baseline, fakes) must satisfy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VideoEncoder(Protocol):
    """Maps clips of shape ``(T, H, W, 3)`` uint8 RGB to unit-norm float32 vectors."""

    @property
    def name(self) -> str:
        """Stable identifier persisted with the index to prevent mixing embedding spaces."""
        ...

    @property
    def dim(self) -> int: ...

    def encode(self, clips: Sequence[np.ndarray]) -> np.ndarray:
        """Return an array of shape ``(len(clips), dim)``, dtype float32, rows L2-normalised."""
        ...


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, eps)
