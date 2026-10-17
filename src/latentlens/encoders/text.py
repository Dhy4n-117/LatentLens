"""Local, free sentence-transformers text encoder for captions and queries."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import cached_property
from typing import Any

import numpy as np

from latentlens.encoders.base import l2_normalize

logger = logging.getLogger(__name__)


class SentenceTransformerTextEncoder:
    """Local text encoder using sentence-transformers (default: all-MiniLM-L6-v2).

    Embeddings are L2-normalized float32 vectors suitable for cosine search.
    Lazy-loads weights on first call.
    """

    def __init__(self, model_id: str = "all-MiniLM-L6-v2", *, device: str = "auto") -> None:
        self._model_id = model_id
        self._requested_device = device
        self._model: Any = None

    @property
    def name(self) -> str:
        return f"st:{self._model_id}"

    @cached_property
    def dim(self) -> int:
        self._ensure_loaded()
        return int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        self._ensure_loaded()
        embeddings = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return l2_normalize(np.asarray(embeddings, dtype=np.float32))

    @cached_property
    def device(self) -> str:
        import torch

        if self._requested_device != "auto":
            return self._requested_device
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        logger.info("Loading text encoder %s on %s...", self._model_id, self.device)
        self._model = SentenceTransformer(self._model_id, device=self.device)
