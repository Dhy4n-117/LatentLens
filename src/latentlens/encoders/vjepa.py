"""V-JEPA 2 video encoder (Meta, open weights) via Hugging Face ``transformers``."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from latentlens.encoders.base import l2_normalize

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class VJEPAEncoder:
    """Frozen V-JEPA 2 backbone producing one embedding per clip.

    The encoder emits a grid of spatio-temporal patch tokens; we mean-pool them
    into a single clip vector. Heavy imports and weight loading are deferred
    until first use so CLI commands that don't need the model stay instant.
    """

    def __init__(self, model_id: str, *, device: str = "auto") -> None:
        self._model_id = model_id
        self._requested_device = device
        self._model: Any = None
        self._processor: Any = None

    # -- VideoEncoder protocol ---------------------------------------------

    @property
    def name(self) -> str:
        return f"vjepa2:{self._model_id}"

    @cached_property
    def dim(self) -> int:
        # Read from config only — no need to download/load weights for this.
        from transformers import AutoConfig

        return int(AutoConfig.from_pretrained(self._model_id).hidden_size)

    def encode(self, clips: Sequence[np.ndarray]) -> np.ndarray:
        if not clips:
            return np.empty((0, self.dim), dtype=np.float32)

        import torch

        self._ensure_loaded()
        videos = [torch.from_numpy(np.ascontiguousarray(c)).permute(0, 3, 1, 2) for c in clips]
        inputs = self._processor(videos, return_tensors="pt")
        inputs = {
            k: v.to(self.device, self.dtype) if v.is_floating_point() else v.to(self.device)
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            tokens = self._model.get_vision_features(**inputs)  # (B, N, D)

        pooled = tokens.float().mean(dim=1).cpu().numpy()
        return l2_normalize(pooled)

    # -- internals ---------------------------------------------------------

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

    @cached_property
    def dtype(self) -> torch.dtype:
        import torch

        return torch.float16 if self.device == "cuda" else torch.float32

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoVideoProcessor

        logger.info("Loading %s on %s (%s)…", self._model_id, self.device, self.dtype)
        self._processor = AutoVideoProcessor.from_pretrained(self._model_id)
        self._model = (
            AutoModel.from_pretrained(self._model_id, torch_dtype=self.dtype).to(self.device).eval()
        )
