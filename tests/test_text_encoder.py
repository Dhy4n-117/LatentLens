from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from latentlens.encoders.base import TextEncoder
from latentlens.encoders.text import SentenceTransformerTextEncoder


def test_satisfies_text_encoder_protocol() -> None:
    encoder = SentenceTransformerTextEncoder("fake-model")
    with patch.object(SentenceTransformerTextEncoder, "_ensure_loaded"):
        encoder._model = MagicMock()
        encoder._model.get_sentence_embedding_dimension.return_value = 384
        assert isinstance(encoder, TextEncoder)
        assert encoder.name == "st:fake-model"
        assert encoder.dim == 384


def test_text_encoder_empty() -> None:
    encoder = SentenceTransformerTextEncoder("fake-model")
    with patch.object(SentenceTransformerTextEncoder, "_ensure_loaded"):
        encoder._model = MagicMock()
        encoder._model.get_sentence_embedding_dimension.return_value = 384
        out = encoder.encode([])
        assert out.shape == (0, 384)
        assert out.dtype == np.float32


def test_text_encoder_encode_mocked() -> None:
    encoder = SentenceTransformerTextEncoder("fake-model")
    with patch.object(SentenceTransformerTextEncoder, "_ensure_loaded"):
        encoder._model = MagicMock()
        encoder._model.get_sentence_embedding_dimension.return_value = 3
        encoder._model.encode.return_value = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        out = encoder.encode(["hello", "world"])
        assert out.shape == (2, 3)
        assert np.isclose(np.linalg.norm(out[0]), 1.0)
        assert np.isclose(np.linalg.norm(out[1]), 1.0)
        assert np.allclose(out[1], [0.0, 1.0, 0.0])
