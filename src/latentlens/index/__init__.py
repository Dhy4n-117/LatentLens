"""Vector storage, caption storage and nearest-neighbour search."""

from latentlens.index.captions import Caption, CaptionStore
from latentlens.index.store import IncompatibleIndexError, VectorStore

__all__ = ["Caption", "CaptionStore", "IncompatibleIndexError", "VectorStore"]
