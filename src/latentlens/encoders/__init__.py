"""Video encoders that map clips to L2-normalised embedding vectors."""

from latentlens.encoders.base import TextEncoder, VideoEncoder, l2_normalize
from latentlens.encoders.loss import InfoNCELoss
from latentlens.encoders.projection import LatentProjectionHead
from latentlens.encoders.text import SentenceTransformerTextEncoder
from latentlens.encoders.vjepa import VJEPAEncoder

__all__ = [
    "InfoNCELoss",
    "LatentProjectionHead",
    "SentenceTransformerTextEncoder",
    "TextEncoder",
    "VJEPAEncoder",
    "VideoEncoder",
    "l2_normalize",
]
