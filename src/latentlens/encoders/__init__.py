"""Video encoders that map clips to L2-normalised embedding vectors."""

from latentlens.encoders.base import VideoEncoder, l2_normalize
from latentlens.encoders.vjepa import VJEPAEncoder

__all__ = ["VJEPAEncoder", "VideoEncoder", "l2_normalize"]
