"""Lightweight projection head to map V-JEPA visual embeddings into language space."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentProjectionHead(nn.Module):
    """Linear or 2-layer MLP projection aligning visual embeddings to text embeddings.

    Projects latent vectors from input_dim (e.g. V-JEPA 1024) to output_dim (e.g. text 384),
    with L2-normalization for cosine similarity scoring.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if input_dim < 1 or output_dim < 1:
            raise ValueError("input_dim and output_dim must be positive")

        if hidden_dim is not None:
            if hidden_dim < 1:
                raise ValueError("hidden_dim must be positive")
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.net = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects and L2-normalizes representations."""
        projected = self.net(x)
        return F.normalize(projected, p=2, dim=-1)
