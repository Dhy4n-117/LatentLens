"""Contrastive InfoNCE loss for training visual-language projection heads."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE loss for contrastive learning between video and text latents.

    Given batch of normalized video embeddings and text embeddings, aligns positive pairs
    along the diagonal and contrasts against batch in-negative samples.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(self, video_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """Compute symmetric cross-entropy contrastive loss.

        Args:
            video_embeds: (B, D) tensor of normalized video embeddings.
            text_embeds: (B, D) tensor of normalized text embeddings.
        """
        if video_embeds.shape != text_embeds.shape:
            raise ValueError(
                f"Shape mismatch: {video_embeds.shape} vs {text_embeds.shape}"
            )
        batch_size = video_embeds.shape[0]
        if batch_size < 2:
            raise ValueError("batch_size must be >= 2 for contrastive negatives")

        logits = torch.matmul(video_embeds, text_embeds.T) / self.temperature
        labels = torch.arange(batch_size, device=video_embeds.device)

        loss_v2t = F.cross_entropy(logits, labels)
        loss_t2v = F.cross_entropy(logits.T, labels)
        return (loss_v2t + loss_t2v) / 2.0
