from __future__ import annotations

import pytest
import torch

from latentlens.encoders.projection import LatentProjectionHead


def test_latent_projection_head_linear() -> None:
    head = LatentProjectionHead(input_dim=1024, output_dim=384)
    x = torch.randn(4, 1024)
    out = head(x)
    assert out.shape == (4, 384)
    norms = torch.norm(out, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_latent_projection_head_mlp() -> None:
    head = LatentProjectionHead(input_dim=512, output_dim=128, hidden_dim=256)
    x = torch.randn(2, 512)
    out = head(x)
    assert out.shape == (2, 128)
    norms = torch.norm(out, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_latent_projection_head_validation() -> None:
    with pytest.raises(ValueError, match="input_dim and output_dim"):
        LatentProjectionHead(input_dim=0, output_dim=10)
    with pytest.raises(ValueError, match="hidden_dim"):
        LatentProjectionHead(input_dim=10, output_dim=10, hidden_dim=0)
