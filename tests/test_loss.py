from __future__ import annotations

import pytest
import torch

from latentlens.encoders.loss import InfoNCELoss


def test_infonce_loss_computes_loss() -> None:
    loss_fn = InfoNCELoss(temperature=0.07)
    # Perfectly aligned features
    v = torch.eye(4)
    t = torch.eye(4)
    loss = loss_fn(v, t)
    assert loss.item() >= 0.0
    # Misaligned features should have higher loss
    t_perm = t[[1, 0, 3, 2]]
    loss_perm = loss_fn(v, t_perm)
    assert loss_perm.item() > loss.item()


def test_infonce_loss_validation() -> None:
    with pytest.raises(ValueError, match="temperature must be positive"):
        InfoNCELoss(temperature=0.0)

    loss_fn = InfoNCELoss()
    with pytest.raises(ValueError, match="Shape mismatch"):
        loss_fn(torch.randn(2, 4), torch.randn(2, 6))

    with pytest.raises(ValueError, match="batch_size must be >= 2"):
        loss_fn(torch.randn(1, 4), torch.randn(1, 4))
