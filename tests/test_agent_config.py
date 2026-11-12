from __future__ import annotations

import pytest

from latentlens.agent.config import HarnessConfig


def test_defaults_are_bounded() -> None:
    cfg = HarnessConfig()
    assert cfg.max_steps == 6
    assert cfg.max_tool_calls_per_step == 4
    assert cfg.max_observation_chars == 4000


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_steps": 0},
        {"max_tool_calls_per_step": 0},
        {"max_observation_chars": 50},
    ],
)
def test_rejects_invalid_budgets(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        HarnessConfig(**kwargs)
