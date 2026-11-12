"""Runtime budgets for the agent harness."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HarnessConfig:
    """Hard limits that keep an agent run bounded in cost and latency.

    Free-tier quotas make runaway loops expensive, so every dimension of the
    loop is capped: reasoning steps, tool calls per step, and the size of each
    observation fed back to the model.
    """

    max_steps: int = 6
    max_tool_calls_per_step: int = 4
    max_observation_chars: int = 4000

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.max_tool_calls_per_step < 1:
            raise ValueError("max_tool_calls_per_step must be >= 1")
        if self.max_observation_chars < 200:
            raise ValueError("max_observation_chars must be >= 200")
