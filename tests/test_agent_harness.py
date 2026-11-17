"""Tests for the VideoAgentHarness multi-turn execution loop."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from latentlens.agent.config import HarnessConfig
from latentlens.agent.harness import VideoAgentHarness
from latentlens.agent.memory import ConversationMemory
from latentlens.agent.model import Message, ModelTurn, ToolCall
from latentlens.agent.rewriter import HeuristicQueryRewriter
from latentlens.agent.tools import build_default_registry
from latentlens.agent.types import ToolDefinition
from latentlens.config import Settings
from latentlens.index.captions import CaptionStore
from latentlens.index.store import VectorStore
from latentlens.pipeline.search import SearchService
from latentlens.types import ClipSpan
from tests.conftest import ColorEncoder
from tests.test_text_search import FakeTextEncoder


class ScriptedAgentModel:
    """Deterministic offline ``AgentModel`` that replays a queue of ``ModelTurn``s."""

    def __init__(self, turns: Sequence[ModelTurn]) -> None:
        self._turns = list(turns)
        self.calls: list[dict[str, object]] = []

    @property
    def name(self) -> str:
        return "scripted"

    def generate(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
    ) -> ModelTurn:
        self.calls.append(
            {
                "system": system,
                "messages": list(messages),
                "tools": [t.name for t in tools],
            }
        )
        return self._turns.pop(0)


def _make_harness(
    tmp_path: Path,
    settings: Settings,
    turns: Sequence[ModelTurn],
    config: HarnessConfig | None = None,
) -> VideoAgentHarness:
    store = VectorStore(dim=3, encoder_name="fake")
    s1 = ClipSpan("kitchen", "/videos/kitchen.mp4", 0.0, 4.0)
    s2 = ClipSpan("kitchen", "/videos/kitchen.mp4", 4.0, 8.0)
    store.add([s1, s2], np.ones((2, 3), dtype=np.float32))

    captions = CaptionStore(tmp_path / "captions.json")
    captions.add(s1, "Person walks into the kitchen carrying a blue backpack", "m")
    captions.add(s2, "Person places the blue backpack on the chair", "m")

    service = SearchService(
        encoder=ColorEncoder(),
        store=store,
        settings=settings,
        text_encoder=FakeTextEncoder(),
        captions=captions,
    )
    registry = build_default_registry(service, store, captions)
    model = ScriptedAgentModel(turns)
    return VideoAgentHarness(model=model, registry=registry, config=config)


def test_harness_single_step_direct_answer(tmp_path: Path, settings: Settings) -> None:
    harness = _make_harness(
        tmp_path,
        settings,
        turns=[ModelTurn(text="No tools needed.")],
    )
    answer, trace = harness.run("Hello?")
    assert answer.answer == "No tools needed."
    assert answer.citations == ()
    assert trace.termination_reason == "final_answer"
    assert trace.total_steps == 1
    assert trace.total_tool_calls == 0


def test_harness_multi_step_search_and_temporal_expansion(
    tmp_path: Path, settings: Settings
) -> None:
    turns = [
        ModelTurn(
            tool_calls=(
                ToolCall(
                    name="search_moments",
                    arguments={"query": "blue backpack", "top_k": 1},
                    call_id="c1",
                ),
            )
        ),
        ModelTurn(
            tool_calls=(
                ToolCall(
                    name="get_adjacent_clips",
                    arguments={"video_id": "kitchen", "timestamp": 2.0, "window_sec": 6.0},
                    call_id="c2",
                ),
            )
        ),
        ModelTurn(
            text=(
                "The person enters with the backpack at [kitchen @ 00:00.0–00:04.0] "
                "and sets it on the chair at [kitchen @ 00:04.0–00:08.0]."
            )
        ),
    ]
    harness = _make_harness(tmp_path, settings, turns=turns)
    answer, trace = harness.run("Where did the blue backpack end up?")

    assert trace.termination_reason == "final_answer"
    assert trace.total_steps == 3
    assert trace.total_tool_calls == 2
    assert len(answer.citations) == 2
    assert answer.citations[0].start_sec == 0.0
    assert answer.citations[1].start_sec == 4.0
    assert len(answer.hits) == 1


def test_harness_budget_exhaustion_triggers_synthesis(tmp_path: Path, settings: Settings) -> None:
    turns = [
        ModelTurn(tool_calls=(ToolCall(name="list_videos", arguments={}),)),
        ModelTurn(
            tool_calls=(
                ToolCall(name="search_moments", arguments={"query": "backpack", "top_k": 1}),
            )
        ),
        ModelTurn(text="Synthesized from budget limit: [kitchen @ 00:00.0–00:04.0]."),
    ]
    harness = _make_harness(
        tmp_path,
        settings,
        turns=turns,
        config=HarnessConfig(max_steps=2),
    )
    answer, trace = harness.run("What happened?")
    assert trace.termination_reason == "max_steps_reached"
    assert trace.total_steps == 3  # 2 tool steps + 1 synthesis step
    assert "Synthesized from budget limit" in answer.answer
    assert len(answer.citations) == 1


def test_harness_caps_tool_calls_per_step_and_handles_unknown_tool(
    tmp_path: Path, settings: Settings
) -> None:
    turns = [
        ModelTurn(
            tool_calls=(
                ToolCall(name="nonexistent_tool", arguments={}),
                ToolCall(name="list_videos", arguments={}),
                ToolCall(name="search_moments", arguments={"query": "extra"}),
            )
        ),
        ModelTurn(text="Recovered after unknown tool."),
    ]
    harness = _make_harness(
        tmp_path,
        settings,
        turns=turns,
        config=HarnessConfig(max_steps=3, max_tool_calls_per_step=2),
    )
    answer, trace = harness.run("Test cap and recovery")
    assert answer.answer == "Recovered after unknown tool."
    assert trace.steps[0].num_tool_calls == 2
    assert trace.steps[0].tool_results[0].is_error is True
    assert trace.steps[0].tool_results[1].is_error is False


def test_harness_rejects_empty_query(tmp_path: Path, settings: Settings) -> None:
    harness = _make_harness(tmp_path, settings, turns=[])
    with pytest.raises(ValueError, match="empty"):
        harness.run("   ")


def test_harness_guards_against_duplicate_tool_calls(tmp_path: Path, settings: Settings) -> None:
    dup_call = ToolCall(name="search_moments", arguments={"query": "backpack", "top_k": 2})
    turns = [
        ModelTurn(tool_calls=(dup_call,)),
        ModelTurn(tool_calls=(dup_call,)),
        ModelTurn(text="Done at [kitchen @ 00:00.0–00:04.0]."),
    ]
    harness = _make_harness(tmp_path, settings, turns=turns)
    answer, trace = harness.run("Check duplicate guard")
    assert trace.steps[0].tool_results[0].is_error is False
    assert trace.steps[1].tool_results[0].is_error is True
    assert "Duplicate call" in str(trace.steps[1].tool_results[0].output)
    assert len(answer.citations) == 1


def test_harness_integrates_memory_and_query_rewriter(tmp_path: Path, settings: Settings) -> None:
    store = VectorStore(dim=3, encoder_name="fake")
    s1 = ClipSpan("kitchen", "/videos/kitchen.mp4", 0.0, 4.0)
    store.add([s1], np.ones((1, 3), dtype=np.float32))
    captions = CaptionStore(tmp_path / "captions.json")
    captions.add(s1, "Person puts blue backpack on chair", "m")
    service = SearchService(
        encoder=ColorEncoder(),
        store=store,
        settings=settings,
        text_encoder=FakeTextEncoder(),
        captions=captions,
    )
    registry = build_default_registry(service, store, captions)
    model = ScriptedAgentModel(
        [
            ModelTurn(
                tool_calls=(
                    ToolCall(name="search_moments", arguments={"query": "backpack", "top_k": 1}),
                )
            ),
            ModelTurn(text="It is on the chair at [kitchen @ 00:00.0–00:04.0]."),
            ModelTurn(text="Follow-up answered at [kitchen @ 00:00.0–00:04.0]."),
        ]
    )
    memory = ConversationMemory()
    harness = VideoAgentHarness(
        model=model,
        registry=registry,
        memory=memory,
        rewriter=HeuristicQueryRewriter(),
    )

    harness.run("Where did the blue backpack go?")
    assert len(memory) == 1
    assert memory.last_turn is not None
    assert len(memory.last_turn.citations) == 1

    harness.run("What color is it?")
    assert len(memory) == 2
    assert "context: Where did the blue backpack go?" in memory.last_turn.rewritten_query
