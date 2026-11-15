"""Multi-turn execution loop for the video question-answering agent."""

from __future__ import annotations

import json
import time

from latentlens.agent.citations import verify_citations
from latentlens.agent.config import HarnessConfig
from latentlens.agent.model import AgentModel, Message, ToolCall
from latentlens.agent.observation import format_observation
from latentlens.agent.prompts import AGENT_SYSTEM_PROMPT, build_synthesis_prompt
from latentlens.agent.registry import ToolRegistry
from latentlens.agent.trace import AgentStep, AgentTrace
from latentlens.agent.types import ToolResult
from latentlens.types import ClipSpan, GroundedAnswer, SearchHit


class VideoAgentHarness:
    """Orchestrates multi-step tool calling and grounded answer synthesis."""

    def __init__(
        self,
        model: AgentModel,
        registry: ToolRegistry,
        config: HarnessConfig | None = None,
        system_prompt: str = AGENT_SYSTEM_PROMPT,
    ) -> None:
        self._model = model
        self._registry = registry
        self._config = config or HarnessConfig()
        self._system_prompt = system_prompt

    @property
    def config(self) -> HarnessConfig:
        return self._config

    def run(self, query: str) -> tuple[GroundedAnswer, AgentTrace]:
        """Execute the agent reasoning loop for ``query`` and return answer + trace."""
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Query must not be empty")

        run_start = time.perf_counter()
        messages: list[Message] = [Message.user(cleaned_query)]
        tool_defs = self._registry.get_definitions()

        steps: list[AgentStep] = []
        discovered_spans: list[ClipSpan] = []
        discovered_hits: list[SearchHit] = []
        seen_calls: set[tuple[str, str]] = set()

        for step_idx in range(self._config.max_steps):
            step_start = time.perf_counter()
            turn = self._model.generate(
                system=self._system_prompt,
                messages=messages,
                tools=tool_defs,
            )
            messages.append(Message.from_turn(turn))

            if turn.is_final:
                elapsed_ms = (time.perf_counter() - step_start) * 1000.0
                answer_text = (turn.text or "").strip()
                steps.append(
                    AgentStep(
                        step_index=step_idx,
                        text=answer_text,
                        elapsed_ms=elapsed_ms,
                    )
                )
                total_ms = (time.perf_counter() - run_start) * 1000.0
                citations = verify_citations(answer_text, discovered_spans)
                return (
                    GroundedAnswer(
                        query=cleaned_query,
                        answer=answer_text,
                        citations=citations,
                        hits=tuple(discovered_hits),
                    ),
                    AgentTrace(
                        query=cleaned_query,
                        steps=steps,
                        termination_reason="final_answer",
                        total_elapsed_ms=total_ms,
                    ),
                )

            bounded_calls: tuple[ToolCall, ...] = turn.tool_calls[
                : self._config.max_tool_calls_per_step
            ]
            step_results: list[ToolResult] = []

            for call in bounded_calls:
                sig = (
                    call.name,
                    json.dumps(call.arguments, sort_keys=True, default=str),
                )
                if sig in seen_calls:
                    result = ToolResult(
                        tool_name=call.name,
                        output=(
                            f"Duplicate call to '{call.name}' with identical arguments "
                            "skipped; use the previous observation or refine arguments."
                        ),
                        is_error=True,
                    )
                else:
                    seen_calls.add(sig)
                    result = self._registry.dispatch(call.name, dict(call.arguments))
                    _collect_evidence(result, discovered_spans, discovered_hits)
                step_results.append(result)
                rendered = format_observation(
                    result,
                    max_chars=self._config.max_observation_chars,
                )
                messages.append(Message.tool(call, result, rendered))

            elapsed_ms = (time.perf_counter() - step_start) * 1000.0
            steps.append(
                AgentStep(
                    step_index=step_idx,
                    text=(turn.text or "").strip(),
                    tool_calls=bounded_calls,
                    tool_results=tuple(step_results),
                    elapsed_ms=elapsed_ms,
                )
            )

        # Budget exhausted: force final synthesis without tools
        synth_start = time.perf_counter()
        messages.append(Message.user(build_synthesis_prompt(cleaned_query)))
        synth_turn = self._model.generate(
            system=self._system_prompt,
            messages=messages,
            tools=(),
        )
        synth_elapsed_ms = (time.perf_counter() - synth_start) * 1000.0
        final_text = (synth_turn.text or "").strip()
        steps.append(
            AgentStep(
                step_index=len(steps),
                text=final_text,
                elapsed_ms=synth_elapsed_ms,
            )
        )
        total_ms = (time.perf_counter() - run_start) * 1000.0
        citations = verify_citations(final_text, discovered_spans)
        return (
            GroundedAnswer(
                query=cleaned_query,
                answer=final_text,
                citations=citations,
                hits=tuple(discovered_hits),
            ),
            AgentTrace(
                query=cleaned_query,
                steps=steps,
                termination_reason="max_steps_reached",
                total_elapsed_ms=total_ms,
            ),
        )


def _collect_evidence(
    result: ToolResult,
    spans: list[ClipSpan],
    hits: list[SearchHit],
) -> None:
    """Extract ``ClipSpan`` and ``SearchHit`` records from tool output dictionaries."""
    if result.is_error or not isinstance(result.output, list):
        return

    seen_keys = {(s.video_id, s.start_sec, s.end_sec) for s in spans}
    for item in result.output:
        if not isinstance(item, dict):
            continue
        video_id = item.get("video_id")
        start_sec = item.get("start_sec")
        end_sec = item.get("end_sec")
        if video_id is None or start_sec is None or end_sec is None:
            continue

        video_path = str(item.get("video_path") or item.get("video") or video_id)
        span = ClipSpan(
            video_id=str(video_id),
            video_path=video_path,
            start_sec=float(start_sec),
            end_sec=float(end_sec),
        )
        key = (span.video_id, span.start_sec, span.end_sec)
        if key not in seen_keys:
            seen_keys.add(key)
            spans.append(span)

        if "score" in item and "rank" in item:
            hits.append(
                SearchHit(
                    rank=int(item["rank"]),
                    score=float(item["score"]),
                    span=span,
                    caption=item.get("caption"),
                )
            )


__all__ = [
    "VideoAgentHarness",
]
