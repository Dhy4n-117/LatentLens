"""Gemini adapter implementing ``AgentModel`` with function calling and rate limiting."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

from latentlens.agent.model import Message, ModelTurn, ToolCall
from latentlens.agent.observation import format_tool_response_dict
from latentlens.agent.types import ToolDefinition
from latentlens.llm.ratelimit import RateLimiter

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class AgentModelError(RuntimeError):
    """Raised when the underlying agent LLM call fails."""


class GeminiAgentModel:
    """Gemini function-calling adapter hardened for free-tier rate limits.

    Preserves provider ``Content`` payloads via ``ModelTurn.raw`` / ``Message.raw``
    so multi-turn thought signatures are replayed verbatim across tool steps.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        limiter: RateLimiter,
        temperature: float = 0.1,
        max_output_tokens: int = 1024,
        max_retries: int = 4,
        backoff_sec: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
        client: Any = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if backoff_sec <= 0:
            raise ValueError("backoff_sec must be positive")
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._limiter = limiter
        self._max_retries = max_retries
        self._backoff_sec = backoff_sec
        self._sleep = sleep
        self._client = client if client is not None else _make_client(api_key)

    @property
    def name(self) -> str:
        return f"gemini:{self.model}"

    def generate(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
    ) -> ModelTurn:
        """Run one reasoning step with optional tool declarations."""
        for attempt in range(self._max_retries + 1):
            self._limiter.acquire()
            try:
                return self._call_gemini(system=system, messages=messages, tools=tools)
            except AgentModelError:
                raise
            except Exception as exc:
                status = getattr(exc, "code", None)
                if status not in RETRYABLE_STATUS or attempt == self._max_retries:
                    raise AgentModelError(f"Gemini agent request failed: {exc}") from exc
                delay = self._backoff_sec * 2**attempt
                logger.warning("Gemini returned %s; retrying in %.0fs", status, delay)
                self._sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    def _call_gemini(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
    ) -> ModelTurn:
        from google.genai import types

        contents = _build_contents(messages)
        gemini_tools: list[Any] | None = None
        if tools:
            decls = [types.FunctionDeclaration(**td.to_function_declaration()) for td in tools]
            gemini_tools = [types.Tool(function_declarations=decls)]

        config = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            tools=gemini_tools,
        )

        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        return _parse_response(response)


def _build_contents(messages: Sequence[Message]) -> list[Any]:
    from google.genai import types

    contents: list[Any] = []
    pending_tool_parts: list[Any] = []

    def flush_tool_parts() -> None:
        nonlocal pending_tool_parts
        if pending_tool_parts:
            contents.append(types.Content(role="user", parts=list(pending_tool_parts)))
            pending_tool_parts = []

    for msg in messages:
        if msg.role == "tool":
            tool_name = (
                msg.tool_call.name
                if msg.tool_call is not None
                else (msg.tool_result.tool_name if msg.tool_result is not None else "tool")
            )
            if msg.tool_result is not None:
                resp_dict = format_tool_response_dict(msg.tool_result)
            else:
                resp_dict = {"result": msg.content or ""}
            pending_tool_parts.append(
                types.Part.from_function_response(name=tool_name, response=resp_dict)
            )
            continue

        flush_tool_parts()

        if msg.role == "user":
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=msg.content or "")],
                )
            )
        elif msg.role == "model":
            if msg.raw is not None:
                contents.append(msg.raw)
            else:
                parts: list[Any] = []
                if msg.content:
                    parts.append(types.Part.from_text(text=msg.content))
                for tc in msg.tool_calls:
                    parts.append(
                        types.Part.from_function_call(name=tc.name, args=dict(tc.arguments))
                    )
                contents.append(types.Content(role="model", parts=parts))

    flush_tool_parts()
    return contents


def _parse_response(response: Any) -> ModelTurn:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise AgentModelError("Gemini returned no candidates or text")
        return ModelTurn(text=text)

    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) or []

    text_chunks: list[str] = []
    tool_calls: list[ToolCall] = []

    for part in parts:
        fc = getattr(part, "function_call", None)
        if fc is not None and getattr(fc, "name", None):
            raw_args = getattr(fc, "args", None) or {}
            tool_calls.append(
                ToolCall(
                    name=str(fc.name),
                    arguments=dict(raw_args),
                    call_id=getattr(fc, "id", None),
                )
            )
            continue

        if getattr(part, "thought", False):
            continue

        part_text = getattr(part, "text", None)
        if part_text:
            text_chunks.append(str(part_text))

    joined_text = "\n".join(t.strip() for t in text_chunks if t.strip()) or None
    return ModelTurn(
        text=joined_text,
        tool_calls=tuple(tool_calls),
        raw=content,
    )


def _make_client(api_key: str) -> Any:
    from google import genai

    return genai.Client(api_key=api_key)


__all__ = [
    "AgentModelError",
    "GeminiAgentModel",
]
