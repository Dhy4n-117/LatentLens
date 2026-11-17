"""Free-tier-safe plumbing for LLM calls: rate limiting, caching, captioning, and QA."""

from latentlens.llm.cache import ResponseCache
from latentlens.llm.captioner import CAPTION_PROMPT, Captioner, CaptionError, GeminiCaptioner
from latentlens.llm.prompts import QA_SYSTEM_PROMPT, build_qa_prompt, format_qa_context
from latentlens.llm.ratelimit import RateLimiter
from latentlens.llm.reasoner import GeminiQAReasoner, QAReasoner, QAReasonerError

__all__ = [
    "CAPTION_PROMPT",
    "QA_SYSTEM_PROMPT",
    "CaptionError",
    "Captioner",
    "GeminiCaptioner",
    "GeminiQAReasoner",
    "QAReasoner",
    "QAReasonerError",
    "RateLimiter",
    "ResponseCache",
    "build_qa_prompt",
    "format_qa_context",
]
