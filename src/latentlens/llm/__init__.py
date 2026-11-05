"""Free-tier-safe plumbing for LLM calls: rate limiting, caching and captioning."""

from latentlens.llm.cache import ResponseCache
from latentlens.llm.captioner import CAPTION_PROMPT, Captioner, CaptionError, GeminiCaptioner
from latentlens.llm.prompts import QA_SYSTEM_PROMPT, build_qa_prompt, format_qa_context
from latentlens.llm.ratelimit import RateLimiter

__all__ = [
    "CAPTION_PROMPT",
    "QA_SYSTEM_PROMPT",
    "CaptionError",
    "Captioner",
    "GeminiCaptioner",
    "RateLimiter",
    "ResponseCache",
    "build_qa_prompt",
    "format_qa_context",
]
