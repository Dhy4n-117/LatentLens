"""Content-addressed on-disk cache for LLM responses.

Free-tier quotas are the scarcest resource in this project: the same keyframes
must never be captioned twice. Keys hash *everything* that affects the output
(model, prompt, image bytes), so changing any of them is an automatic miss.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from latentlens.utils import atomic_write


class ResponseCache:
    """A tiny, dependency-free key/value store laid out as ``<dir>/<ab>/<key>.json``."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    @staticmethod
    def make_key(*parts: str | bytes) -> str:
        """Hash ``parts`` unambiguously: ("ab", "c") and ("a", "bc") give different keys."""
        digest = hashlib.blake2b(digest_size=16)
        for part in parts:
            data = part.encode("utf-8") if isinstance(part, str) else part
            digest.update(len(data).to_bytes(8, "big"))
            digest.update(data)
        return digest.hexdigest()

    def get(self, key: str) -> str | None:
        path = self._path(key)
        try:
            return str(json.loads(path.read_text(encoding="utf-8"))["value"])
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, KeyError):
            path.unlink(missing_ok=True)  # self-heal a corrupt entry
            return None

    def set(self, key: str, value: str) -> None:
        record = {"value": value, "created_at": time.time()}
        atomic_write(self._path(key), json.dumps(record, ensure_ascii=False).encode("utf-8"))

    def __contains__(self, key: str) -> bool:
        return self._path(key).is_file()

    def __len__(self) -> int:
        return sum(1 for _ in self.directory.glob("*/*.json")) if self.directory.exists() else 0

    def _path(self, key: str) -> Path:
        if len(key) < 3 or not key.isalnum():
            raise ValueError(f"Invalid cache key: {key!r}")
        return self.directory / key[:2] / f"{key}.json"
