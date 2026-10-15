from __future__ import annotations

from pathlib import Path

import pytest

from latentlens.llm import ResponseCache


@pytest.fixture
def cache(tmp_path: Path) -> ResponseCache:
    return ResponseCache(tmp_path / "cache")


def test_roundtrip_and_membership(cache: ResponseCache) -> None:
    key = ResponseCache.make_key("gemini-2.5-flash", "describe", b"\x89PNG")
    assert cache.get(key) is None and key not in cache

    cache.set(key, "A person slices onions — quickly.")

    assert key in cache
    assert cache.get(key) == "A person slices onions — quickly."
    assert len(cache) == 1


def test_keys_are_deterministic_and_unambiguous() -> None:
    assert ResponseCache.make_key("a", b"b") == ResponseCache.make_key("a", b"b")
    assert ResponseCache.make_key("ab", "c") != ResponseCache.make_key("a", "bc")
    assert ResponseCache.make_key("model-a", "p") != ResponseCache.make_key("model-b", "p")


def test_overwrite_replaces_value(cache: ResponseCache) -> None:
    key = ResponseCache.make_key("k")
    cache.set(key, "old")
    cache.set(key, "new")
    assert cache.get(key) == "new"
    assert len(cache) == 1


def test_corrupt_entry_is_treated_as_miss_and_removed(cache: ResponseCache) -> None:
    key = ResponseCache.make_key("k")
    cache.set(key, "ok")
    path = cache.directory / key[:2] / f"{key}.json"
    path.write_text("{not json", encoding="utf-8")

    assert cache.get(key) is None
    assert not path.exists()


@pytest.mark.parametrize("bad", ["", "ab", "../../etc", "a/b/c"])
def test_rejects_unsafe_keys(cache: ResponseCache, bad: str) -> None:
    with pytest.raises(ValueError):
        cache.get(bad)


def test_empty_cache_has_zero_length(tmp_path: Path) -> None:
    assert len(ResponseCache(tmp_path / "missing")) == 0
