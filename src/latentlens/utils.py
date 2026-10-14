"""Small, dependency-free helpers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import islice
from typing import TypeVar

T = TypeVar("T")


def batched(items: Iterable[T], size: int) -> Iterator[list[T]]:
    """Yield successive lists of at most ``size`` items (``itertools.batched`` for 3.10+)."""
    if size < 1:
        raise ValueError("size must be >= 1")
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch
