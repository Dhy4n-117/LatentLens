"""Small, dependency-free helpers."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


def batched(items: Iterable[T], size: int) -> Iterator[list[T]]:
    """Yield successive lists of at most ``size`` items (``itertools.batched`` for 3.10+)."""
    if size < 1:
        raise ValueError("size must be >= 1")
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch


def atomic_write(path: Path, data: bytes) -> None:
    """Write-then-rename so a crash never leaves a half-written file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
