"""Tiny JSON-file persistence layer.

Keeps the tools module free of I/O concerns: it just calls ``load()``
once at import time and ``save()`` after every write. Storage location
is configurable via ``AGENT_DATA_FILE`` (defaults to ``.agent_data.json``
in the current working directory) so tests can point it at a temp file.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

_LOCK = threading.Lock()


def _data_path() -> str:
    return os.getenv("AGENT_DATA_FILE", ".agent_data.json")


def load() -> dict[str, list[dict[str, Any]]]:
    """Load persisted state. Returns an empty store if the file is
    missing or corrupt (corrupt data should never crash the agent)."""
    path = _data_path()
    if not os.path.exists(path):
        return {"bookings": [], "reminders": []}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {"bookings": [], "reminders": []}
    data.setdefault("bookings", [])
    data.setdefault("reminders", [])
    return data


def save(data: dict[str, list[dict[str, Any]]]) -> None:
    path = _data_path()
    with _LOCK:
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
