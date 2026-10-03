"""Append-only swipe event log (data/events/YYYY-MM-DD.jsonl).

Every swipe is logged with what the recommender believed when it showed the card (rank,
candidate sources, predicted p(like)). This is the raw material for training a ranker on
real swipes later, or for off-policy evaluation of a new policy against the logged one.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import date
from pathlib import Path

DEFAULT_DIR = Path(os.environ.get("FYF_EVENTS_DIR") or Path(__file__).resolve().parents[2] / "data" / "events")


class EventLog:
    def __init__(self, directory: Path | str | None = DEFAULT_DIR):
        self.dir = Path(directory) if directory else None
        self._lock = threading.Lock()
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **fields):
        if not self.dir:
            return
        line = json.dumps({"ts": round(time.time(), 3), "event": event, **fields}, default=str)
        with self._lock, open(self.dir / f"{date.today().isoformat()}.jsonl", "a", encoding="utf-8") as f:
            f.write(line + "\n")
