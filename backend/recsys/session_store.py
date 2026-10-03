"""Per-session state: swipes, seen items, text steers, filters.

Only small values are stored (catalog rows, 4 scalar features per swipe, steer vectors), so
a Redis/database store only needs the same create / get / save methods plus serialisation.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

import numpy as np

VALID_DIRECTIONS = ("left", "right")
MAX_STEERS = 3


@dataclass
class Swipe:
    row: int
    liked: bool
    scalars: np.ndarray          # scalar features as they were when the card was shown
    ts: float = field(default_factory=time.time)


@dataclass
class Steer:
    text: str
    vector: np.ndarray           # FashionCLIP text embedding
    z: np.ndarray                # z-scored text->image similarity for every catalog item


@dataclass
class SessionState:
    session_id: str
    seed: int
    swipes: list = field(default_factory=list)
    seen_codes: set = field(default_factory=set)      # product codes (all colour variants) already swiped
    steers: list = field(default_factory=list)
    index_groups: list | None = None                  # optional filter, e.g. ["Menswear"]
    served: dict = field(default_factory=dict)        # row -> explanation of the last batch (for event logs)
    requests: int = 0                                 # recommendation calls (seeds Thompson sampling)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def liked_rows(self):
        return [s.row for s in self.swipes if s.liked]

    @property
    def disliked_rows(self):
        return [s.row for s in self.swipes if not s.liked]

    @property
    def seen_rows(self):
        return {s.row for s in self.swipes}

    def touch(self):
        self.updated_at = time.time()


class InMemorySessionStore:
    """Thread-safe in-process store with idle expiry. State is lost on restart."""

    def __init__(self, ttl_seconds=6 * 3600, max_sessions=10_000):
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()
        self._write_lock = threading.RLock()
        self.ttl = ttl_seconds
        self.max_sessions = max_sessions

    def create(self, index_groups=None) -> SessionState:
        state = SessionState(session_id=uuid.uuid4().hex, seed=int(np.random.SeedSequence().entropy % 2**32),
                             index_groups=index_groups)
        with self._lock:
            self._evict_locked()
            self._sessions[state.session_id] = state
        return state

    def get(self, session_id) -> SessionState | None:
        if not isinstance(session_id, str):
            return None
        with self._lock:
            state = self._sessions.get(session_id)
            if state and time.time() - state.updated_at > self.ttl:
                del self._sessions[session_id]
                return None
            return state

    def save(self, state: SessionState):
        state.touch()
        with self._lock:
            self._sessions[state.session_id] = state

    def lock(self):
        """Serialise read-modify-write on sessions (double clicks, concurrent tabs)."""
        return self._write_lock

    def __len__(self):
        return len(self._sessions)

    def _evict_locked(self):
        now = time.time()
        for sid in [sid for sid, s in self._sessions.items() if now - s.updated_at > self.ttl]:
            del self._sessions[sid]
        while len(self._sessions) >= self.max_sessions:
            oldest = min(self._sessions.values(), key=lambda s: s.updated_at)
            del self._sessions[oldest.session_id]
