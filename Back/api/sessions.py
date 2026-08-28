# -*- coding: utf-8 -*-
import time
import threading
from typing import Dict, Any, Optional, Tuple
from collections import OrderedDict

# --------- Mémoire de session ---------
SESSION_TTL_SEC = 60 * 60  # 1h d'inactivité

# thread_id -> {
#   "roleplay": bool,
#   "last": timestamp,
#   "last_math": str | None,
#   "no_context_once": bool | None,
#   "last_source_mode": str | None
# }
SESSIONS: Dict[str, Dict[str, Any]] = {}
_SESS_LOCK = threading.Lock()

def _touch_session(thread_id: str):
    now = time.time()
    with _SESS_LOCK:
        sess = SESSIONS.get(thread_id) or {"roleplay": False, "last": now}
        sess["last"] = now
        SESSIONS[thread_id] = sess

def _get_session(thread_id: str) -> Dict[str, Any]:
    with _SESS_LOCK:
        sess = SESSIONS.get(thread_id)
        if not sess:
            return {}
        if time.time() - sess.get("last", 0) > SESSION_TTL_SEC:
            SESSIONS.pop(thread_id, None)
            return {}
        return dict(sess)

def _set_session(thread_id: str, **kwargs):
    with _SESS_LOCK:
        now = time.time()
        sess = SESSIONS.get(thread_id) or {"roleplay": False, "last": now}
        sess.update(kwargs)
        sess["last"] = now
        SESSIONS[thread_id] = sess

def _get_roleplay(thread_id: str) -> bool:
    s = _get_session(thread_id)
    return bool(s.get("roleplay", False))

def _set_roleplay(thread_id: str, value: bool):
    _touch_session(thread_id)
    _set_session(thread_id, roleplay=bool(value))

# --------- LRU cache simple (TTL) pour /ask ---------
class _LRUCacheTTL:
    def __init__(self, capacity: int = 256, ttl: int = 60):
        self.capacity = capacity
        self.ttl = ttl
        self._data: "OrderedDict[str, Tuple[float, Any]]" = OrderedDict()
        self._lock = threading.Lock()

    def _purge_expired(self):
        now = time.time()
        keys = []
        for k, (t, _) in self._data.items():
            if now - t > self.ttl:
                keys.append(k)
        for k in keys:
            self._data.pop(k, None)

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            self._purge_expired()
            if key not in self._data:
                return None
            t, v = self._data.pop(key)
            self._data[key] = (t, v)  # move to end (recent)
            return v

    def set(self, key: str, value: Any):
        with self._lock:
            self._purge_expired()
            if key in self._data:
                self._data.pop(key, None)
            self._data[key] = (time.time(), value)
            if len(self._data) > self.capacity:
                self._data.popitem(last=False)  # remove oldest

response_cache = _LRUCacheTTL(capacity=256, ttl=60)

# --------- Rate limiting (token bucket simple) ---------
class _RateLimiter:
    def __init__(self, max_req: int, per_sec: int):
        self.max_req = max_req
        self.per_sec = per_sec
        self._buckets: Dict[str, Tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            tokens, last = self._buckets.get(key, (self.max_req, now))
            # refill
            tokens = min(self.max_req, tokens + (now - last) * (self.max_req / self.per_sec))
            if tokens < 1.0:
                self._buckets[key] = (tokens, now)
                return False
            self._buckets[key] = (tokens - 1.0, now)
            return True
