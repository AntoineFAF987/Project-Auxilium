"""Thread-safe, in-memory response traces for local diagnostics only."""
from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
from threading import Lock
from typing import Any


class ResponseTraceStore:
    def __init__(self, limit: int = 20) -> None:
        self.limit = limit
        self._items: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = Lock()

    def start(self, request_id: str, data: dict[str, Any]) -> None:
        with self._lock:
            self._items[request_id] = {"request_id": request_id, "timestamp": datetime.now(timezone.utc).isoformat(), **data, "stages": {}}
            self._items.move_to_end(request_id)
            while len(self._items) > self.limit:
                self._items.popitem(last=False)

    def stage(self, request_id: str, name: str, data: dict[str, Any]) -> None:
        with self._lock:
            trace = self._items.get(request_id)
            if trace is not None:
                trace["stages"][name] = data

    def finish(self, request_id: str, data: dict[str, Any]) -> None:
        self.stage(request_id, "answer", data)

    def get(self, request_id: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._items.get(request_id)
            return dict(item) if item else None

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [{"request_id": key, "timestamp": value["timestamp"], "original_user_message": value.get("original_user_message"), "mode": value.get("mode")} for key, value in reversed(self._items.items())]
