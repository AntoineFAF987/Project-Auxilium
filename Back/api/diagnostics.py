"""Small request-scoped diagnostics helpers shared by the API transports."""

from __future__ import annotations

import json
import traceback
import uuid
from contextvars import ContextVar
from typing import Any

from fastapi.encoders import jsonable_encoder


_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_stage: ContextVar[str] = ContextVar("stream_stage", default="unknown")


def start_request(request_id: str | None = None) -> str:
    value = request_id or str(uuid.uuid4())
    _request_id.set(value)
    _stage.set("unknown")
    return value


def current_request_id() -> str | None:
    return _request_id.get()


def set_stage(stage: str) -> None:
    _stage.set(stage)


def current_stage() -> str:
    return _stage.get()


def json_safe(value: Any) -> Any:
    """Convert API/database payloads using FastAPI's standard encoder."""

    return jsonable_encoder(value)


def log_event(request_id: str, event: str, **fields: Any) -> None:
    payload = {"request_id": request_id, "event": event, **fields}
    print(json.dumps(json_safe(payload), ensure_ascii=False))


def log_stream_error(request_id: str, stage: str, exc: BaseException) -> None:
    log_event(
        request_id,
        "stream_error",
        stage=stage,
        error_type=type(exc).__name__,
        error_message=str(exc),
        traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    )
