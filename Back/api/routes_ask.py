# -*- coding: utf-8 -*-
"""Adaptateurs HTTP du pipeline de reponse partage."""

import json
import time
import traceback
from queue import Queue
from threading import Thread
from typing import Iterator, Optional, Tuple

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from .answer_pipeline import AnswerPipelineResult, run_answer_pipeline, validate_answer_request
from .schemas import AskIn, AskOut
from .diagnostics import current_stage, json_safe, log_event, log_stream_error, start_request, set_stage


router = APIRouter()


def _sse(payload: dict, event: Optional[str] = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(json_safe(payload), ensure_ascii=False)}\n\n"


class _CitationTailFilter:
    """Masque le bloc technique CITATIONS sans bufferiser le texte de reponse."""

    marker = "<citations>"

    def __init__(self) -> None:
        self.pending = ""
        self.hidden = False

    def feed(self, chunk: str) -> str:
        if self.hidden or not chunk:
            return ""
        self.pending += chunk
        marker_at = self.pending.lower().find(self.marker)
        if marker_at >= 0:
            visible = self.pending[:marker_at]
            self.pending = ""
            self.hidden = True
            return visible
        keep = len(self.marker) - 1
        if len(self.pending) <= keep:
            return ""
        visible, self.pending = self.pending[:-keep], self.pending[-keep:]
        return visible

    def finish(self) -> str:
        if self.hidden:
            return ""
        visible, self.pending = self.pending, ""
        return visible


def _validated_sse(
    body: AskIn,
    request: Request,
    request_id: Optional[str] = None,
) -> Iterator[str]:
    """Diffuse la generation, puis la revue additive et les metadonnees."""

    events: Queue[Tuple[str, object]] = Queue()
    citation_filter = _CitationTailFilter()
    request_id = request_id or start_request()
    status_order = 0

    def token_sink(chunk: str) -> None:
        visible = citation_filter.feed(chunk)
        if visible:
            events.put(("content", visible))

    def status_sink(stage: str, label: str) -> None:
        # Public functional progress only; never model reasoning or prompt text.
        nonlocal status_order
        status_order += 1
        log_event(
            request_id,
            "sse_status_emitted",
            stage=stage,
            status_order=status_order,
            timestamp_ms=round(time.time() * 1000),
        )
        events.put(("status", {"type": "status", "stage": stage, "label": label}))

    def produce() -> None:
        start_request(request_id)
        try:
            result = run_answer_pipeline(body, request, token_sink=token_sink, status_sink=status_sink)
            tail = citation_filter.finish()
            if tail:
                events.put(("content", tail))
            events.put(("result", result))
        except HTTPException as exc:
            log_stream_error(request_id, current_stage(), exc)
            events.put(("error", {"request_id": request_id}))
        except Exception as exc:
            log_stream_error(request_id, current_stage(), exc)
            events.put(("error", {"request_id": request_id}))

    Thread(target=produce, daemon=True).start()
    emitted_content = False

    while True:
        kind, payload = events.get()
        if kind == "content":
            emitted_content = True
            yield _sse({"type": "content", "content": str(payload)})
            continue
        if kind == "status":
            status_payload = payload if isinstance(payload, dict) else {}
            yield _sse(status_payload, event="status")
            continue
        if kind == "error":
            error_payload = payload if isinstance(payload, dict) else {"request_id": request_id}
            yield _sse({
                "type": "error",
                "message": "Une erreur interne est survenue pendant la génération.",
                "request_id": error_payload.get("request_id", request_id),
            }, event="error")
            return

        result = payload
        assert isinstance(result, AnswerPipelineResult)
        # Les abstentions, caches et outils non-LLM n'ont pas traverse token_sink.
        if result.answer and not emitted_content:
            yield _sse({"type": "content", "content": result.answer})
        if result.review.has_caveat:
            yield _sse({"type": "caveat", **result.review.to_dict()}, event="caveat")
        try:
            set_stage("source_serialization")
            safe_sources = json_safe(result.sources)
            done_payload = {
                "type": "done",
                "sources": safe_sources,
                "mode": result.mode,
                "chat_id": result.chat_id,
                "request_id": result.request_id or request_id,
                "status": result.status,
                "validation_performed": result.validation_performed,
                "review": result.review.to_dict(),
                "faithfulness_review": result.faithfulness_review,
            }
            set_stage("sse_emit")
            yield _sse(done_payload)
            log_event(request_id, "stream_done")
        except Exception as exc:
            log_stream_error(request_id, current_stage(), exc)
            yield _sse({
                "type": "error",
                "message": "Une erreur interne est survenue pendant la génération.",
                "request_id": request_id,
            }, event="error")
        return


@router.post("/ask", response_model=AskOut)
def ask(body: AskIn, request: Request) -> AskOut:
    """Transport JSON du pipeline partage, revue comprise."""

    return run_answer_pipeline(body, request).to_ask_out()


@router.post("/ask/stream")
async def ask_stream(body: AskIn, request: Request) -> StreamingResponse:
    """Transport SSE token par token apres les controles amont."""

    # Conserve le contrat HTTP historique pour les requêtes invalides.
    request_id = start_request()
    log_event(request_id, "request_start", path="/ask/stream")
    try:
        validate_answer_request(body)
    except Exception as exc:
        log_event(
            request_id,
            "http_failure_before_stream",
            stage="unknown",
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )
        raise
    return StreamingResponse(
        _validated_sse(body, request, request_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Request-ID": request_id,
        },
    )
