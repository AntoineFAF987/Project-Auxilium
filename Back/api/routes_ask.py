# -*- coding: utf-8 -*-
"""Adaptateurs HTTP du pipeline de reponse partage."""

import json
from queue import Queue
from threading import Thread
from typing import Iterator, Optional, Tuple

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from .answer_pipeline import AnswerPipelineResult, run_answer_pipeline, validate_answer_request
from .schemas import AskIn, AskOut


router = APIRouter()


def _sse(payload: dict, event: Optional[str] = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(payload, ensure_ascii=False)}\n\n"


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
) -> Iterator[str]:
    """Diffuse la generation, puis la revue additive et les metadonnees."""

    events: Queue[Tuple[str, object]] = Queue()
    citation_filter = _CitationTailFilter()

    def token_sink(chunk: str) -> None:
        visible = citation_filter.feed(chunk)
        if visible:
            events.put(("content", visible))

    def produce() -> None:
        try:
            result = run_answer_pipeline(body, request, token_sink=token_sink)
            tail = citation_filter.finish()
            if tail:
                events.put(("content", tail))
            events.put(("result", result))
        except HTTPException as exc:
            events.put(("http_error", exc))
        except Exception as exc:
            events.put(("error", exc))

    Thread(target=produce, daemon=True).start()
    emitted_content = False

    while True:
        kind, payload = events.get()
        if kind == "content":
            emitted_content = True
            yield _sse({"type": "content", "content": str(payload)})
            continue
        if kind == "http_error":
            exc = payload
            assert isinstance(exc, HTTPException)
            yield _sse({"type": "error", "error": str(exc.detail), "status_code": exc.status_code})
            return
        if kind == "error":
            yield _sse({"type": "error", "error": str(payload)})
            return

        result = payload
        assert isinstance(result, AnswerPipelineResult)
        # Les abstentions, caches et outils non-LLM n'ont pas traverse token_sink.
        if result.answer and not emitted_content:
            yield _sse({"type": "content", "content": result.answer})
        if result.review.has_caveat:
            yield _sse({"type": "caveat", **result.review.to_dict()}, event="caveat")
        yield _sse(
            {
                "type": "done",
                "sources": result.sources,
                "mode": result.mode,
                "chat_id": result.chat_id,
                "request_id": result.request_id,
                "status": result.status,
                "validation_performed": result.validation_performed,
                "review": result.review.to_dict(),
                "faithfulness_review": result.faithfulness_review,
            }
        )
        return


@router.post("/ask", response_model=AskOut)
def ask(body: AskIn, request: Request) -> AskOut:
    """Transport JSON du pipeline partage, revue comprise."""

    return run_answer_pipeline(body, request).to_ask_out()


@router.post("/ask/stream")
async def ask_stream(body: AskIn, request: Request) -> StreamingResponse:
    """Transport SSE token par token apres les controles amont."""

    # Conserve le contrat HTTP historique pour les requêtes invalides.
    validate_answer_request(body)
    return StreamingResponse(
        _validated_sse(body, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
