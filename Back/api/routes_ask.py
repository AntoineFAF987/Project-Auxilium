# -*- coding: utf-8 -*-
"""Adaptateurs HTTP du pipeline de réponse partagé.

La route SSE bufferise la réponse complète : aucun contenu n'est envoyé au
client avant la fin des contrôles de faithfulness et de post-validation.
"""

import json
from typing import Iterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from .answer_pipeline import run_answer_pipeline, validate_answer_request
from .schemas import AskIn, AskOut


router = APIRouter()


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _validated_sse(
    body: AskIn,
    request: Request,
) -> Iterator[str]:
    """Exécute et valide entièrement le pipeline avant le premier contenu SSE."""

    try:
        result = run_answer_pipeline(body, request)
    except HTTPException as exc:
        yield _sse(
            {
                "type": "error",
                "error": str(exc.detail),
                "status_code": exc.status_code,
            }
        )
        return
    except Exception as exc:
        yield _sse({"type": "error", "error": str(exc)})
        return

    # Ce premier événement n'est émis qu'après toutes les validations du pipeline.
    if result.answer:
        yield _sse({"type": "content", "content": result.answer})
    yield _sse(
        {
            "type": "done",
            "sources": result.sources,
            "mode": result.mode,
            "chat_id": result.chat_id,
            "request_id": result.request_id,
            "status": result.status,
            "validation_performed": result.validation_performed,
        }
    )


@router.post("/ask", response_model=AskOut)
def ask(body: AskIn, request: Request) -> AskOut:
    """Transport JSON du pipeline partagé."""

    return run_answer_pipeline(body, request).to_ask_out()


@router.post("/ask/stream")
async def ask_stream(body: AskIn, request: Request) -> StreamingResponse:
    """Transport SSE bufferisé du même pipeline partagé."""

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
