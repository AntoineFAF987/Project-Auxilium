# -*- coding: utf-8 -*-
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

Role = Literal["user", "assistant"]

class HistoryMsg(BaseModel):
    role: Role
    content: str

class ReplyTo(BaseModel):
    """Cible d'une réponse (thread léger) injectée en contexte LLM."""
    id: str
    role: Role
    content: str

class AskIn(BaseModel):
    q: str
    history: Optional[List[HistoryMsg]] = Field(default=None)
    reply_history: Optional[List[HistoryMsg]] = Field(
        default=None,
        description="Fenêtre d’historique ANCRÉE autour du message ciblé (prend le pas sur 'history')."
    )
    thread_id: Optional[str] = Field(
        default=None, description="Identifiant de la conversation (fourni par le front)"
    )
    source_mode: Optional[str] = Field(
        default="auto", description="auto | local | general | web_live"
    )
    reply_to: Optional[ReplyTo] = Field(
        default=None,
        description="Cible à laquelle l'utilisateur répond (id, role, extrait)."
    )

class PostGenerationReviewOut(BaseModel):
    status: Literal["OK", "CAVEAT"] = "OK"
    caveat_type: Optional[
        Literal[
            "STALE_SOURCE",
            "INDIRECT_EVIDENCE",
            "PARTIAL_EVIDENCE",
            "CONFLICTING_EVIDENCE",
            "INFERENCE",
            "UNSUPPORTED_CLAIM",
            "CONTRADICTED_CLAIM",
            "WEB_RECOMMENDED",
        ]
    ] = None
    message: Optional[str] = None
    severity: Literal["info", "warning"] = "info"
    suggest_web: bool = False


class AskOut(BaseModel):
    answer: str
    sources: List[dict] = Field(default_factory=list)
    mode: Optional[str] = None
    ctx_len: Optional[int] = None
    request_id: Optional[str] = None
    chat_id: Optional[str] = None
    review: Optional[PostGenerationReviewOut] = None
    faithfulness_review: Optional[dict] = None
