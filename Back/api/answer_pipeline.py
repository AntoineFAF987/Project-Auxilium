# -*- coding: utf-8 -*-
import re, json, uuid, hashlib, logging
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List, Dict, Literal, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from fastapi import HTTPException, Request

from .schemas import AskIn, AskOut
from .index_singleton import (
    idx,
    format_context_for_llm, fuse_contiguous_passages, clip_context_blocks,
    ask_mistral_with_context, answerability_guard, keyword_overlap_count,
    trim_history, classify_smalltalk_semantic, web_search_context,
    RETRIEVE_K, TOP_K_FAISS, HYBRID_ALPHA, FUSE_ADJACENT_GAP, MAX_CONTEXT_CHARS, FINAL_K
)
from .sessions import _touch_session, _get_roleplay, _set_roleplay, SESSIONS, response_cache
from .roleplay import detect_roleplay_trigger, looks_factual
from .math_tools import _is_explain_followup, _looks_like_equation, _solve_math
from rag_core.faithfulness import (
    CitationStatus,
    ClaimStatus,
    FaithfulnessReview,
    should_fallback_to_general,
    verify_answer_claims,
)
from rag_core.llm_stream import ask_mistral_with_context_stream
from rag_core.turn_type import classify_turn
from rag_core.context_sufficiency import AnswerabilityDecision, evaluate_answerability
from runtime_settings import get_runtime_settings
from .orchestration import (
    OrchestrationPlan, OrchestrationPlanOutputError, build_prompt, compact_history,
    explain_candidate_rejection, filter_retrieval_candidates, plan_once, plan_json_retry,
    sanitize_plan_for_retrieval,
)
from .orchestration_debug import record_snapshot
from .response_trace import ResponseTraceStore
from .iterative_retrieval import run_iterative_evidence_retrieval
from .multi_query_retrieval import build_retrieval_queries, reciprocal_rank_fusion, resolve_retrieval_query
from .intelligent_retry import build_retry_queries, derive_retrieval_gap, merge_cumulative_evidence
from .catalog_probe import probe_document_catalog
from .chats_db import create_chat, append_message, mark_chat_title_generation_attempted, set_generated_chat_title, should_generate_chat_title
from .chat_titles import generate_chat_title
from .source_references import select_cited_references
from .answer_presentation import (
    enforce_final_citation_contract, extract_citations_and_clean_answer,
    remap_inline_citations, select_web_source_indices,
)
from .diagnostics import current_request_id, log_event, log_stream_error, set_stage
from auth_ms import verify_ms_token

# --- Anti-429: concurrence + retries ---
import threading, random, time

logger = logging.getLogger(__name__)


AnswerStatus = Literal["answered", "abstained", "fallback"]
EvidenceMode = Literal["direct", "related", "none"]
LocalContextState = Literal["relevant_and_sufficient", "relevant_but_incomplete", "irrelevant"]
ReviewStatus = Literal["OK", "CAVEAT"]
CaveatType = Literal[
    "STALE_SOURCE",
    "INDIRECT_EVIDENCE",
    "PARTIAL_EVIDENCE",
    "CONFLICTING_EVIDENCE",
    "INFERENCE",
    "UNSUPPORTED_CLAIM",
    "CONTRADICTED_CLAIM",
    "CITATION_MISMATCH",
    "WEB_RECOMMENDED",
]

_CAVEAT_TYPES = {
    "STALE_SOURCE",
    "INDIRECT_EVIDENCE",
    "PARTIAL_EVIDENCE",
    "CONFLICTING_EVIDENCE",
    "INFERENCE",
    "UNSUPPORTED_CLAIM",
    "CONTRADICTED_CLAIM",
    "CITATION_MISMATCH",
    "WEB_RECOMMENDED",
}


@dataclass(frozen=True)
class PostGenerationReview:
    """Revue informative ajoutée après génération, jamais destructive."""

    status: ReviewStatus = "OK"
    caveat_type: Optional[CaveatType] = None
    message: Optional[str] = None
    severity: Literal["info", "warning"] = "info"
    suggest_web: bool = False

    @property
    def has_caveat(self) -> bool:
        return self.status == "CAVEAT" and bool(self.caveat_type and self.message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "caveat_type": self.caveat_type,
            "message": self.message,
            "severity": self.severity,
            "suggest_web": self.suggest_web,
        }


@dataclass(frozen=True)
class AnswerPipelineResult:
    """Résultat interne commun aux transports JSON et SSE."""

    answer: str
    artifacts: List[Dict[str, str]] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    mode: Optional[str] = None
    status: AnswerStatus = "answered"
    context_length: Optional[int] = None
    request_id: Optional[str] = None
    chat_id: Optional[str] = None
    chat_title: Optional[str] = None
    assistant_message_id: Optional[int] = None
    route_mode: Optional[str] = None
    abstention_reason: Optional[str] = None
    fallback_reason: Optional[str] = None
    validations: Dict[str, Any] = field(default_factory=dict)
    review: PostGenerationReview = field(default_factory=PostGenerationReview)
    faithfulness_review: Optional[Dict[str, Any]] = None

    @property
    def validation_performed(self) -> bool:
        return bool(self.validations)

    def to_ask_out(self) -> AskOut:
        return AskOut(
            answer=self.answer,
            artifacts=self.artifacts,
            sources=self.sources,
            mode=self.mode,
            ctx_len=self.context_length,
            request_id=self.request_id,
            chat_id=self.chat_id,
            chat_title=self.chat_title,
            review=self.review.to_dict(),
            faithfulness_review=self.faithfulness_review,
        )


@dataclass(frozen=True)
class EvidenceDecision:
    mode: EvidenceMode
    reason: str
    context_is_relevant: bool
    context_relevance_reason: str
    morphological_topic_overlap: int = 0


def _infer_result_status(
    answer: str,
    mode: Optional[str],
) -> Tuple[AnswerStatus, Optional[str], Optional[str]]:
    normalized = (answer or "").lower()
    if (mode or "").startswith("FALLBACK"):
        reason = (mode or "").removeprefix("FALLBACK(").removesuffix(")")
        return "fallback", None, reason or None
    abstention_markers = (
        "index local n'existe pas",
        "rien trouvé",
        "rien trouvé de pertinent",
        "rien de suffisamment pertinent",
        "je n’ai rien trouvé",
        "je n'ai rien trouvé",
        "je n’ai parcouru",
        "je n'ai parcouru",
        "je préfère vérifier",
    )
    if any(marker in normalized for marker in abstention_markers):
        return "abstained", answer, None
    return "answered", None, None


def _result(
    *,
    answer: str,
    artifacts: Optional[List[Dict[str, str]]] = None,
    sources: Optional[List[Dict[str, Any]]] = None,
    mode: Optional[str] = None,
    ctx_len: Optional[int] = None,
    request_id: Optional[str] = None,
    chat_id: Optional[str] = None,
    assistant_message_id: Optional[int] = None,
    route_mode: Optional[str] = None,
    status: Optional[AnswerStatus] = None,
    abstention_reason: Optional[str] = None,
    fallback_reason: Optional[str] = None,
    validations: Optional[Dict[str, Any]] = None,
    review: Optional[PostGenerationReview | Dict[str, Any]] = None,
    faithfulness_review: Optional[Dict[str, Any]] = None,
) -> AnswerPipelineResult:
    inferred_status, inferred_abstention, inferred_fallback = _infer_result_status(answer, mode)
    if isinstance(review, dict):
        review = PostGenerationReview(**review)
    return AnswerPipelineResult(
        answer=answer,
        artifacts=list(artifacts or []),
        sources=list(sources or []),
        mode=mode,
        status=status or inferred_status,
        context_length=ctx_len,
        request_id=request_id,
        chat_id=chat_id,
        assistant_message_id=assistant_message_id,
        route_mode=route_mode,
        abstention_reason=abstention_reason or inferred_abstention,
        fallback_reason=fallback_reason or inferred_fallback,
        validations=dict(validations or {}),
        review=review or PostGenerationReview(),
        faithfulness_review=faithfulness_review,
    )


def _previous_validated_evidence(history: List[Dict[str, Any]]) -> tuple[Dict[str, Any] | None, int | None]:
    """Return the most recent assistant turn with durable RAG provenance."""
    for offset, item in enumerate(reversed(history or []), start=1):
        if item.get("role") != "assistant":
            continue
        meta = item.get("meta") or {}
        provenance = meta.get("evidence_provenance") or {}
        if provenance.get("evidence_sufficient") and provenance.get("evidence_chunk_uids"):
            return meta, len(history) - offset
    return None, None


class _GeneratedArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["email_draft"]
    subject: str = Field(min_length=1, max_length=240)
    content: str = Field(min_length=1, max_length=12000)

    @field_validator("subject")
    @classmethod
    def normalize_subject(cls, value: str) -> str:
        value = re.sub(r"^\s*(?:objet|subject)\s*:\s*", "", value, flags=re.IGNORECASE).strip().strip('"').rstrip(".").strip()
        if not value:
            raise ValueError("subject cannot be empty")
        return value


class _EmailDraftGeneration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str = Field(min_length=1, max_length=12000)
    artifacts: List[_GeneratedArtifact] = Field(min_length=1, max_length=4)
    citations: List[int] = Field(default_factory=list, max_length=100)


def _parse_email_draft_generation(raw: str) -> tuple[str, List[Dict[str, str]], List[int]]:
    """Validate the generator envelope; no prose-marker parsing is involved."""
    payload = json.loads((raw or "").strip())
    output = _EmailDraftGeneration.model_validate(payload)
    citations = [index for index in output.citations if index > 0]
    return output.answer.strip(), [artifact.model_dump() for artifact in output.artifacts], citations


def _sender_first_name_from_request(request: Request) -> str | None:
    """Use authenticated profile claims only; never manufacture a sender name."""
    auth = request.headers.get("authorization") or request.headers.get("Authorization") or ""
    if " " not in auth:
        return None
    try:
        token = auth.split(" ", 1)[1]
        try: claims = verify_ms_token(token, expect_id_token=True)
        except Exception: claims = verify_ms_token(token, expect_id_token=False)
        value = str(claims.get("given_name") or claims.get("first_name") or claims.get("name") or "").strip()
        return value.split()[0] if value else None
    except Exception:
        return None


def _hydrate_email_sender(artifacts: List[Dict[str, Any]], request: Request) -> List[Dict[str, Any]]:
    sender = _sender_first_name_from_request(request)
    return [{**artifact, "sender_first_name": artifact.get("sender_first_name") or sender} if artifact.get("type") == "email_draft" else artifact for artifact in artifacts]


def _partial_json_string(raw: str, key: str, *, after: int = 0) -> Optional[str]:
    """Read an incomplete JSON string value without accepting prose markers."""
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"', raw[after:])
    if not match:
        return None
    index = after + match.end()
    chars: List[str] = []
    escapes = {"\\\"": '"', "\\\\": "\\", "\\/": "/", "\\b": "\b", "\\f": "\f", "\\n": "\n", "\\r": "\r", "\\t": "\t"}
    while index < len(raw):
        char = raw[index]
        if char == '"':
            return "".join(chars)
        if char != "\\":
            chars.append(char)
            index += 1
            continue
        if index + 1 >= len(raw):
            break
        token = raw[index:index + 2]
        if token == "\\u":
            if index + 6 > len(raw):
                break
            try:
                chars.append(chr(int(raw[index + 2:index + 6], 16)))
            except ValueError:
                break
            index += 6
            continue
        decoded = escapes.get(token)
        if decoded is None:
            break
        chars.append(decoded)
        index += 2
    return "".join(chars)


class _StreamingEmailDraftDecoder:
    """Converts the structured generator envelope into safe SSE deltas."""

    def __init__(self, text_sink: Callable[[str], None], artifact_sink: Callable[[str, Dict[str, Any]], None]) -> None:
        self._raw = ""
        self._text_sink = text_sink
        self._artifact_sink = artifact_sink
        self._answer_length = 0
        self._content_length = 0
        self._subject = ""
        self._started = False

    def feed(self, chunk: str) -> None:
        self._raw += chunk
        answer = _partial_json_string(self._raw, "answer")
        if answer is not None and len(answer) > self._answer_length:
            self._text_sink(answer[self._answer_length:])
            self._answer_length = len(answer)

        artifacts_at = self._raw.find('"artifacts"')
        if artifacts_at < 0:
            return
        artifact_type = _partial_json_string(self._raw, "type", after=artifacts_at)
        if artifact_type != "email_draft":
            return
        if not self._started:
            self._started = True
            self._artifact_sink("artifact_start", {"index": 0, "type": "email_draft"})
        subject = _partial_json_string(self._raw, "subject", after=artifacts_at)
        if subject is not None and subject != self._subject:
            self._subject = subject
            self._artifact_sink("artifact_subject", {"index": 0, "subject": subject})
        content = _partial_json_string(self._raw, "content", after=artifacts_at)
        if content is not None and len(content) > self._content_length:
            self._artifact_sink("artifact_delta", {"index": 0, "content": content[self._content_length:]})
            self._content_length = len(content)

    def finish(self) -> None:
        if self._started:
            self._artifact_sink("artifact_end", {"index": 0})


def validate_answer_request(body: AskIn) -> str:
    """Validation sans effet de bord, réutilisable avant d'ouvrir un flux SSE."""

    q = (body.q or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Champ 'q' vide")
    return q


EXEC = ThreadPoolExecutor(max_workers=8)

_RUNTIME_SETTINGS = get_runtime_settings()
LLM_TIMEOUT_SEC = _RUNTIME_SETTINGS.timeouts.llm_sec
WEB_TIMEOUT_SEC = _RUNTIME_SETTINGS.timeouts.web_sec
MATH_TIMEOUT_SEC = _RUNTIME_SETTINGS.timeouts.math_sec
ANS_THRESHOLD = _RUNTIME_SETTINGS.thresholds.answerability
OVERLAP_MIN = _RUNTIME_SETTINGS.thresholds.overlap_min
CONTEXT_RELEVANCE_THRESHOLD = _RUNTIME_SETTINGS.thresholds.context_relevance
ENABLE_RERANKER = _RUNTIME_SETTINGS.features.enable_reranker
ENABLE_CONDENSATION = _RUNTIME_SETTINGS.features.enable_query_condensation
ENABLE_EXPANSION = _RUNTIME_SETTINGS.features.enable_query_expansion
ENABLE_INTELLIGENT_RETRY = _RUNTIME_SETTINGS.features.enable_intelligent_retry
MAX_RETRY_ROUNDS = _RUNTIME_SETTINGS.retrieval.max_retry_rounds
MAX_RETRY_QUERIES = _RUNTIME_SETTINGS.retrieval.max_retry_queries
ENABLE_POST_GENERATION_REVIEW = _RUNTIME_SETTINGS.features.enable_post_generation_review
ENABLE_FAITHFULNESS_CHECK = _RUNTIME_SETTINGS.features.enable_faithfulness_check
FAITHFULNESS_THRESHOLD = _RUNTIME_SETTINGS.thresholds.faithfulness
FAITHFULNESS_STRICT_ONLY = _RUNTIME_SETTINGS.features.faithfulness_strict_only
HISTORY_MAX_TURNS = _RUNTIME_SETTINGS.conversation.history_max_turns
REPLY_HISTORY_MAX_TURNS = _RUNTIME_SETTINGS.conversation.reply_history_max_turns
WEB_MAX_CHARS = _RUNTIME_SETTINGS.conversation.web_max_chars
WEB_RESULT_K = _RUNTIME_SETTINGS.conversation.web_result_k
ORCHESTRATOR_SETTINGS = _RUNTIME_SETTINGS.orchestrator
RESPONSE_TRACE_ENABLED = _RUNTIME_SETTINGS.debug.response_trace
RESPONSE_TRACES = ResponseTraceStore(_RUNTIME_SETTINGS.debug.response_trace_limit)

LLM_SEM = threading.Semaphore(4)  # limite d'appels LLM en parallèle

def _with_timeout(fn, *args, timeout=10, **kwargs):
    fut = EXEC.submit(fn, *args, **kwargs)
    return fut.result(timeout=timeout)

def _call_llm_with_retries(fn, *args, timeout=18, retries=2, **kwargs):
    for attempt in range(retries + 1):
        try:
            return _with_timeout(fn, *args, timeout=timeout, **kwargs)
        except Exception as e:
            msg = str(e)
            ratelimited = ("429" in msg) or ("rate" in msg.lower() and "limit" in msg.lower())
            if ratelimited and attempt < retries:
                time.sleep(0.25 + random.random() * (0.35 * (attempt + 1)))
                continue
            raise

def _safe_llm(fn, *args, **kwargs):
    if not LLM_SEM.acquire(timeout=5):
        raise HTTPException(status_code=429, detail="Serveur occupé, réessaie dans 1–2 secondes.")
    try:
        return _call_llm_with_retries(fn, *args, **kwargs)
    finally:
        LLM_SEM.release()


def _run_orchestration(q: str, history: List[Dict]) -> OrchestrationPlan:
    """Plan with one JSON-only repair attempt; never performs retrieval."""
    if not LLM_SEM.acquire(timeout=5):
        raise RuntimeError("orchestrator_busy")
    try:
        recent_history = history[-max(ORCHESTRATOR_SETTINGS.history_max_messages, 6):]
        call_llm = lambda prompt, **kwargs: _with_timeout(ask_mistral_with_context, prompt, **kwargs)
        model = ORCHESTRATOR_SETTINGS.model or _RUNTIME_SETTINGS.generation.model
        try:
            return plan_once(q, recent_history, call_llm, model=model, timeout=ORCHESTRATOR_SETTINGS.timeout)
        except OrchestrationPlanOutputError:
            logger.warning("orchestrator_invalid_json_retrying_once")
            try:
                return plan_json_retry(
                    q, recent_history, call_llm, model=model,
                    timeout=min(ORCHESTRATOR_SETTINGS.timeout, 5),
                )
            except OrchestrationPlanOutputError:
                logger.warning("orchestrator_json_retry_failed")
                raise
    finally:
        LLM_SEM.release()


def _sanitize_provider_error_message(value: Any) -> str | None:
    if not value:
        return None
    text = str(value)
    text = re.sub(r"(?i)(bearer\s+|sk-[a-z0-9_-]+)[a-z0-9_.-]+", r"\1[redacted]", text)
    return text[:800]


def _orchestration_failure_details(exc: Exception) -> dict[str, Any]:
    """Classify provider failures separately from invalid model JSON."""
    provider = _RUNTIME_SETTINGS.generation.provider
    model = ORCHESTRATOR_SETTINGS.model or _RUNTIME_SETTINGS.generation.model
    endpoint = "responses.create" if provider == "openai" else "POST /v1/chat/completions"
    details: dict[str, Any] = {
        "provider": provider, "model": model,
        "request_schema_version": "orchestration_plan_v2",
        "endpoint": endpoint,
        "reasoning_effort": _RUNTIME_SETTINGS.generation.reasoning_effort,
        "timeout": ORCHESTRATOR_SETTINGS.timeout,
    }
    if isinstance(exc, OrchestrationPlanOutputError):
        return {
            **details, "error_category": "model_output_validation_error",
            "provider_error_type": None, "provider_error_code": None,
            "provider_error_message": None,
            "validation_error_details": exc.validation_error_details,
        }
    if isinstance(exc, FuturesTimeout):
        return {
            **details, "error_category": "timeout", "provider_error_type": type(exc).__name__,
            "provider_error_code": None, "provider_error_message": "orchestration call timed out",
            "validation_error_details": None,
        }
    response = getattr(exc, "response", None)
    body: Any = None
    if response is not None:
        try:
            body = response.json()
        except Exception:
            body = getattr(response, "text", None)
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None) or getattr(response, "status_code", None)
    message = getattr(exc, "message", None) or str(exc)
    if isinstance(body, dict):
        code = body.get("code") or (body.get("error") or {}).get("code") or code
        message = body.get("message") or (body.get("error") or {}).get("message") or message
        error_body = body.get("error") if isinstance(body.get("error"), dict) else body
        safe_response = {
            key: _sanitize_provider_error_message(error_body.get(key))
            for key in ("type", "code", "message", "param") if error_body.get(key) is not None
        }
    else:
        safe_response = None
    return {
        **details, "error_category": "provider_error", "provider_error_type": type(exc).__name__,
        "provider_error_code": str(code) if code is not None else None,
        "provider_error_message": _sanitize_provider_error_message(message),
        "validation_error_details": {"provider_response": safe_response} if safe_response else None,
    }


def _looks_like_documentary_fallback(question: str, history: List[Dict]) -> bool:
    """Conservative fallback: prefer local retrieval for likely user-owned facts."""
    text = " ".join([question] + [str(item.get("content", "")) for item in history[-4:] if item.get("role") == "user"]).lower()
    markers = (
        "status", "decision", "approval", "approved", "rejected", "request", "case", "file", "response",
        "authorization", "validation", "dossier", "demande", "decision", "autorisation", "validation",
        "reponse", "accepte", "refuse", "toujours", "enfin", "verifie", "cherche", "mails", "mail", "email", "courriel", "documents", "document",
        "contenu", "contient", "etudie", "étudie", "lire",
    )
    return any(re.search(rf"\b{re.escape(marker)}\b", text) for marker in markers)


def _recover_query_from_history(question: str, history: List[Dict]) -> str:
    """Keep the active user subject when orchestration is unavailable.

    This is intentionally deterministic: it preserves the previous user turn
    verbatim instead of guessing entities or inventing a rewritten request.
    """
    for item in reversed(history or []):
        if str(item.get("role", "")).lower() != "user":
            continue
        subject = " ".join(str(item.get("content", "")).split())[:350]
        if subject and subject.casefold() != question.casefold():
            return f"{question}\nContexte conversationnel actif : {subject}"[:700]
    return question


def _has_explicit_web_request(question: str) -> bool:
    """Recognise only an unambiguous, user-requested Web source override."""
    text = " ".join((question or "").casefold().split())
    return bool(re.search(
        r"\b(?:search|look|check|find|use|cherche|recherche|regarde|consulte|utilise|utiliser)\b.{0,40}"
        r"\b(?:web|internet|online|en ligne)\b|\b(?:web|internet|online)\s+(?:search|recherche)\b",
        text,
    ))


def _explicit_web_query(question: str, history: List[Dict]) -> str:
    """Resolve an explicit Web follow-up without requiring orchestration."""
    if not history:
        return question
    condensed = _condense_question(history, question)
    return condensed if _norm(condensed) != _norm(question) else _recover_query_from_history(question, history)


def _safe_llm_stream(fn, *args, **kwargs):
    """Itere un flux LLM sous le meme garde-fou de concurrence que les appels sync."""

    kwargs.pop("timeout", None)
    if not LLM_SEM.acquire(timeout=5):
        raise HTTPException(status_code=429, detail="Serveur occupe, reessaie dans 1-2 secondes.")
    try:
        yield from fn(*args, **kwargs)
    finally:
        LLM_SEM.release()


def _generate_answer(
    question: str,
    context_text: str,
    *,
    history: List[Dict],
    token_sink: Optional[Callable[[str], None]] = None,
    artifact_sink: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    diagnostic_stage_sink: Optional[Callable[[str], None]] = None,
    progress_sink: Optional[Callable[[str, str], None]] = None,
    **kwargs,
) -> str:
    """Genere en sync ou transmet chaque fragment au transport SSE."""

    response_format = str(kwargs.get("response_format") or "normal")
    set_stage("generation")
    if diagnostic_stage_sink:
        diagnostic_stage_sink("generation")
    request_id = current_request_id()
    if request_id:
        log_event(request_id, "generation_start")
    if progress_sink:
        progress_sink("prepare_response", "Préparation de la réponse")
        progress_sink("draft_response", "Rédaction de la réponse")

    try:
        if token_sink is None:
            answer = _safe_llm(
                ask_mistral_with_context,
                question,
                context_text=context_text,
                history=history,
                timeout=LLM_TIMEOUT_SEC,
                **kwargs,
            )
        else:
            parts: List[str] = []
            decoder = _StreamingEmailDraftDecoder(token_sink, artifact_sink) if response_format == "email_draft" and artifact_sink else None
            for chunk in _safe_llm_stream(
                ask_mistral_with_context_stream,
                question,
                context_text=context_text,
                history=history,
                **kwargs,
            ):
                text = str(chunk or "")
                if text:
                    parts.append(text)
                    if decoder:
                        decoder.feed(text)
                    else:
                        token_sink(text)
            answer = "".join(parts)
            if decoder:
                decoder.finish()
    except Exception:
        raise

    if request_id:
        log_event(request_id, "generation_end")
    return answer

def _local_index_ready() -> bool:
    try:
        return all(Path(p).exists() for p in [idx.corpus_path, idx.emb_path, idx.faiss_path])
    except Exception:
        return False

def _missing_index_message(mode_in: str) -> str:
    if mode_in in {"local", "web_index"}:
        return "L'index local n'existe pas encore. Lance une reindexation avant d'utiliser ce mode."
    return ""

# ---------- Auth helpers (optionnels) ----------
def _try_get_auth_ids(request: Request):
    """Essaye d'extraire (tenant_id, user_id) depuis Authorization.
    Si absent/invalide → retourne (None, None)."""
    auth = request.headers.get("authorization") or request.headers.get("Authorization") or ""
    if not auth or " " not in auth:
        return None, None
    scheme, token = auth.split(" ", 1)
    if scheme.lower() != "bearer" or not token:
        return None, None
    try:
        # Essayer d'abord ID token, puis Access token
        try:
            p = verify_ms_token(token, expect_id_token=True)
        except Exception:
            p = verify_ms_token(token, expect_id_token=False)
        tenant_id = p.get("tid") or p.get("tenant_id")
        user_id = p.get("oid") or p.get("sub")
        return tenant_id, user_id
    except Exception:
        return None, None

# ==================== Query Rewriting Helpers (Tier 0 improvements) ====================
def _check_context_relevance(question: str, context: str, threshold: float = None) -> bool:
    """
    Vérifie si le contexte trouvé est réellement pertinent pour la question.
    Utilise le keyword overlap comme proxy rapide de pertinence thématique.
    Retourne False si le contexte semble complètement hors sujet.
    """
    if threshold is None:
        threshold = CONTEXT_RELEVANCE_THRESHOLD
    if not context or not context.strip():
        return False
    # Si peu de mots en commun ET contexte long, probablement hors sujet
    overlap = keyword_overlap_count(question, context)
    q_words = len([w for w in question.lower().split() if len(w) > 3])
    if q_words == 0:
        return True  # question trop courte pour juger
    # A lexical ratio alone makes a lone shared location/person/year look
    # relevant. Keep it only as a first gate, then require alignment with the
    # information need itself.
    ratio = overlap / max(1, q_words)
    return ratio >= threshold and _has_information_need_alignment(question, context)


def _specific_anchors(text: str) -> set[str]:
    """Extract stable, domain-neutral identifiers (serial/model/reference-like tokens)."""
    return {
        token.lower()
        for token in re.findall(r"\b(?:[A-Za-z]+[A-Za-z0-9-]*\d[A-Za-z0-9-]*|\d+(?:\.\d+)+|\d{2,})\b", text or "")
    }


def _classify_evidence_mode(question: str, context: str, *, relevant: bool) -> EvidenceMode:
    """Separate direct support from useful, but entity-mismatched, context."""
    if not context or not relevant:
        return "none"
    requested = _specific_anchors(question)
    documented = _specific_anchors(context)
    if not requested:
        return "direct"
    if requested.issubset(documented):
        return "direct"
    if documented - requested:
        return "related"
    return "none"


def _evidence_text(blocks: List[Tuple[float, Dict]]) -> str:
    """Use chunk bodies only: formatted context also contains citation/chunk numbers."""
    return "\n".join(str(meta.get("text") or "") for _, meta in (blocks or []))


def _morphological_topic_overlap(question: str, context: str) -> int:
    """Count close lexical variants without tying evidence to an exact word form.

    This is deliberately only a supporting signal: the reranker guard and a
    distinct documented anchor are still required before such a block can be
    considered related evidence.  It makes ordinary inflectional or
    derivational variants (for example a verb and its noun) comparable without
    embedding a domain vocabulary in the policy.
    """
    def tokens(text: str) -> set[str]:
        return {
            token.casefold()
            for token in re.findall(r"\b[^\W\d_]{4,}\b", text or "", flags=re.UNICODE)
        }

    question_tokens = tokens(question)
    context_tokens = tokens(context)
    related = 0
    for q_token in question_tokens:
        for c_token in context_tokens:
            if q_token == c_token:
                continue
            common_prefix = 0
            for q_char, c_char in zip(q_token, c_token):
                if q_char != c_char:
                    break
                common_prefix += 1
            shorter = min(len(q_token), len(c_token))
            # A long shared prefix relative to both words is a lightweight,
            # language-agnostic morphology signal, not a domain rule.
            if common_prefix >= 5 and common_prefix / shorter >= 0.6:
                related += 1
                break
    return related


_INFORMATIONAL_STOPWORDS = frozenset({
    "avec", "dans", "depuis", "pour", "sans", "sur", "vers", "entre", "mais", "plus", "moins",
    "peux", "peut", "trouve", "trouver", "cherche", "chercher", "veux", "voudrais", "faire",
    "quel", "quelle", "quels", "quelles", "est", "sont", "avoir", "avons", "nous", "vous",
    "this", "that", "with", "from", "into", "about", "find", "search", "want", "need", "can",
    "what", "which", "when", "where", "have", "been", "were", "will", "your", "their",
})


def _informational_terms(text: str) -> set[str]:
    """Extract topic-bearing words while leaving identifiers to anchor handling."""
    return {
        token.casefold()
        for token in re.findall(r"\b[^\W\d_]{4,}\b", text or "", flags=re.UNICODE)
        if token.casefold() not in _INFORMATIONAL_STOPWORDS
    }


def _anchor_families(text: str) -> set[str]:
    """Capture stable reference prefixes, e.g. AX-17 and AX-18 share AX."""
    families: set[str] = set()
    for anchor in _specific_anchors(text):
        prefix = re.match(r"[a-z]+", anchor)
        if prefix and len(prefix.group(0)) >= 2:
            families.add(prefix.group(0))
    return families


def _has_information_need_alignment(question: str, context: str) -> bool:
    """Require more than one coincidental entity before accepting local evidence.

    Alignment is established by two shared topical terms, or by a shared topic
    term reinforced by a morphological relation, matching reference, or model
    family. A lone city, person, organisation, or year is therefore not enough.
    """
    question_terms = _informational_terms(question)
    context_terms = _informational_terms(context)
    exact_terms = question_terms & context_terms
    if len(exact_terms) >= 2:
        return True

    requested = _specific_anchors(question)
    documented = _specific_anchors(context)
    exact_anchor = bool(requested & documented)
    family_match = bool(_anchor_families(question) & _anchor_families(context))
    morphology = _morphological_topic_overlap(question, context) > 0
    if exact_terms and (exact_anchor or family_match or morphology):
        return True
    # A shared technical predicate plus two distinct structured references is
    # meaningful related evidence even when the product family is expressed
    # only numerically (for example successive model numbers).
    return morphology and bool(requested) and bool(documented)


def _evaluate_evidence(
    question: str,
    blocks: List[Tuple[float, Dict]],
    *,
    guard_ok: bool,
    context_is_relevant: bool,
    overlap: int,
) -> EvidenceDecision:
    """Single source of truth for usable local evidence after final block selection."""
    if not blocks:
        return EvidenceDecision("none", "no_final_blocks", context_is_relevant, "empty_context", 0)

    requested = _specific_anchors(question)
    evidence_text = _evidence_text(blocks)
    documented = _specific_anchors(evidence_text)
    morphological_overlap = _morphological_topic_overlap(question, evidence_text)
    information_need_aligned = _has_information_need_alignment(question, evidence_text)
    relevance_reason = (
        "information_need_aligned" if context_is_relevant else
        ("information_need_misaligned" if not information_need_aligned else "lexical_relevance_below_threshold")
    )
    usable_direct_signal = bool(guard_ok or context_is_relevant or overlap >= OVERLAP_MIN)

    if requested and requested.issubset(documented) and usable_direct_signal:
        return EvidenceDecision(
            "direct", "requested_anchors_in_final_blocks", context_is_relevant, relevance_reason, morphological_overlap
        )
    if not requested and context_is_relevant and usable_direct_signal:
        return EvidenceDecision(
            "direct", "context_relevant_without_specific_anchor", context_is_relevant, relevance_reason, morphological_overlap
        )

    # A different reference can be useful only with reranker support and a
    # topical relation. Exact overlap helps, but cannot be an absolute gate:
    # equivalent requests regularly use different inflected word forms.
    topical_relation = information_need_aligned and bool(
        context_is_relevant or overlap >= OVERLAP_MIN or morphological_overlap
    )
    if requested and (documented - requested) and guard_ok and topical_relation:
        return EvidenceDecision(
            "related",
            "different_anchor_with_guard_and_topic_relation",
            context_is_relevant,
            relevance_reason,
            morphological_overlap,
        )

    # A topically relevant, reranker-approved context is still useful when it
    # has no alternative model/reference identifier. It supports a cautious
    # local answer, not an automatic Web fallback.
    if requested and context_is_relevant and information_need_aligned and guard_ok:
        return EvidenceDecision(
            "related",
            "topically_relevant_context_without_matching_anchor",
            context_is_relevant,
            relevance_reason,
            morphological_overlap,
        )

    if not guard_ok:
        reason = "guard_rejected_final_blocks"
    elif not topical_relation:
        reason = "no_topic_relation_for_related_evidence"
    elif requested and not (documented - requested):
        reason = "no_documented_related_anchor"
    else:
        reason = "final_blocks_not_usable_as_evidence"
    return EvidenceDecision("none", reason, context_is_relevant, relevance_reason, morphological_overlap)


def _classify_local_context_state(
    decision: EvidenceDecision,
    *,
    retrieval_sufficient: bool | None = None,
) -> LocalContextState:
    """Classify local evidence for Auto routing, independently of answer completeness.

    ``related`` and an incomplete iterative retrieval remain local: they can
    establish useful facts while being unable to settle the exact question.
    Only a context with no usable topical evidence is ``irrelevant``.
    """
    if decision.mode == "none":
        return "irrelevant"
    if decision.mode == "related" or retrieval_sufficient is False:
        return "relevant_but_incomplete"
    return "relevant_and_sufficient"

def _condense_question(hist: List[Dict], q: str) -> str:
    """
    Réécriture de la question en requête autonome (standalone query) en utilisant l'historique.
    Améliore le rappel sur les questions elliptiques ou avec des pronoms ambigus.
    Safe: timeout court (≤6s), fallback silencieux vers q original si erreur.
    """
    if not hist:
        return q
    try:
        # Prendre les 3-4 derniers messages pour le contexte
        hist_txt = "\n".join([f"{m.get('role','user')}: {m.get('content','')[:300]}" for m in hist[-4:]])

        # Extraire le sujet principal du dernier message utilisateur pour ancrage sémantique
        last_user_msg = ""
        for m in reversed(hist):
            if m.get("role") == "user":
                last_user_msg = m.get("content", "")[:200]
                break

        prompt = (
            "RÉÉCRITURE DE QUESTION:\n"
            "Tu dois réécrire la QUESTION en une requête autonome et concise.\n"
            "RÈGLES CRITIQUES:\n"
            "- Remplace les pronoms ambigus (il, ça, cette chose, cela) par les NOMS PRÉCIS de l'historique\n"
            "- GARDE le MÊME SUJET que la conversation en cours\n"
            "- Si la question est un simple follow-up (\"comment ?\", \"pourquoi ?\"), intègre le SUJET PRINCIPAL du message précédent\n"
            "- Rends UNIQUEMENT la requête réécrite (plain text, 1 phrase, max 30 mots)\n\n"
            f"HISTORIQUE:\n{hist_txt}\n\n"
            f"SUJET en cours: {last_user_msg}\n\n"
            f"QUESTION à réécrire:\n{q}\n\n"
            "REQUÊTE AUTONOME:"
        )
        out = _safe_llm(
            ask_mistral_with_context,
            prompt,
            context_text="",
            history=[],
            timeout=min(LLM_TIMEOUT_SEC, 6),
            max_tokens=80
        )
        s = (out or "").strip()
        # Validation: doit être raisonnable et ne pas diverger complètement
        if 8 <= len(s) <= 400:
            # Vérification basique: si le sujet change radicalement, fallback
            q_lower = q.lower()
            s_lower = s.lower()
            # Si la question originale contient des mots-clés importants (>4 chars), vérifier qu'ils ne disparaissent pas
            q_keywords = set(w for w in q_lower.split() if len(w) > 4)
            hist_keywords = set(w for w in last_user_msg.lower().split() if len(w) > 4)
            combined_keywords = q_keywords | hist_keywords
            s_words = set(s_lower.split())
            # Si aucun mot-clé important n'est présent dans la réécriture ET qu'on en avait, fallback
            if combined_keywords and not any(kw in s_lower for kw in combined_keywords):
                return q
            return s
    except Exception:
        pass
    return q

def _multi_query_expand(q: str, n: int = 3) -> List[str]:
    """
    Génère N variantes sémantiques de la requête pour améliorer le rappel documentaire.
    Retourne [q] + variantes (max n au total), dédupliquées.
    Safe: timeout court, fallback vers [q] si erreur.
    """
    try:
        prompt = (
            "MULTI-REQUÊTES:\n"
            "Génère des variantes DIVERSES et utiles de la requête ci-dessous pour améliorer la recherche documentaire.\n"
            f"Rends UNIQUEMENT un JSON compact sous la forme {{\"queries\":[...]}}, max {n} éléments.\n"
            "Chaque variante doit reformuler ou étendre la requête avec des synonymes ou perspectives différentes.\n\n"
            f"REQUÊTE:\n{q}\n"
        )
        raw = _safe_llm(
            ask_mistral_with_context,
            prompt,
            context_text="",
            history=[],
            timeout=min(LLM_TIMEOUT_SEC, 6),
            max_tokens=180
        )
        data = json.loads((raw or "{}").strip())
        arr = data.get("queries") or []
        cand = [str(x).strip() for x in arr if isinstance(x, (str, int, float))]
        cand = [c for c in cand if c and len(c) >= 5]
        # Déduplication case-insensitive
        uniq = []
        seen = set()
        for s in [q] + cand:
            k = s.lower()
            if k not in seen:
                seen.add(k)
                uniq.append(s)
        return uniq[:max(1, n)]
    except Exception:
        return [q]

def _merge_prelims(prelims: List[List[Tuple[float, Dict]]]) -> List[Tuple[float, Dict]]:
    """
    Fusionne plusieurs listes de résultats préliminaires [(score, meta), ...] en prenant le max score
    par chunk unique (identifié par meta['idx'] ou fallback sur (path, chunk_id)).
    Retourne une liste triée par score décroissant.
    """
    best = {}
    for arr in prelims:
        for sc, meta in (arr or []):
            try:
                mid = int(meta.get("idx"))
            except Exception:
                mid = None
            # Clé unique: idx si disponible, sinon (path, chunk_id)
            key = mid if mid is not None else (meta.get("path"), meta.get("chunk_id"))
            cur = best.get(key)
            if (cur is None) or (float(sc) > float(cur[0])):
                best[key] = (float(sc), meta)
    merged = list(best.values())
    merged.sort(key=lambda x: x[0], reverse=True)
    return merged

# --- Détection "actu" simple ---
FRESH_HINTS_DEFAULT = [
    r"\baujourd'hui\b",
    r"\ben\s+ce\s+moment\b",
    r"\bce\s*(matin|soir|mois|trimestre|semestre|week[-\s]*end)\b",
    r"\bderni(e|è)re?\s*(version|maj|mise\s*à\s*jour|news|actualit(é|e))\b",
    r"\b(CEO|PDG|chiffres?\s*d'affaire?s?|b(ê|e)ta|sortie|release|score|match|résultats?)\b",
    r"\b20(2[4-9]|3[0-5])\b",
]
FRESH_HINTS = [re.compile(p, re.IGNORECASE)
               for p in _RUNTIME_SETTINGS.conversation.fresh_news_keywords] \
               or [re.compile(p, re.IGNORECASE) for p in FRESH_HINTS_DEFAULT]

def _looks_fresh_news(q: str) -> bool:
    s = (q or "").lower()
    return any(p.search(s) for p in FRESH_HINTS)

def _parse_web_links(web_text: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if not web_text:
        return out
    blocks = [b for b in web_text.split("\n\n") if b.strip()]
    for b in blocks:
        lines = [l.strip() for l in b.splitlines() if l.strip()]
        if not lines:
            continue
        title_line = lines[0]
        url_line = ""
        for ln in lines[1:3]:
            if ln.startswith("http://") or ln.startswith("https://"):
                url_line = ln
                break
        title = title_line.replace("[WEB]", "").strip()
        if url_line:
            out.append({"title": title or url_line, "url": url_line})
    return out

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()

def _stable_hash(value: Any) -> str:
    """Hash structured execution inputs without retaining their content in logs."""
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _cache_key(
    *, conversation_id: str, q: str, mode_in: str, route_mode: str,
    reply_to: dict[str, str] | None, history_hash: str, rag_context_hash: str,
) -> str:
    """Key a final answer to its complete conversational evidence scope."""
    return _stable_hash({
        "conversation_id": conversation_id,
        "question": _norm(q),
        "source_mode": mode_in,
        "route_mode": route_mode,
        "reply_to": reply_to,
        "history_hash": history_hash,
        "rag_context_hash": rag_context_hash,
    })

def _log_event(request_id: str, payload: Dict):
    payload = {"request_id": request_id, **payload}
    print(json.dumps(payload, ensure_ascii=False))

# ============ LLM-as-router (mini appel) ============
def _llm_route(q: str, signals: dict, history: List[Dict]) -> str:
    """
    Retourne l'un des modes: smalltalk | math | strict_local | web_live | general
    Réponse attendue: JSON {"mode":"...","reason":"..."}
    """
    router_prompt = (
        "ROUTEUR:\n"
        "Tu es un *routeur* qui choisit un mode d'exécution pour répondre à l'utilisateur.\n"
        "Tu dois rendre EXCLUSIVEMENT un JSON compact, sans explication autour, de la forme:\n"
        "{\"mode\":\"smalltalk|math|strict_local|web_live|general\",\"reason\":\"...\"}\n\n"
        "Règles synthétiques:\n"
        "- math si la question est une équation ou un calcul.\n"
        "- smalltalk pour conversation banale (salutations, humeur, etc.).\n"
        "- strict_local si local_context_state vaut relevant_and_sufficient ou relevant_but_incomplete.\n"
        "- web_live UNIQUEMENT si local_context_state vaut irrelevant et que la question est adaptée à une recherche Web publique ou actuelle.\n"
        "- Une absence de réponse exacte, un modèle/référence différent(e), ou evidence_mode=related ne justifie JAMAIS web_live : réponds alors en strict_local avec prudence.\n"
        "- sinon general.\n\n"
        "Question:\n"
        f"{q}\n\n"
        "Signals (JSON):\n"
        f"{json.dumps(signals, ensure_ascii=False)}\n\n"
        "Réponds maintenant en JSON uniquement."
    )
    try:
        raw = _safe_llm(
            ask_mistral_with_context,
            router_prompt,
            context_text="",
            history=history,
            timeout=min(LLM_TIMEOUT_SEC, 8),
        )
        data = json.loads(raw.strip())
        mode = str(data.get("mode") or "").strip().lower()
        if mode in {"smalltalk", "math", "strict_local", "web_live", "general"}:
            return mode
    except FuturesTimeout:
        return "general"
    except Exception:
        return "general"
    return "general"


def _product_ids(text: str) -> set[str]:
    """Extrait les references produit numeriques en ignorant les annees."""

    return {
        token
        for token in re.findall(r"\b\d{3,6}\b", text or "")
        if not (1900 <= int(token) <= 2099)
    }


def _source_years(text: str) -> List[int]:
    current_year = datetime.now(timezone.utc).year
    return [
        int(year)
        for year in re.findall(r"\b(?:19|20)\d{2}\b", text or "")
        if 1900 <= int(year) <= current_year
    ]


def _post_generation_review(
    question: str,
    answer: str,
    context: str,
    sources: List[Dict[str, Any]],
    faithfulness: Optional[Dict[str, Any]] = None,
    claim_review: Optional[FaithfulnessReview] = None,
) -> PostGenerationReview:
    """Produit un commentaire additif. Cette fonction ne modifie jamais la reponse."""

    if claim_review and claim_review.caveat_required:
        problems = [
            item for item in claim_review.claims
            if item.status != ClaimStatus.SUPPORTED
            or (
                item.status == ClaimStatus.SUPPORTED
                and (item.citation_status == CitationStatus.MISMATCHED or item.citation_correct is False)
            )
        ]
        priority = {
            ClaimStatus.CONTRADICTED: 0,
            ClaimStatus.UNSUPPORTED: 1,
            ClaimStatus.PARTIALLY_SUPPORTED: 2,
            ClaimStatus.INFERRED: 3,
            ClaimStatus.SUPPORTED: 4,
        }
        problem = min(problems, key=lambda item: (
            priority[item.status],
            0 if item.citation_status == CitationStatus.MISMATCHED else 1,
        ))
        claim_text = problem.claim.text.strip().rstrip(".!?")
        if len(claim_text) > 160:
            claim_text = claim_text[:157].rstrip() + "..."
        if problem.status == ClaimStatus.SUPPORTED and (
            problem.citation_status == CitationStatus.MISMATCHED or problem.citation_correct is False
        ):
            return PostGenerationReview(
                status="CAVEAT",
                caveat_type="CITATION_MISMATCH",
                message=f"L'affirmation « {claim_text} » est soutenue par le contexte, mais pas par la source citée.",
                severity="warning",
            )
        if problem.status == ClaimStatus.CONTRADICTED:
            evidence = problem.evidence[0].text if problem.evidence else "une source disponible"
            return PostGenerationReview(
                status="CAVEAT",
                caveat_type="CONTRADICTED_CLAIM",
                message=f"L'affirmation « {claim_text} » semble contredite par la source : « {evidence[:180]} »",
                severity="warning",
            )
        if problem.status == ClaimStatus.UNSUPPORTED:
            return PostGenerationReview(
                status="CAVEAT",
                caveat_type="UNSUPPORTED_CLAIM",
                message=f"L'affirmation « {claim_text} » n'est pas directement soutenue par les documents retrouvés.",
                severity="warning",
            )
        if problem.status == ClaimStatus.PARTIALLY_SUPPORTED:
            return PostGenerationReview(
                status="CAVEAT",
                caveat_type="PARTIAL_EVIDENCE",
                message=f"Les sources ne confirment qu'une partie de l'affirmation « {claim_text} ».",
                severity="warning",
            )
        return PostGenerationReview(
            status="CAVEAT",
            caveat_type="INFERENCE",
            message=f"L'affirmation « {claim_text} » est déduite des sources sans y être formulée explicitement.",
            severity="info",
        )

    prompt = (
        "VERIFIEUR JSON:\n"
        "Evalue la prudence utile apres generation. Ne reecris pas la reponse. "
        "Rends uniquement un JSON compact avec status, caveat_type, message, severity, suggest_web.\n"
        "caveat_type vaut null ou STALE_SOURCE, INDIRECT_EVIDENCE, PARTIAL_EVIDENCE, "
        "CONFLICTING_EVIDENCE, INFERENCE, WEB_RECOMMENDED.\n"
        "Le message doit citer la reserve concrete (date, modele, partie manquante ou contradiction). "
        "N'ajoute rien si les preuves repondent directement et completement.\n"
        f"QUESTION:{question}\nANSWER:{answer}\nSOURCES_COUNT:{len(sources)}\n"
        f"CONTEXT:\n{(context or '')[:7000]}"
    )
    data: Dict[str, Any] = {}
    try:
        raw = _safe_llm(
            ask_mistral_with_context,
            prompt,
            context_text="",
            history=[],
            timeout=min(LLM_TIMEOUT_SEC, 8),
        )
        parsed = json.loads((raw or "{}").strip())
        if isinstance(parsed, dict):
            data = parsed
    except Exception:
        data = {}

    requested_type = str(data.get("caveat_type") or "").strip().upper()
    action = str(data.get("action") or "none").strip().lower()
    requested_message = str(data.get("message") or "").strip()
    severity = "warning" if str(data.get("severity") or "").lower() == "warning" else "info"
    suggest_web = bool(data.get("suggest_web")) or action == "ask_web"

    if requested_type in _CAVEAT_TYPES and requested_type != "WEB_RECOMMENDED":
        fallback_messages = {
            "STALE_SOURCE": "Les sources disponibles sont anciennes ; l'information peut avoir evolue depuis.",
            "INDIRECT_EVIDENCE": "Les sources portent sur une reference proche, pas directement sur celle demandee.",
            "PARTIAL_EVIDENCE": "Les sources disponibles ne couvrent qu'une partie de la question.",
            "CONFLICTING_EVIDENCE": "Les sources disponibles donnent des informations contradictoires sur ce point.",
            "INFERENCE": "Cette conclusion repose en partie sur une deduction qui n'est pas formulee explicitement dans les sources.",
        }
        return PostGenerationReview(
            status="CAVEAT",
            caveat_type=requested_type,  # type: ignore[arg-type]
            message=requested_message or fallback_messages[requested_type],
            severity=severity,
            suggest_web=suggest_web,
        )

    question_products = _product_ids(question)
    context_products = _product_ids(context)
    missing_products = sorted(question_products - context_products)
    if missing_products and context_products:
        requested = ", ".join(missing_products)
        available = ", ".join(sorted(context_products))
        return PostGenerationReview(
            status="CAVEAT",
            caveat_type="INDIRECT_EVIDENCE",
            message=(
                f"Les sources disponibles concernent {available}, pas directement {requested}. "
                "La conclusion est donc indirecte."
            ),
            severity="warning",
        )

    years = _source_years(context)
    current_year = datetime.now(timezone.utc).year
    if years and max(years) <= current_year - 2:
        latest_year = max(years)
        return PostGenerationReview(
            status="CAVEAT",
            caveat_type="STALE_SOURCE",
            message=(
                f"La source la plus recente retrouvee date de {latest_year} ; "
                "cette information peut avoir evolue depuis."
            ),
            severity="warning",
            suggest_web=True,
        )

    if faithfulness and should_fallback_to_general(faithfulness, strict_mode=True):
        return PostGenerationReview(
            status="CAVEAT",
            caveat_type="INFERENCE",
            message=(
                "Une partie de la reponse n'est pas explicitement confirmee par les sources fournies ; "
                "elle doit etre lue comme une inference."
            ),
            severity="warning",
        )

    if requested_type == "WEB_RECOMMENDED" or action == "ask_web":
        return PostGenerationReview(
            status="CAVEAT",
            caveat_type="WEB_RECOMMENDED",
            message=requested_message or (
                "Une verification web peut etre utile pour confirmer l'etat actuel, "
                "sans remettre en cause la reponse locale affichee."
            ),
            severity=severity,
            suggest_web=True,
        )

    return PostGenerationReview()


def run_answer_pipeline(
    body: AskIn,
    request: Request,
    *,
    token_sink: Optional[Callable[[str], None]] = None,
    artifact_sink: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    status_sink: Optional[Callable[[str, str], None]] = None,
) -> AnswerPipelineResult:
    request_id = current_request_id() or str(uuid.uuid4())
    set_stage("unknown")
    validations: Dict[str, Any] = {}
    q = validate_answer_request(body)

    def emit_status(stage: str, label: str) -> None:
        if status_sink:
            status_sink(stage, label)

    emit_status("analyze_request", "Analyse de la demande")
    trace_started = time.perf_counter()
    if RESPONSE_TRACE_ENABLED:
        RESPONSE_TRACES.start(request_id, {"original_user_message": q, "mode": body.source_mode or "auto", "conversation_id": body.thread_id})

    def trace(stage: str, data: Dict[str, Any]) -> None:
        if RESPONSE_TRACE_ENABLED:
            RESPONSE_TRACES.stage(request_id, stage, data)

    # The visible conversation lifecycle is independent from the response path:
    # persist the user turn before processing, then persist exactly one completed
    # assistant turn through ``finalize_result`` below.  This applies equally to
    # RAG, general, conversational and streaming requests.
    tenant_id, user_id = _try_get_auth_ids(request)
    chat_id = (body.thread_id or "").strip()
    user_turn_persisted = False
    if tenant_id and user_id:
        try:
            title = (q[:60] + "…") if len(q) > 60 else q
            chat_id = chat_id or str(uuid.uuid4())
            create_chat(tenant_id, user_id, title=title or "Nouveau chat", chat_id=chat_id, project_id=body.project_id)
            append_message(tenant_id, user_id, chat_id, "user", q, meta={"request_id": request_id})
            user_turn_persisted = True
        except Exception:
            # Persistence failures must never prevent a response.  If processing
            # later fails after a successful user write, that user turn remains.
            pass

    def finalize_result(result: AnswerPipelineResult) -> AnswerPipelineResult:
        """Persist and transport only the canonical reader-facing answer."""
        canonical_answer = enforce_final_citation_contract(result.answer, len(result.sources))
        result = replace(result, answer=canonical_answer)
        generated_chat_title: Optional[str] = None
        persisted_assistant_message_id: Optional[int] = None
        if tenant_id and user_id and user_turn_persisted:
            log_event(request_id, "message_persistence_start")
            set_stage("message_persistence")
            try:
                persisted_assistant_message_id = append_message(
                    tenant_id,
                    user_id,
                    chat_id,
                    "assistant",
                    result.answer,
                    meta={"mode": result.mode, "review": result.review.to_dict(), "request_id": request_id, "sources": result.sources, "artifacts": result.artifacts,
                          "generation_mode": result.route_mode or result.mode,
                          "response_format": result.validations.get("response_format", "normal"),
                          "timestamp": datetime.now(timezone.utc).isoformat(),
                          "evidence_provenance": result.validations.get("evidence_provenance", {}),
                          "retrieval_query": result.validations.get("retrieval_query")},
                )
            except Exception as exc:
                log_stream_error(request_id, "message_persistence", exc)
                # Keep the HTTP/SSE result available even if its assistant write fails.
            else:
                log_event(request_id, "message_persistence_end")
                try:
                    should_title = should_generate_chat_title(tenant_id, user_id, chat_id)
                except Exception:
                    should_title = False
                if should_title:
                    try:
                        candidate_title = generate_chat_title([
                            {"role": "user", "content": q},
                            {"role": "assistant", "content": result.answer},
                        ])
                        if candidate_title and set_generated_chat_title(tenant_id, user_id, chat_id, candidate_title):
                            generated_chat_title = candidate_title
                    except Exception:
                        # Titling is best-effort and must never affect the response lifecycle.
                        pass
                    finally:
                        if not generated_chat_title:
                            try:
                                mark_chat_title_generation_attempted(tenant_id, user_id, chat_id)
                            except Exception:
                                pass
        persisted_chat_id = chat_id if user_turn_persisted else result.chat_id
        final = replace(result, chat_id=(persisted_chat_id or None), chat_title=generated_chat_title, assistant_message_id=persisted_assistant_message_id or result.assistant_message_id)
        trace("answer", {"final_answer": final.answer, "artifacts": final.artifacts, "mode": final.mode, "sources": final.sources, "total_ms": round((time.perf_counter() - trace_started) * 1000, 1)})
        return final

    # --- Historique & thread ---
    if body.reply_history:
        raw_hist: List[Dict] = [{"role": m.role, "content": m.content, "meta": m.meta or {}} for m in body.reply_history]
        hist = trim_history(raw_hist, max_turns=REPLY_HISTORY_MAX_TURNS)
    else:
        raw_hist: List[Dict] = [{"role": m.role, "content": m.content, "meta": m.meta or {}} for m in (body.history or [])]
        hist = trim_history(raw_hist, max_turns=HISTORY_MAX_TURNS)

    trace("request", {"history_used": hist, "orchestrator_enabled": ORCHESTRATOR_SETTINGS.enabled, "provider": _RUNTIME_SETTINGS.generation.provider, "orchestrator_model": ORCHESTRATOR_SETTINGS.model or _RUNTIME_SETTINGS.generation.model, "generation_model": _RUNTIME_SETTINGS.generation.model})

    thread_id = (body.thread_id or "default").strip() or "default"
    _touch_session(thread_id)
    sess = SESSIONS.get(thread_id) or {}

    # --- Ciblage (reply_to) ---
    reply_preamble = ""
    if body.reply_to and (body.reply_to.content or "").strip():
        src_role = body.reply_to.role
        src_excerpt = (body.reply_to.content or "").strip()[:800]
        reply_preamble = (
            "CIBLE_REPONSE:\n"
            f"Role_origine: {src_role}\n"
            f"Extrait: {src_excerpt}\n"
            "FIN_CIBLE\n\n"
        )

    # --- Source mode demandé par l'utilisateur ---
    mode_in = (body.source_mode or "auto").strip().lower()
    if mode_in not in {"auto", "local", "general", "web_live", "web_index"}:
        mode_in = "auto"

    allowed_sources = None
    if mode_in == "local":
        allowed_sources = {"email", "pdf", "file"}
    elif mode_in == "web_index":
        allowed_sources = {"web"}

    # --- Roleplay on/off ---
    trig = None if ORCHESTRATOR_SETTINGS.enabled else detect_roleplay_trigger(hist, q)
    if trig is True:
        _set_roleplay(thread_id, True)
    elif trig is False:
        _set_roleplay(thread_id, False)
    roleplay_active = _get_roleplay(thread_id)

    # Roleplay actif (et pas factuel) -> réponse courte roleplay
    if (not ORCHESTRATOR_SETTINGS.enabled) and roleplay_active and (not looks_factual(q)):
        try:
            ans = _generate_answer(
                q, "", history=hist, token_sink=token_sink, progress_sink=emit_status,
                roleplay_mode=True, max_tokens=220,
            )
            return finalize_result(_result(answer=ans, sources=[], request_id=request_id))
        except FuturesTimeout:
            raise HTTPException(status_code=504, detail="LLM timeout (roleplay)")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # --- Small talk rapide ---
    try:
        skind = "" if ORCHESTRATOR_SETTINGS.enabled else classify_smalltalk_semantic(q, idx.embed_model)
    except Exception:
        skind = ""
    if (not ORCHESTRATOR_SETTINGS.enabled) and skind:
        try:
            ans = _generate_answer(
                q, reply_preamble, history=hist, token_sink=token_sink, progress_sink=emit_status,
                smalltalk_mode=True, smalltalk_kind=skind, max_tokens=200,
            )
            return finalize_result(_result(answer=ans, sources=[], request_id=request_id))
        except FuturesTimeout:
            raise HTTPException(status_code=504, detail="LLM timeout (smalltalk)")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # --- Suivi "explique" après équation précédente ---
    if (not ORCHESTRATOR_SETTINGS.enabled) and _is_explain_followup(q) and sess.get("last_math"):
        math_ans = _solve_math(sess["last_math"], detailed=True, timeout_sec=MATH_TIMEOUT_SEC)
        if math_ans:
            sess["no_context_once"] = True
            SESSIONS[thread_id] = sess
            return finalize_result(_result(answer=math_ans, sources=[], request_id=request_id))

    # --- Maths directes ---
    if (not ORCHESTRATOR_SETTINGS.enabled) and _looks_like_equation(q):
        math_ans = _solve_math(q, detailed=True, timeout_sec=MATH_TIMEOUT_SEC)
        if math_ans:
            sess["last_math"] = q
            sess["no_context_once"] = True
            SESSIONS[thread_id] = sess
            return finalize_result(_result(answer=math_ans, sources=[], request_id=request_id))

    # --- Mode "general" forcé par l'utilisateur ---
    if mode_in == "general":
        try:
            ans = _generate_answer(q, reply_preamble, history=hist, token_sink=token_sink, progress_sink=emit_status)
            return finalize_result(_result(answer=ans, sources=[], mode="GENERAL(no-context)", ctx_len=len(reply_preamble), request_id=request_id))
        except FuturesTimeout:
            raise HTTPException(status_code=504, detail="LLM timeout")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # One-shot orchestration is opt-in. Any failure falls through to the
    # unchanged legacy path; it never prevents a user request from completing.
    orchestration_plan: Optional[OrchestrationPlan] = None
    raw_orchestration_plan: Optional[OrchestrationPlan] = None
    orchestration_ms: Optional[float] = None
    orchestration_failure: dict[str, Any] | None = None
    documentary_orchestration_fallback = False
    orchestration_failed = False
    if ORCHESTRATOR_SETTINGS.enabled:
        emit_status("identify_response_mode", "Identification du mode de réponse")
        started = time.perf_counter()
        try:
            trace("orchestrator_input", {"system_prompt": build_prompt(q, hist).split("\nCURRENT_DATE_UTC:", 1)[0], "user_message": q, "history": hist, "active_subject_candidates": compact_history(hist).get("active_subject_candidates", [])})
            set_stage("orchestration")
            raw_orchestration_plan = _run_orchestration(q, hist)
            sanitized_plan = sanitize_plan_for_retrieval(raw_orchestration_plan, user_message=q)
            orchestration_plan = sanitized_plan.plan
            orchestration_ms = round((time.perf_counter() - started) * 1000, 1)
            constraints_count = len(orchestration_plan.temporal_constraints) + sum(
                bool(value) for value in orchestration_plan.metadata_constraints.model_dump().values()
            )
            _log_event(request_id, {
                "event": "orchestration", "orchestration_ms": orchestration_ms,
                "orchestration_intent": orchestration_plan.intent,
                "orchestration_needs_retrieval": orchestration_plan.needs_retrieval,
                "orchestration_use_history": orchestration_plan.use_history,
                "orchestration_reuse_previous_subject": orchestration_plan.reuse_previous_subject,
                # Preserve observability without exposing a sensitive standalone query.
                "orchestration_query": hashlib.sha256((orchestration_plan.retrieval_query or "").encode("utf-8")).hexdigest()[:12] if orchestration_plan.retrieval_query else None,
                "orchestration_constraints_count": constraints_count,
            })
            record_snapshot(raw_orchestration_plan, orchestration_ms=orchestration_ms, fallback_used=False)
            trace("orchestration", {
                "raw_orchestration_plan": raw_orchestration_plan.model_dump(mode="json"),
                "validated_orchestration_plan": orchestration_plan.model_dump(mode="json"),
                "constraints": {
                    "hard_filters": sanitized_plan.hard_filters,
                    "soft_preferences": sanitized_plan.soft_preferences,
                    "removed": sanitized_plan.removed_constraints,
                },
                "orchestration_latency_ms": orchestration_ms,
                "fallback_used": False,
                "validation_error": None,
                "validation_error_details": None,
                "raw_model_output": raw_orchestration_plan._raw_model_output,
                "error_category": None,
                "provider_error_type": None,
                "provider_error_code": None,
                "provider_error_message": None,
                "fallback_strategy": None,
            })
        except Exception as exc:
            orchestration_failed = True
            orchestration_ms = round((time.perf_counter() - started) * 1000, 1)
            orchestration_failure = _orchestration_failure_details(exc)
            documentary_orchestration_fallback = _looks_like_documentary_fallback(q, hist)
            fallback_strategy = "documentary_retrieval" if documentary_orchestration_fallback else (
                "conversation" if not _looks_like_documentary_fallback(q, hist) else "legacy_routing"
            )
            _log_event(request_id, {"event": "orchestration_fallback", "orchestration_ms": orchestration_ms, "reason": type(exc).__name__, "error_category": orchestration_failure["error_category"], "fallback_strategy": fallback_strategy})
            trace("orchestration_context_recovery", {
                "enabled": bool(hist), "recovered_query": _recover_query_from_history(q, hist),
            })
            record_snapshot(None, orchestration_ms=orchestration_ms, fallback_used=True)
            trace("orchestration", {
                "raw_orchestration_plan": None,
                "validated_orchestration_plan": None,
                "constraints": {"hard_filters": [], "soft_preferences": [], "removed": []},
                "orchestration_latency_ms": orchestration_ms,
                "fallback_used": True,
                "validation_error": type(exc).__name__,
                "validation_error_details": orchestration_failure["validation_error_details"],
                "raw_model_output": exc.raw_model_output if isinstance(exc, OrchestrationPlanOutputError) else None,
                "error_category": orchestration_failure["error_category"],
                "provider_error_type": orchestration_failure["provider_error_type"],
                "provider_error_code": orchestration_failure["provider_error_code"],
                "provider_error_message": orchestration_failure["provider_error_message"],
                "provider": orchestration_failure["provider"], "model": orchestration_failure["model"],
                "request_schema_version": orchestration_failure["request_schema_version"],
                "endpoint": orchestration_failure["endpoint"],
                "reasoning_effort": orchestration_failure["reasoning_effort"],
                "timeout": orchestration_failure["timeout"],
                "fallback_strategy": fallback_strategy,
            })

    # Explicit mode wins. In auto mode, the planner may explicitly choose the
    # live Web route. This branch intentionally runs before the local-index
    # readiness check and before any idx.search call.
    explicit_web_request = _has_explicit_web_request(q)
    email_draft_requested = bool(orchestration_plan and orchestration_plan.response_format == "email_draft")
    current_user_first_name = _sender_first_name_from_request(request) if email_draft_requested else None
    if email_draft_requested:
        # Private generation context from authenticated identity, never from
        # retrieved documents. The prompt owns the fallback placeholder.
        reply_preamble += f"\nCURRENT_USER_FIRST_NAME: {current_user_first_name or '[Prénom]'}\n"
        _log_event(request_id, {"event": "email_draft_identity", "current_user_first_name_available": bool(current_user_first_name)})
    local_web_override = mode_in == "local" and (
        bool(orchestration_plan and orchestration_plan.intent == "web_search" and orchestration_plan.web_request_explicit)
        or explicit_web_request
    )
    auto_live_web_request = (
        mode_in == "auto"
        and _looks_fresh_news(q)
        and not _looks_like_documentary_fallback(q, hist)
    )
    web_requested = mode_in == "web_live" or explicit_web_request or local_web_override or auto_live_web_request or (
        mode_in == "auto" and orchestration_plan is not None and orchestration_plan.intent == "web_search"
    )
    if web_requested:
        web_follow_up = bool(
            orchestration_plan
            and orchestration_plan.reuse_previous_subject
            and orchestration_plan.intent in {"refine_previous_search", "web_search"}
        )
        web_query = (
            orchestration_plan.retrieval_query
            if orchestration_plan is not None and orchestration_plan.intent == "web_search"
            else None
        ) or (
            _explicit_web_query(q, hist) if explicit_web_request else
            (_recover_query_from_history(q, hist) if orchestration_failed else q)
        )
        if web_follow_up:
            web_query = resolve_retrieval_query(
                raw_user_message=q, orchestrator_query=web_query, history=hist,
            )
        trace("web_retrieval_input", {
            "raw_user_message": q, "web_query": web_query,
            "follow_up_web_search": web_follow_up,
            "orchestration_intent": orchestration_plan.intent if orchestration_plan else None,
        })
        emit_status("search_web", "Recherche sur internet")
        try:
            web_text = _with_timeout(
                web_search_context, web_query, max_chars=WEB_MAX_CHARS,
                k=WEB_RESULT_K, timeout=WEB_TIMEOUT_SEC,
            ) or ""
        except FuturesTimeout:
            return finalize_result(_result(
                answer="Recherche web trop longue. Reessaie ou passe en mode local.", sources=[],
                mode="STRICT(web_live)", ctx_len=0, request_id=request_id, route_mode="web_live",
                validations={"evidence_mode": "web_live"},
            ))
        except Exception:
            web_text = ""

        web_sources_list = _parse_web_links(web_text)
        if not web_text.strip():
            return finalize_result(_result(
                answer="Je n'ai rien trouve via la recherche web en direct.", sources=[],
                mode="STRICT(web_live)", ctx_len=0, request_id=request_id, route_mode="web_live",
                validations={"evidence_mode": "web_live"},
            ))

        emit_status("analyze_web_sources", "Analyse des sources trouvees")
        context_for_llm = f"{reply_preamble}{web_text}"
        try:
            generated = _generate_answer(
                q, context_for_llm, history=hist, token_sink=token_sink, artifact_sink=artifact_sink, progress_sink=emit_status,
                evidence_mode="web_live",
                response_format="email_draft" if email_draft_requested else "normal",
            )
        except FuturesTimeout:
            raise HTTPException(status_code=504, detail="LLM timeout")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM error: {e}")

        artifacts: List[Dict[str, str]] = []
        if email_draft_requested:
            try:
                answer, artifacts, citations_idx = _parse_email_draft_generation(generated)
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                logger.warning("email_draft_generation_invalid: %s", type(exc).__name__)
                answer, citations_idx = extract_citations_and_clean_answer(generated)
        else:
            answer, citations_idx = extract_citations_and_clean_answer(generated)
        selected_web_indices, web_citation_map = select_web_source_indices(len(web_sources_list), citations_idx)
        answer = remap_inline_citations(answer, web_citation_map)

        claim_review = None
        faithfulness_result = None
        if ENABLE_POST_GENERATION_REVIEW and ENABLE_FAITHFULNESS_CHECK:
            try:
                evidence_blocks = [{
                    "text": web_text, "chunk_uid": "web-live-context",
                    "path": web_sources_list[0]["url"] if web_sources_list else None,
                }]
                claim_review = verify_answer_claims("\n\n".join([answer, *(item["content"] for item in artifacts)]), evidence_blocks, cited_source_indices=citations_idx, use_nli=True)
                faithfulness_result = claim_review.legacy_summary()
                validations["faithfulness"] = {"performed": True, **faithfulness_result}
                validations["claim_faithfulness"] = {"performed": True, **claim_review.to_dict()}
            except Exception:
                pass
        sources = [{"path": web_sources_list[index - 1]["url"], "chunk": -1} for index in selected_web_indices]
        review = PostGenerationReview()
        if ENABLE_POST_GENERATION_REVIEW:
            review = _post_generation_review(q, answer, web_text, sources, faithfulness_result, claim_review)
            validations["post_answer"] = {"performed": True, **review.to_dict()}
        validations.update({
            "evidence_mode": "web_live",
            "orchestration_intent": orchestration_plan.intent if orchestration_plan else None,
            "orchestration_needs_retrieval": orchestration_plan.needs_retrieval if orchestration_plan else None,
        })
        _log_event(request_id, {
            "event": "ask", "mode_in": mode_in, "mode_out": "STRICT(web_live)",
            "route_mode": "web_live", "web_query": web_query, "web_links": len(web_sources_list),
            "local_retrieval_executed": False,
        })
        return finalize_result(_result(
            answer=answer, artifacts=artifacts, sources=sources, mode="STRICT(web_live)", ctx_len=len(context_for_llm),
            review=review.to_dict(), faithfulness_review=claim_review.to_dict() if claim_review else None,
            request_id=request_id, route_mode="web_live", validations=validations,
        ))

    if mode_in == "auto" and orchestration_plan is not None and not orchestration_plan.needs_retrieval:
        catalog_probe = None
        if orchestration_plan.intent == "general_question":
            set_stage("catalog_probe")
            catalog_probe = probe_document_catalog(q, getattr(idx, "corpus", None))
            trace("catalog_probe", catalog_probe.trace())
            if catalog_probe.strong_match:
                orchestration_plan = orchestration_plan.model_copy(update={
                    "intent": "document_question", "needs_retrieval": True, "retrieval_query": q,
                    "response_strategy": "answer",
                })
        else:
            trace("catalog_probe", {"executed": False, "strong_match": False, "reason": "intent_not_general_question", "timing_ms": 0.0})
    if mode_in == "auto" and orchestration_plan is not None and not orchestration_plan.needs_retrieval:
        conversational = orchestration_plan.response_strategy == "ask_for_missing_information" and not email_draft_requested
        previous_meta, previous_index = _previous_validated_evidence(hist)
        reuse_previous_evidence = bool(
            orchestration_plan.use_history
            and orchestration_plan.reuse_previous_subject
            and previous_meta
        )
        trace("follow_up_evidence_reuse", {
            "follow_up_reused_evidence": reuse_previous_evidence,
            "previous_evidence_source_message_id": previous_meta.get("message_id") if previous_meta else None,
            "previous_history_index": previous_index,
        })
        try:
            generated = _generate_answer(
                q, reply_preamble, history=hist,
                token_sink=token_sink, artifact_sink=artifact_sink, progress_sink=emit_status,
                conversational_mode=conversational, max_tokens=160 if conversational else None,
                response_format="email_draft" if email_draft_requested else "normal",
            )
            artifacts: List[Dict[str, str]] = []
            if email_draft_requested:
                try:
                    ans, artifacts, _ = _parse_email_draft_generation(generated)
                except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                    logger.warning("email_draft_generation_invalid: %s", type(exc).__name__)
                    ans = generated
            else:
                ans = generated
            return finalize_result(_result(
                answer=ans, artifacts=artifacts, sources=(previous_meta.get("sources", []) if reuse_previous_evidence else []), request_id=request_id, route_mode="general",
                mode="GENERAL(orchestrated-conversation)" if conversational else "GENERAL(orchestrated)",
                ctx_len=len(reply_preamble),
                validations={"orchestration_intent": orchestration_plan.intent,
                             "orchestration_needs_retrieval": False, "evidence_mode": "none",
                             "response_format": orchestration_plan.response_format,
                             "evidence_provenance": (previous_meta.get("evidence_provenance", {}) if reuse_previous_evidence else {}),
                             "follow_up_reused_evidence": reuse_previous_evidence,
                             "previous_evidence_source_message_id": previous_meta.get("message_id") if previous_meta else None},
            ))
        except FuturesTimeout:
            raise HTTPException(status_code=504, detail="LLM timeout")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # Decide whether this turn is answerable now before rewriting or searching.
    turn_decision = classify_turn(q, idx.embed_model, hist) if orchestration_plan is None and not documentary_orchestration_fallback else None
    turn_type = turn_decision.turn_type if turn_decision else ("documentary_fallback" if documentary_orchestration_fallback else "orchestrated")
    if turn_type == "conversational_continuation":
        _log_event(request_id, {
            "event": "ask", "turn_type": turn_type, "q_eff": q,
            "should_condense": False, "route_mode": "general", "guard_ok": False,
            "hits": 0, "context_blocks": 0, "evidence_mode": "none",
            "turn_type_reason": turn_decision.decision_reason,
            "turn_type_margin": getattr(turn_decision, "semantic_margin", None),
        })
        try:
            ans = _generate_answer(
                q, reply_preamble, history=hist, token_sink=token_sink, progress_sink=emit_status,
                conversational_mode=True, max_tokens=160,
            )
            return finalize_result(_result(
                answer=ans, sources=[], mode="GENERAL(conversational)",
                ctx_len=len(reply_preamble), request_id=request_id,
                route_mode="general", validations={"turn_type": turn_type, "evidence_mode": "none"},
            ))
        except FuturesTimeout:
            raise HTTPException(status_code=504, detail="LLM timeout")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # ===================== RAG (pré-recherche pour signaux) =====================
    # TIER 0 amélioration: condensation de question avec historique
    # Désactiver si historique trop court (risque de dérive) - compter messages utilisateur uniquement
    user_msg_count_for_condense = len([m for m in hist if m.get("role") == "user"])
    should_condense = (hist and user_msg_count_for_condense >= 2 and ENABLE_CONDENSATION and orchestration_plan is None)
    if orchestration_plan:
        q_eff = orchestration_plan.retrieval_query
    elif orchestration_failed:
        condensed = _condense_question(hist, q) if should_condense else q
        # If the same provider outage also prevented condensation, retain the
        # active subject deterministically instead of searching the raw turn.
        q_eff = condensed if _norm(condensed) != _norm(q) else _recover_query_from_history(q, hist)
    else:
        q_eff = _condense_question(hist, q) if should_condense else q

    # Protection: si question très vague (<30 chars pure question) ET pas assez d'historique => forcer GENERAL
    # Questions typiques: "Comment ça marche ?", "Pourquoi ?", "Explique", "How does it work?"
    q_words = [w for w in q.strip().split() if len(w) > 2]  # mots significatifs
    is_vague_question = (
        len(q_words) <= 5 and  # question courte (max 5 mots significatifs)
        any(w in q.lower() for w in ["comment", "pourquoi", "quoi", "how", "why", "what", "explain", "explique", "c'est quoi", "ça marche"])
    )
    # Compter UNIQUEMENT les messages utilisateur dans l'historique (pas les réponses assistant)
    user_msg_count = len([m for m in hist if m.get("role") == "user"])
    if mode_in == "auto" and is_vague_question and user_msg_count < 2 and orchestration_plan is None:
        # Pas assez de contexte conversationnel pour ancrer la recherche => GENERAL direct
        _log_event(request_id, {"event": "vague_question_fallback", "q": q, "user_msg_count": user_msg_count})
        try:
            ans = _generate_answer(q, reply_preamble, history=hist, token_sink=token_sink, progress_sink=emit_status)
            return finalize_result(_result(answer=ans, sources=[], mode="GENERAL(vague-no-history)", ctx_len=len(reply_preamble), request_id=request_id))
        except FuturesTimeout:
            raise HTTPException(status_code=504, detail="LLM timeout")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    if not _local_index_ready():
        if mode_in in {"local", "web_index"}:
            return finalize_result(_result(
                answer=_missing_index_message(mode_in),
                sources=[],
                mode="STRICT(local)",
                ctx_len=0,
                request_id=request_id,
            ))
        try:
            ans = _generate_answer(q, reply_preamble, history=hist, token_sink=token_sink, progress_sink=emit_status)
            return finalize_result(_result(answer=ans, sources=[], mode="GENERAL(no-index)", ctx_len=len(reply_preamble), request_id=request_id))
        except FuturesTimeout:
            raise HTTPException(status_code=504, detail="LLM timeout")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    if orchestration_plan and orchestration_plan.source_types:
        planned_sources = set(orchestration_plan.source_types)
        allowed_sources = planned_sources if allowed_sources is None else (allowed_sources & planned_sources)
        if not allowed_sources:
            # An explicit source mode remains authoritative; an empty
            # intersection must not accidentally mean "all sources" to idx.search.
            allowed_sources = {"__no_matching_source__"}
    follow_up = bool(orchestration_plan and orchestration_plan.intent == "refine_previous_search" and orchestration_plan.reuse_previous_subject)
    resolved_retrieval_query = resolve_retrieval_query(raw_user_message=q, orchestrator_query=orchestration_plan.retrieval_query or q, history=hist) if follow_up else None
    retrieval_queries = build_retrieval_queries(original_query=q, orchestrator_query=orchestration_plan.retrieval_query, resolved_query=resolved_retrieval_query, follow_up=follow_up) if orchestration_plan else [("original_autonomous", q_eff)]
    if resolved_retrieval_query:
        q_eff = resolved_retrieval_query
    trace("retrieval_input", {"raw_user_message": q, "resolved_retrieval_query": resolved_retrieval_query, "follow_up_retrieval": follow_up, "raw_query_used_for_retrieval": not follow_up, "raw_query_exclusion_reason": "conversational_followup" if follow_up else None, "original_query": q, "orchestrator_query": orchestration_plan.retrieval_query if orchestration_plan else None, "q_eff": q_eff, "q_eff_source": "resolved_retrieval_query" if resolved_retrieval_query else ("orchestrator_query" if orchestration_plan else ("condensed_query" if should_condense else "original_query")), "retrieval_queries": [{"type": kind, "source": kind, "query": query} for kind, query in retrieval_queries], "should_condense": should_condense, "condensed_query": q_eff if should_condense else None, "allowed_sources": sorted(allowed_sources) if allowed_sources else None, "source_types": orchestration_plan.source_types if orchestration_plan else [], "temporal_constraints": [item.model_dump(mode="json") for item in orchestration_plan.temporal_constraints] if orchestration_plan else [], "metadata_constraints": orchestration_plan.metadata_constraints.model_dump() if orchestration_plan else {}})
    set_stage("retrieval")
    if allowed_sources == {"email"}:
        emit_status("search_emails", "Recherche dans vos e-mails")
    elif allowed_sources == {"web"}:
        emit_status("search_indexed_sources", "Recherche dans vos sources indexées")
    elif allowed_sources and "email" not in allowed_sources:
        emit_status("search_documents", "Recherche dans vos documents")
    else:
        emit_status("search_local_sources", "Recherche dans vos documents et e-mails")
    _log_event(request_id, {"event": "rag_search_start", "q_eff": q_eff, "should_condense": should_condense, "hist_len": len(hist), "orchestrated": orchestration_plan is not None})

    retrieval_started = time.perf_counter()
    query_rankings = []
    query_trace = []
    ce_scores = []
    rejected_candidates = []
    for query_kind, query_text in retrieval_queries:
        rows, scores = idx.search(query_text, retrieve_k=RETRIEVE_K, top_k_faiss=TOP_K_FAISS, hybrid_alpha=HYBRID_ALPHA, use_rerank=ENABLE_RERANKER, allowed_sources=allowed_sources)
        if query_kind == "original_autonomous" or not ce_scores:
            ce_scores = scores
        before_filter = len(rows)
        if orchestration_plan:
            rejected_candidates.extend([{"query_type": query_kind, "filename": meta.get("file"), "source": meta.get("source"), "score": score, **reason} for score, meta in rows if (reason := explain_candidate_rejection(meta, orchestration_plan))])
            rows = filter_retrieval_candidates(rows, orchestration_plan)
        query_rankings.append((query_kind, rows))
        query_trace.append({"type": query_kind, "query": query_text, "candidate_count": before_filter, "after_constraint_filter": len(rows)})
    prelim = reciprocal_rank_fusion(query_rankings) if len(query_rankings) > 1 else (query_rankings[0][1] if query_rankings else [])
    emit_status("analyze_results", "Analyse des résultats trouvés")
    before_constraints = sum(item["candidate_count"] for item in query_trace)
    trace("retrieval", {"retrieval_ms": round((time.perf_counter() - retrieval_started) * 1000, 1), "retrieval_queries": query_trace, "idx_search_candidates": before_constraints, "after_constraint_filter": len(prelim), "candidate_fusion": {"fusion_method": "rrf" if len(query_rankings) > 1 else "single_query", "unique_candidates_before_final_pool": len(prelim)}, "initial_candidates": [{"filename": meta.get("file"), "source": meta.get("source"), "score": score, "retrieved_by": meta.get("retrieved_by"), "per_query_rank": meta.get("per_query_rank"), "per_query_score": meta.get("per_query_score"), "fusion_score": meta.get("fusion_score"), "document_metadata": meta.get("document_metadata") or {}} for score, meta in prelim[:20]], "rejected_candidates": rejected_candidates[:20]})
    prelim_hits = len(prelim)
    fused = fuse_contiguous_passages(prelim, gap=FUSE_ADJACENT_GAP)
    blocks = clip_context_blocks(fused, max_chars=MAX_CONTEXT_CHARS, keep=FINAL_K)
    emit_status("select_passages", "Sélection des passages pertinents")

    # Bounded evidence enrichment happens after the stable first retrieval and
    # before evidence_mode/generation. It follows only existing corpus links.
    retrieval_rounds = []
    iterative_sufficiency = None
    if mode_in in {"local", "web_index"}:
        route_mode = "strict_local"
    elif orchestration_plan is not None:
        set_stage("iterative_retrieval")
        def on_retrieval_action(action) -> None:
            if action.type == "EXPAND":
                emit_status("expand_document", "Expansion du document")
            elif action.relation == "attachment":
                emit_status("inspect_attachment", "Lecture des pièces jointes")
            else:
                emit_status("follow_related_source", "Analyse des sources liées")

        blocks, retrieval_rounds, iterative_sufficiency = run_iterative_evidence_retrieval(
            candidate_pool=prelim,
            initial_evidence=blocks,
            corpus=list(getattr(idx, "corpus", []) or []),
            query=q_eff,
            semantics=orchestration_plan.query_semantics,
            max_rounds=3,
            max_expanded_chunks=4,
            on_action=on_retrieval_action,
        )
        blocks = clip_context_blocks(blocks, max_chars=MAX_CONTEXT_CHARS, keep=FINAL_K)
        trace("retrieval_rounds", {
            "rounds": retrieval_rounds,
            "stop_reason": iterative_sufficiency.reason,
            "evidence_sufficient": iterative_sufficiency.sufficient,
            "final_evidence_chunk_uids": [meta.get("chunk_uid") for _, meta in blocks],
            "final_evidence_member_chunk_uids": [
                member for _, meta in blocks
                for member in (meta.get("fused_chunk_uids") or [meta.get("chunk_uid")])
                if member
            ],
            "core_evidence_chunk_uids": retrieval_rounds[-1].get("core_evidence_chunk_uids", []) if retrieval_rounds else [],
            "optional_evidence_chunk_uids": retrieval_rounds[-1].get("optional_evidence_chunk_uids", []) if retrieval_rounds else [],
            "anchor_documents": retrieval_rounds[0].get("anchor_documents", []) if retrieval_rounds else [],
        })

    gated_ok = answerability_guard(ce_scores, threshold=ANS_THRESHOLD)
    context_local = format_context_for_llm(blocks) if blocks else ""
    overlap = keyword_overlap_count(q, context_local)

    # --- NEW: Vérification de cohérence thématique du contexte ---
    context_is_relevant = _check_context_relevance(q, context_local)
    emit_status("verify_sources", "Vérification des sources")
    evidence_decision = _evaluate_evidence(
        q, blocks, guard_ok=bool(gated_ok), context_is_relevant=context_is_relevant, overlap=overlap,
    )
    evidence_mode = evidence_decision.mode
    trace("evidence", {"evidence_mode": evidence_mode, "evidence_mode_reason": evidence_decision.reason, "context_is_relevant": evidence_decision.context_is_relevant, "context_relevance_reason": evidence_decision.context_relevance_reason, "morphological_topic_overlap": evidence_decision.morphological_topic_overlap, "guard_ok": bool(gated_ok), "strict_local_ok": evidence_mode in {"direct", "related"}, "final_context_blocks": [{"metadata": meta, "score": score, "text": meta.get("text")} for score, meta in blocks]})

    # Only the evidence decision controls whether local blocks are usable.
    strict_local_ok = evidence_mode in {"direct", "related"}
    local_context_state = _classify_local_context_state(
        evidence_decision,
        retrieval_sufficient=(iterative_sufficiency.sufficient if iterative_sufficiency is not None else None),
    )

    # TIER 0 amélioration: multi-query expansion si peu de hits ou faible overlap OU contexte hors sujet
    # The legacy broad expansion is disabled when the bounded, gap-driven
    # retry is enabled below.  It must not consume a second retrieval round
    # before answerability has identified what is actually missing.
    if ENABLE_EXPANSION and not ENABLE_INTELLIGENT_RETRY and (not strict_local_ok or not context_is_relevant) and prelim_hits < max(4, RETRIEVE_K // 2):
        try:
            variants = _multi_query_expand(q_eff, n=3)
            all_prelims = [prelim]
            for vq in variants[1:]:  # première variante est déjà q_eff
                try:
                    p2, _ = idx.search(
                        vq,
                        retrieve_k=RETRIEVE_K,
                        top_k_faiss=TOP_K_FAISS,
                        hybrid_alpha=HYBRID_ALPHA,
                        use_rerank=ENABLE_RERANKER,
                        allowed_sources=allowed_sources,
                    )
                    if orchestration_plan:
                        p2 = filter_retrieval_candidates(p2, orchestration_plan)
                    all_prelims.append(p2)
                except Exception:
                    continue
            # Fusionner les résultats (dédup par idx)
            merged = _merge_prelims(all_prelims)
            fused = fuse_contiguous_passages(merged, gap=FUSE_ADJACENT_GAP)
            blocks = clip_context_blocks(fused, max_chars=MAX_CONTEXT_CHARS, keep=FINAL_K)
            context_local = format_context_for_llm(blocks) if blocks else ""
            overlap = keyword_overlap_count(q, context_local)
            context_is_relevant = _check_context_relevance(q, context_local)
            evidence_decision = _evaluate_evidence(
                q, blocks, guard_ok=bool(gated_ok), context_is_relevant=context_is_relevant, overlap=overlap,
            )
            evidence_mode = evidence_decision.mode
            strict_local_ok = evidence_mode in {"direct", "related"}
            local_context_state = _classify_local_context_state(
                evidence_decision,
                retrieval_sufficient=(iterative_sufficiency.sufficient if iterative_sufficiency is not None else None),
            )
        except Exception:
            # Fallback silencieux: on garde les résultats initiaux
            pass

    # This is deliberately after all bounded structural enrichment and any
    # query expansion.  It governs local answer generation, not retrieval.
    answerability_decision: AnswerabilityDecision = evaluate_answerability(
        query=q,
        evidence_mode=evidence_mode,
        context_is_relevant=context_is_relevant,
        evidence_sufficient=(iterative_sufficiency.sufficient if iterative_sufficiency is not None else None),
        reranker_accepted=bool(gated_ok),
        blocks=[meta for _, meta in blocks],
    )
    trace("answerability", answerability_decision.to_dict())

    # Phase 2: one cumulative, gap-driven retry before generation.  This is
    # intentionally non-recursive: the second decision is final.
    initial_evidence_chunk_uids = [str(meta.get("chunk_uid")) for _, meta in blocks if meta.get("chunk_uid")]
    retry_trace: Dict[str, Any] = {
        "retry_triggered": False,
        "retry_reason": None,
        "missing_aspects": [],
        "preserved_anchors": [],
        "retry_queries": [],
        "retry_queries_rejected": [],
        "initial_evidence_chunk_uids": initial_evidence_chunk_uids,
        "retry_evidence_chunk_uids": [],
        "merged_evidence_chunk_uids": initial_evidence_chunk_uids,
        "answerability_before_retry": answerability_decision.status,
        "answerability_after_retry": answerability_decision.status,
        "supported_aspects_before_retry": [],
        "supported_aspects_after_retry": [],
    }
    if ENABLE_INTELLIGENT_RETRY and MAX_RETRY_ROUNDS == 1 and answerability_decision.status in {"partial", "unanswerable"}:
        retry_gap = derive_retrieval_gap(
            query=q_eff, context_text=context_local, answerability=answerability_decision,
            evidence_mode=evidence_mode, evidence_reason=evidence_decision.reason,
        )
        retry_queries, retry_rejected = build_retry_queries(
            original_user_query=q, orchestrator_query=(orchestration_plan.retrieval_query if orchestration_plan else None),
            resolved_retrieval_query=resolved_retrieval_query or q_eff, gap=retry_gap,
            max_queries=MAX_RETRY_QUERIES,
        )
        retry_trace.update({
            "retry_reason": retry_gap.reason,
            "missing_aspects": retry_gap.missing_aspects,
            "preserved_anchors": retry_gap.anchors,
            "supported_aspects_before_retry": retry_gap.supported_aspects,
            "retry_queries": retry_queries,
            "retry_queries_rejected": retry_rejected,
        })
        if retry_queries:
            retry_trace["retry_triggered"] = True
            emit_status("retry_retrieval", "Recherche complementaire ciblee")
            retry_rankings = []
            retry_scores: list[float] = []
            for retry_index, retry_query in enumerate(retry_queries, start=1):
                rows, scores = idx.search(
                    retry_query, retrieve_k=RETRIEVE_K, top_k_faiss=TOP_K_FAISS,
                    hybrid_alpha=HYBRID_ALPHA, use_rerank=ENABLE_RERANKER,
                    allowed_sources=allowed_sources,
                )
                if orchestration_plan:
                    rows = filter_retrieval_candidates(rows, orchestration_plan)
                retry_rankings.append((f"retry_{retry_index}", rows))
                retry_scores.extend(scores)
            retry_prelim = reciprocal_rank_fusion(retry_rankings) if len(retry_rankings) > 1 else (retry_rankings[0][1] if retry_rankings else [])
            retry_trace["retry_evidence_chunk_uids"] = [str(meta.get("chunk_uid")) for _, meta in retry_prelim if meta.get("chunk_uid")]
            # Never replace the first pass: deduplicate by uid while retaining
            # first-pass score and all provenance on a repeated chunk.
            # Include first-pass structural additions too (email bodies,
            # thread neighbours, attachments): they are already-confirmed
            # evidence and must survive even if retry ranking omits them.
            prelim = merge_cumulative_evidence([*prelim, *blocks], retry_prelim, retry_query_count=len(retry_queries))
            fused = fuse_contiguous_passages(prelim, gap=FUSE_ADJACENT_GAP)
            blocks = clip_context_blocks(fused, max_chars=MAX_CONTEXT_CHARS, keep=FINAL_K)
            if orchestration_plan is not None:
                blocks, retry_rounds, iterative_sufficiency = run_iterative_evidence_retrieval(
                    candidate_pool=prelim, initial_evidence=blocks,
                    corpus=list(getattr(idx, "corpus", []) or []), query=q_eff,
                    semantics=orchestration_plan.query_semantics, max_rounds=3,
                    max_expanded_chunks=4,
                )
                retrieval_rounds.extend(retry_rounds)
                blocks = clip_context_blocks(blocks, max_chars=MAX_CONTEXT_CHARS, keep=FINAL_K)
            ce_scores.extend(retry_scores)
            gated_ok = answerability_guard(ce_scores, threshold=ANS_THRESHOLD)
            context_local = format_context_for_llm(blocks) if blocks else ""
            overlap = keyword_overlap_count(q, context_local)
            context_is_relevant = _check_context_relevance(q, context_local)
            evidence_decision = _evaluate_evidence(q, blocks, guard_ok=bool(gated_ok), context_is_relevant=context_is_relevant, overlap=overlap)
            evidence_mode = evidence_decision.mode
            strict_local_ok = evidence_mode in {"direct", "related"}
            local_context_state = _classify_local_context_state(
                evidence_decision,
                retrieval_sufficient=(iterative_sufficiency.sufficient if iterative_sufficiency is not None else None),
            )
            answerability_decision = evaluate_answerability(
                query=q, evidence_mode=evidence_mode, context_is_relevant=context_is_relevant,
                evidence_sufficient=(iterative_sufficiency.sufficient if iterative_sufficiency is not None else None),
                reranker_accepted=bool(gated_ok), blocks=[meta for _, meta in blocks],
            )
            after_gap = derive_retrieval_gap(
                query=q_eff, context_text=context_local, answerability=answerability_decision,
                evidence_mode=evidence_mode, evidence_reason=evidence_decision.reason,
            )
            retry_trace.update({
                "merged_evidence_chunk_uids": [str(meta.get("chunk_uid")) for _, meta in blocks if meta.get("chunk_uid")],
                "answerability_after_retry": answerability_decision.status,
                "supported_aspects_after_retry": after_gap.supported_aspects,
            })
    trace("intelligent_retry", retry_trace)
    if retry_trace["retry_triggered"]:
        trace("evidence_after_retry", {
            "evidence_mode": evidence_mode,
            "evidence_mode_reason": evidence_decision.reason,
            "context_is_relevant": context_is_relevant,
            **answerability_decision.to_dict(),
        })

    # --- Signaux pour routeur LLM ---
    signals = {
        "hits": int(prelim_hits),
        "guard_ok": bool(gated_ok),
        "overlap": int(overlap),
        "overlap_min": int(OVERLAP_MIN),
        "ctx_len": int(len(context_local or "")),
        "context_is_relevant": bool(context_is_relevant),
        "a_des_dates_recentes": bool(_looks_fresh_news(q)),
        "looks_equation": bool(_looks_like_equation(q)),
        "smalltalk_hint": bool(skind),
        "evidence_mode": evidence_mode,
        "local_context_state": local_context_state,
    }

    # --- Choix du mode (règle déterministe actu ⇒ web_live, hors sujet ⇒ general) ---
    if mode_in in {"local", "web_index"}:
        # Explicit local/indexed-source modes are never overridden by a plan.
        route_mode = "strict_local"
    elif mode_in == "auto" and local_context_state != "irrelevant":
        # Any useful local evidence wins over an automatic Web fallback,
        # including evidence that cannot settle the exact requested case.
        route_mode = "strict_local"
    elif documentary_orchestration_fallback:
        route_mode = "strict_local"
    elif (
        orchestration_plan is not None
        and orchestration_plan.needs_retrieval
        and orchestration_plan.intent != "web_search"
    ):
        # A documentary request with insufficient local evidence must not
        # silently become a general-knowledge technical answer.
        route_mode = "strict_local"
    else:
        # With irrelevant local context, the router may select Web for a
        # public/Web-suitable question; it cannot do so for partial evidence.
        route_mode = _llm_route(q, signals, hist)

    # --- Garde-fou pro : si routeur dit "general" mais local pertinent -> forcer local ---
    if route_mode == "general" and strict_local_ok and mode_in != "web_live":
        route_mode = "strict_local"

    _log_event(request_id, {
        "event": "evidence_decision",
        "evidence_mode": evidence_mode,
        "evidence_mode_reason": evidence_decision.reason,
        "context_is_relevant": evidence_decision.context_is_relevant,
        "context_relevance_reason": evidence_decision.context_relevance_reason,
        "morphological_topic_overlap": evidence_decision.morphological_topic_overlap,
        "strict_local_ok": strict_local_ok,
        "local_context_state": local_context_state,
        **answerability_decision.to_dict(),
        "guard_ok": bool(gated_ok),
        "hits": len(prelim),
        "final_context_block_count": len(blocks),
        "top_hit_filenames": [str(meta.get("file") or Path(str(meta.get("path") or "")).name) for _, meta in prelim[:3]],
        "top_hit_scores": [round(float(score), 4) for score, _ in prelim[:3]],
    })

    # ------ Cache court (intègre le route_mode) ------
    rag_context_hash = hashlib.sha256((context_local or "").encode("utf-8")).hexdigest()
    rag_chunk_uids = [str(meta.get("chunk_uid") or f"{meta.get('document_id')}:{meta.get('chunk_id')}") for _, meta in blocks]
    validations["response_format"] = orchestration_plan.response_format if orchestration_plan else "normal"
    validations["retrieval_query"] = q_eff
    validations["answerability"] = answerability_decision.to_dict()
    validations["intelligent_retry"] = retry_trace
    validations["evidence_provenance"] = {
        "evidence_sufficient": bool(iterative_sufficiency.sufficient) if iterative_sufficiency is not None else bool(strict_local_ok),
        "evidence_chunk_uids": rag_chunk_uids,
        "evidence_document_ids": sorted({str(meta.get("document_id")) for _, meta in blocks if meta.get("document_id")}),
        "anchor_documents": retrieval_rounds[0].get("anchor_documents", []) if retrieval_rounds else [],
    }
    history_hash = _stable_hash(hist)
    cache_conversation_id = chat_id or (body.thread_id or "").strip()
    reply_to_cache_value = ({
        "id": str(body.reply_to.id), "role": str(body.reply_to.role), "content": str(body.reply_to.content),
    } if body.reply_to else None)
    # The web context is fetched after this point, so it cannot safely share a
    # final-answer cache entry before its actual context is known.
    cache_enabled = bool(cache_conversation_id) and route_mode != "web_live"
    ck = _cache_key(
        conversation_id=cache_conversation_id, q=q, mode_in=mode_in, route_mode=route_mode,
        reply_to=reply_to_cache_value, history_hash=history_hash, rag_context_hash=rag_context_hash,
    ) if cache_enabled else None
    cached = response_cache.get(ck) if ck else None
    cache_diagnostic = {
        "event": "response_cache", "cache_key_hash": ck, "cache_enabled": cache_enabled, "cache_hit": bool(cached),
        "conversation_id": cache_conversation_id or None, "chat_id": chat_id or None,
        "history_hash": history_hash, "rag_context_hash": rag_context_hash, "rag_chunk_uids": rag_chunk_uids,
    }
    _log_event(request_id, cache_diagnostic)
    trace("response_cache", cache_diagnostic)
    if cached:
        cached["request_id"] = request_id
        _log_event(request_id, {
            "event": "generation_result", "provider_call_effective": False,
            "reason": "response_cache_hit",
            "response_hash": hashlib.sha256(str(cached.get("answer") or "").encode("utf-8")).hexdigest(),
        })
        return finalize_result(_result(**cached, route_mode=route_mode))

    # -------- Verrou post-maths + RESPECT strict du route_mode --------
    if sess.get("no_context_once"):
        context_for_llm = reply_preamble
        use_strict = False
        sess["no_context_once"] = False
        SESSIONS[thread_id] = sess
    else:
        if route_mode == "strict_local":
            context_for_llm = reply_preamble + (context_local or "")
            use_strict = True
        elif route_mode == "web_live":
            # on posera le contexte après la recherche web
            context_for_llm = reply_preamble
            use_strict = True
        else:
            # GENERAL => jamais de contexte local
            context_for_llm = reply_preamble
            use_strict = False

    # ------------------- Exécution selon le mode -------------------
    web_sources_list: List[Dict[str, str]] = []
    mode_label = "GENERAL(no-context)"  # défaut

    # A) web_live
    if route_mode == "web_live":
        emit_status("search_web", "Recherche sur internet")
        try:
            web_text = _with_timeout(
                web_search_context,
                q_eff,
                max_chars=WEB_MAX_CHARS,
                k=WEB_RESULT_K,
                timeout=WEB_TIMEOUT_SEC,
            ) or ""
        except FuturesTimeout:
            out = {"answer": "Recherche web trop longue. Réessaie ou passe en mode local.", "sources": [], "mode": "STRICT(web_live)", "ctx_len": 0}
            if cache_enabled:
                response_cache.set(ck, out)
            return finalize_result(_result(**out, request_id=request_id, route_mode=route_mode, validations=validations))
        except Exception:
            web_text = ""

        web_sources_list = _parse_web_links(web_text)
        emit_status("analyze_web_sources", "Analyse des sources trouvées")
        if len(web_sources_list) > 1:
            emit_status("cross_reference", "Croisement des informations")
        if not web_text.strip():
            out = {"answer": "Je n’ai rien trouvé via la **recherche web en direct**.", "sources": [], "mode": "STRICT(web_live)", "ctx_len": 0}
            if cache_enabled:
                response_cache.set(ck, out)
            return finalize_result(_result(**out, request_id=request_id, route_mode=route_mode, validations=validations))

        context_for_llm = f"{reply_preamble}{web_text}"
        use_strict = True
        blocks = []
        mode_label = "STRICT(web_live)"

    # B) strict_local
    elif route_mode == "strict_local":
        if not blocks:
            msg = "Je n’ai rien trouvé de pertinent dans les **sources autorisées**."
            out = {"answer": msg, "sources": [], "mode": "STRICT(local)", "ctx_len": 0}
            if cache_enabled:
                response_cache.set(ck, out)
            return finalize_result(_result(**out, request_id=request_id, route_mode=route_mode, validations=validations))

        if answerability_decision.status == "unanswerable":
            msg = "Je n’ai pas trouvé suffisamment d’informations dans les documents disponibles pour répondre de manière fiable à cette question."
            out = {"answer": msg, "sources": [], "mode": "STRICT(local)", "ctx_len": len(context_local)}
            if cache_enabled:
                response_cache.set(ck, out)
            return finalize_result(_result(
                **out, request_id=request_id, route_mode=route_mode,
                status="abstained", abstention_reason=msg, validations=validations,
            ))

        if evidence_mode == "none":
            msg = "J’ai parcouru tes **sources**, mais rien de suffisamment pertinent."
            out = {"answer": msg, "sources": [], "mode": "STRICT(local)", "ctx_len": len(context_local)}
            if cache_enabled:
                response_cache.set(ck, out)
            return finalize_result(_result(**out, request_id=request_id, route_mode=route_mode, validations=validations))

        context_for_llm = reply_preamble + context_local
        use_strict = True
        mode_label = "STRICT(local)"

    # C) general (par défaut) -> pas de contexte local

    # --- Sanity check : pas de contexte local en GENERAL ---
    if mode_label == "GENERAL(no-context)":
        assert context_for_llm == reply_preamble, "GENERAL ne doit pas embarquer de contexte local"

    # ===================== Appel LLM principal =====================
    try:
        generation_started = time.perf_counter()
        trace("generation_input", {"generation_mode": mode_label, "question": q, "context_length": len(context_for_llm or ""), "context": context_for_llm, "provider": _RUNTIME_SETTINGS.generation.provider, "model": _RUNTIME_SETTINGS.generation.model, "reasoning_effort": _RUNTIME_SETTINGS.generation.reasoning_effort, "configured_temperature": _RUNTIME_SETTINGS.generation.strict_temperature if use_strict else _RUNTIME_SETTINGS.generation.temperature, "configured_top_p": _RUNTIME_SETTINGS.generation.strict_top_p if use_strict else _RUNTIME_SETTINGS.generation.top_p, "rag_context_hash": rag_context_hash, "rag_chunk_uids": rag_chunk_uids, "system_prompt": "constructed by rag_core.llm from generation mode and evidence mode", "evidence_mode": evidence_mode if use_strict else "none"})
        set_stage("generation")
        generated = _generate_answer(
            q, context_for_llm, history=hist, token_sink=token_sink, artifact_sink=artifact_sink, progress_sink=emit_status,
            evidence_mode="web_live" if route_mode == "web_live" else (evidence_mode if use_strict else "none"),
            answerability=(answerability_decision.status if use_strict and route_mode == "strict_local" else "answerable"),
            response_format="email_draft" if email_draft_requested else "normal",
        )
        trace("generation", {"generation_total_ms": round((time.perf_counter() - generation_started) * 1000, 1), "first_token_ms": None if token_sink is None else None})
        # The citation tail is private metadata. Keep it out of the answer
        # while preserving its indices for source selection and validation.
        artifacts = []
        if email_draft_requested:
            try:
                answer, artifacts, citations_idx = _parse_email_draft_generation(generated)
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                logger.warning("email_draft_generation_invalid: %s", type(exc).__name__)
                answer, citations_idx = extract_citations_and_clean_answer(generated)
        else:
            answer, citations_idx = extract_citations_and_clean_answer(generated)
    except FuturesTimeout:
        raise HTTPException(status_code=504, detail="LLM timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # ---------- Claim-level faithfulness (post-generation, non destructif) ----------
    faithfulness_result: Optional[Dict[str, Any]] = None
    claim_review: Optional[FaithfulnessReview] = None
    context_for_check = ""
    if ENABLE_POST_GENERATION_REVIEW and ENABLE_FAITHFULNESS_CHECK:
        try:
            # Déterminer le contexte pour la vérification (local strict ou web strict)
            if route_mode == "strict_local":
                context_for_check = context_local or ""
            elif route_mode == "web_live":
                # web_text peut ne pas exister si pas de branche web; utiliser locals() pour vérifier
                context_for_check = locals().get("web_text", "") or ""

            # Si demandé en strict only, ne vérifier que quand on a un contexte strict
            should_check_here = (not FAITHFULNESS_STRICT_ONLY) or (route_mode in {"strict_local", "web_live"})
            if should_check_here and (context_for_check.strip()):
                if route_mode == "strict_local":
                    evidence_blocks = [dict(block[1]) for block in blocks]
                else:
                    evidence_blocks = [{
                        "text": context_for_check,
                        "chunk_uid": "web-live-context",
                        "path": web_sources_list[0]["url"] if web_sources_list else None,
                    }]
                claim_review = verify_answer_claims(
                    "\n\n".join([answer, *(item["content"] for item in artifacts)]),
                    evidence_blocks,
                    cited_source_indices=citations_idx,
                    use_nli=True,
                )
                faithfulness_result = claim_review.legacy_summary()
                validations["faithfulness"] = {"performed": True, **faithfulness_result}
                validations["claim_faithfulness"] = {
                    "performed": True,
                    **claim_review.to_dict(),
                }
        except Exception:
            # Si le check échoue, on n'empêche pas la réponse
            pass

    # ===================== Sources à renvoyer =====================
    if use_strict and context_for_llm and web_sources_list:
        # web strict
        selected_web_indices, web_citation_map = select_web_source_indices(len(web_sources_list), citations_idx)
        answer = remap_inline_citations(answer, web_citation_map)
        sources = [{"path": web_sources_list[index - 1]["url"], "chunk": -1} for index in selected_web_indices]
    elif use_strict and blocks:
        # Context indices identify chunks; the source panel identifies unique
        # documents. Both use this exact mapping.
        sources, citation_map = select_cited_references(blocks, citations_idx)
        answer = remap_inline_citations(answer, citation_map)
        set_stage("source_serialization")
    else:
        sources = []

    if not sources:
        answer = remap_inline_citations(answer, {})

    # Contrat API conservé : lorsque les post-checks sont désactivés, la revue
    # reste neutre et aucun travail n'est exécuté après la génération.
    review = PostGenerationReview()
    if ENABLE_POST_GENERATION_REVIEW:
        review = _post_generation_review(
            q,
            answer,
            context_for_check or (context_for_llm if use_strict else ""),
            sources,
            faithfulness_result,
            claim_review,
        )
        validations["post_answer"] = {"performed": True, **review.to_dict()}

    # Logs structurés
    _log_event(request_id, {
        "event": "ask",
        "mode_in": mode_in,
        "mode_out": mode_label,
        "route_mode": route_mode,
        "hits": len(prelim),
        "reply_to": bool(body.reply_to),
        "strict": use_strict,
        "web_links": len(web_sources_list),
        "ctx_len": len(context_for_llm or ""),
        "context_blocks": len(blocks),
        "q_eff": q_eff,
        "should_condense": should_condense,
        "overlap": overlap,
        "guard_ok": bool(gated_ok),
        "turn_type": turn_type,
        "turn_type_reason": turn_decision.decision_reason if turn_decision else "orchestrator_primary",
        "turn_type_margin": getattr(turn_decision, "semantic_margin", None) if turn_decision else None,
        "orchestration_intent": orchestration_plan.intent if orchestration_plan else None,
        "orchestration_needs_retrieval": orchestration_plan.needs_retrieval if orchestration_plan else None,
        "response_format": orchestration_plan.response_format if orchestration_plan else "normal",
        "orchestration_use_history": orchestration_plan.use_history if orchestration_plan else None,
        "orchestration_reuse_previous_subject": orchestration_plan.reuse_previous_subject if orchestration_plan else None,
        "orchestration_ms": orchestration_ms,
        "evidence_mode": evidence_mode,
        "evidence_mode_reason": evidence_decision.reason,
        "context_is_relevant": evidence_decision.context_is_relevant,
        "context_relevance_reason": evidence_decision.context_relevance_reason,
        "morphological_topic_overlap": evidence_decision.morphological_topic_overlap,
        "strict_local_ok": strict_local_ok,
        "local_context_state": local_context_state,
        "source_count": len(sources),
    })

    out = {
        "answer": answer,
        "artifacts": artifacts,
        "sources": sources,
        "mode": mode_label,
        "ctx_len": len(context_for_llm or ""),
        "review": review.to_dict(),
        "faithfulness_review": claim_review.to_dict() if claim_review else None,
    }
    _log_event(request_id, {
        "event": "generation_result", "provider_call_effective": True,
        "response_hash": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
    })
    if cache_enabled and not body.reply_to:
        response_cache.set(ck, out)
    return finalize_result(_result(
        **out,
        request_id=request_id,
        route_mode=route_mode,
        validations=validations,
    ))
