# -*- coding: utf-8 -*-
import re, json, uuid, hashlib
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List, Dict, Literal, Optional, Tuple
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
from runtime_settings import get_runtime_settings
from .orchestration import (
    OrchestrationPlan, OrchestrationPlanOutputError, build_prompt, compact_history,
    explain_candidate_rejection, filter_retrieval_candidates, plan_once,
    sanitize_plan_for_retrieval,
)
from .orchestration_debug import record_snapshot
from .response_trace import ResponseTraceStore
from .iterative_retrieval import run_iterative_evidence_retrieval
from .chats_db import create_chat, append_message
from auth_ms import verify_ms_token

# --- Anti-429: concurrence + retries ---
import threading, random, time


AnswerStatus = Literal["answered", "abstained", "fallback"]
EvidenceMode = Literal["direct", "related", "none"]
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
    sources: List[Dict[str, Any]] = field(default_factory=list)
    mode: Optional[str] = None
    status: AnswerStatus = "answered"
    context_length: Optional[int] = None
    request_id: Optional[str] = None
    chat_id: Optional[str] = None
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
            sources=self.sources,
            mode=self.mode,
            ctx_len=self.context_length,
            request_id=self.request_id,
            chat_id=self.chat_id,
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
    sources: Optional[List[Dict[str, Any]]] = None,
    mode: Optional[str] = None,
    ctx_len: Optional[int] = None,
    request_id: Optional[str] = None,
    chat_id: Optional[str] = None,
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
        sources=list(sources or []),
        mode=mode,
        status=status or inferred_status,
        context_length=ctx_len,
        request_id=request_id,
        chat_id=chat_id,
        route_mode=route_mode,
        abstention_reason=abstention_reason or inferred_abstention,
        fallback_reason=fallback_reason or inferred_fallback,
        validations=dict(validations or {}),
        review=review or PostGenerationReview(),
        faithfulness_review=faithfulness_review,
    )


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
    """One planner call only: no retry loop and no retrieval side effect."""
    if not LLM_SEM.acquire(timeout=5):
        raise RuntimeError("orchestrator_busy")
    try:
        return plan_once(
            q,
            # Keep enough alternating user/assistant turns to retain the
            # substantive subject before a terse source-selection follow-up.
            history[-max(ORCHESTRATOR_SETTINGS.history_max_messages, 6):],
            lambda prompt, **kwargs: _with_timeout(ask_mistral_with_context, prompt, **kwargs),
            model=ORCHESTRATOR_SETTINGS.model or _RUNTIME_SETTINGS.generation.model,
            timeout=ORCHESTRATOR_SETTINGS.timeout,
        )
    finally:
        LLM_SEM.release()


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
    **kwargs,
) -> str:
    """Genere en sync ou transmet chaque fragment au transport SSE."""

    if token_sink is None:
        return _safe_llm(
            ask_mistral_with_context,
            question,
            context_text=context_text,
            history=history,
            timeout=LLM_TIMEOUT_SEC,
            **kwargs,
        )

    parts: List[str] = []
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
            token_sink(text)
    return "".join(parts)

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
    # Ratio mots clés partagés / mots clés question
    ratio = overlap / max(1, q_words)
    return ratio >= threshold


def _specific_anchors(text: str) -> set[str]:
    """Extract stable, domain-neutral identifiers (serial/model/reference-like tokens)."""
    return {
        token.lower()
        for token in re.findall(r"\b(?=[\w-]*\d)[\w-]{2,}\b", text or "")
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
    relevance_reason = "lexical_relevance" if context_is_relevant else "lexical_relevance_below_threshold"
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
    topical_relation = bool(context_is_relevant or overlap >= OVERLAP_MIN or morphological_overlap)
    if requested and (documented - requested) and guard_ok and topical_relation:
        return EvidenceDecision(
            "related",
            "different_anchor_with_guard_and_topic_relation",
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

def _cache_key(q: str, mode_in: str, route_mode: str, reply_to: Optional[str]) -> str:
    base = f"{_norm(q)}|{mode_in}|{route_mode}|{bool(reply_to)}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

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
        "- strict_local si le contexte trouvé est pertinent (guard_ok=true ET overlap>=overlap_min).\n"
        "- web_live si on a besoin d'info *fraîche* (actu/données qui changent vite) OU si strict_local est faux ET a_des_dates_recentes=true.\n"
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
) -> AnswerPipelineResult:
    request_id = str(uuid.uuid4())
    validations: Dict[str, Any] = {}
    q = validate_answer_request(body)
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
            create_chat(tenant_id, user_id, title=title or "Nouveau chat", chat_id=chat_id)
            append_message(tenant_id, user_id, chat_id, "user", q, meta={"request_id": request_id})
            user_turn_persisted = True
        except Exception:
            # Persistence failures must never prevent a response.  If processing
            # later fails after a successful user write, that user turn remains.
            pass

    def finalize_result(result: AnswerPipelineResult) -> AnswerPipelineResult:
        """Persist one complete assistant message after any successful path."""
        if tenant_id and user_id and user_turn_persisted:
            try:
                append_message(
                    tenant_id,
                    user_id,
                    chat_id,
                    "assistant",
                    result.answer,
                    meta={"mode": result.mode, "review": result.review.to_dict(), "request_id": request_id},
                )
            except Exception:
                # Keep the HTTP/SSE result available even if its assistant write fails.
                pass
        persisted_chat_id = chat_id if user_turn_persisted else result.chat_id
        final = replace(result, chat_id=(persisted_chat_id or None))
        trace("answer", {"final_answer": final.answer, "mode": final.mode, "sources": final.sources, "total_ms": round((time.perf_counter() - trace_started) * 1000, 1)})
        return final

    # --- Historique & thread ---
    if body.reply_history:
        raw_hist: List[Dict] = [{"role": m.role, "content": m.content} for m in body.reply_history]
        hist = trim_history(raw_hist, max_turns=REPLY_HISTORY_MAX_TURNS)
    else:
        raw_hist: List[Dict] = [{"role": m.role, "content": m.content} for m in (body.history or [])]
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
                q, "", history=hist, token_sink=token_sink,
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
                q, reply_preamble, history=hist, token_sink=token_sink,
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
    if (not ORCHESTRATOR_SETTINGS.enabled) and mode_in == "general":
        try:
            ans = _generate_answer(q, reply_preamble, history=hist, token_sink=token_sink)
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
    if ORCHESTRATOR_SETTINGS.enabled:
        started = time.perf_counter()
        try:
            trace("orchestrator_input", {"system_prompt": build_prompt(q, hist).split("\nCURRENT_DATE_UTC:", 1)[0], "user_message": q, "history": hist, "active_subject_candidates": compact_history(hist).get("active_subject_candidates", [])})
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
            })
        except Exception as exc:
            orchestration_ms = round((time.perf_counter() - started) * 1000, 1)
            _log_event(request_id, {"event": "orchestration_fallback", "orchestration_ms": orchestration_ms, "reason": type(exc).__name__})
            record_snapshot(None, orchestration_ms=orchestration_ms, fallback_used=True)
            trace("orchestration", {
                "raw_orchestration_plan": None,
                "validated_orchestration_plan": None,
                "constraints": {"hard_filters": [], "soft_preferences": [], "removed": []},
                "orchestration_latency_ms": orchestration_ms,
                "fallback_used": True,
                "validation_error": type(exc).__name__,
                "validation_error_details": exc.validation_error_details if isinstance(exc, OrchestrationPlanOutputError) else None,
                "raw_model_output": exc.raw_model_output if isinstance(exc, OrchestrationPlanOutputError) else None,
            })

    if orchestration_plan is not None and not orchestration_plan.needs_retrieval:
        conversational = orchestration_plan.response_strategy == "ask_for_missing_information"
        try:
            ans = _generate_answer(q, reply_preamble, history=hist, token_sink=token_sink,
                                   conversational_mode=conversational, max_tokens=160 if conversational else None)
            return finalize_result(_result(
                answer=ans, sources=[], request_id=request_id, route_mode="general",
                mode="GENERAL(orchestrated-conversation)" if conversational else "GENERAL(orchestrated)",
                ctx_len=len(reply_preamble),
                validations={"orchestration_intent": orchestration_plan.intent,
                             "orchestration_needs_retrieval": False, "evidence_mode": "none"},
            ))
        except FuturesTimeout:
            raise HTTPException(status_code=504, detail="LLM timeout")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # Decide whether this turn is answerable now before rewriting or searching.
    turn_decision = classify_turn(q, idx.embed_model, hist) if orchestration_plan is None else None
    turn_type = turn_decision.turn_type if turn_decision else "orchestrated"
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
                q, reply_preamble, history=hist, token_sink=token_sink,
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
    q_eff = orchestration_plan.retrieval_query if orchestration_plan else (_condense_question(hist, q) if should_condense else q)

    # Protection: si question très vague (<30 chars pure question) ET pas assez d'historique => forcer GENERAL
    # Questions typiques: "Comment ça marche ?", "Pourquoi ?", "Explique", "How does it work?"
    q_words = [w for w in q.strip().split() if len(w) > 2]  # mots significatifs
    is_vague_question = (
        len(q_words) <= 5 and  # question courte (max 5 mots significatifs)
        any(w in q.lower() for w in ["comment", "pourquoi", "quoi", "how", "why", "what", "explain", "explique", "c'est quoi", "ça marche"])
    )
    # Compter UNIQUEMENT les messages utilisateur dans l'historique (pas les réponses assistant)
    user_msg_count = len([m for m in hist if m.get("role") == "user"])
    if is_vague_question and user_msg_count < 2 and orchestration_plan is None:
        # Pas assez de contexte conversationnel pour ancrer la recherche => GENERAL direct
        _log_event(request_id, {"event": "vague_question_fallback", "q": q, "user_msg_count": user_msg_count})
        try:
            ans = _generate_answer(q, reply_preamble, history=hist, token_sink=token_sink)
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
            ans = _generate_answer(q, reply_preamble, history=hist, token_sink=token_sink)
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
    trace("retrieval_input", {"original_query": q, "orchestrator_query": orchestration_plan.retrieval_query if orchestration_plan else None, "q_eff": q_eff, "q_eff_source": "orchestrator_query" if orchestration_plan else ("condensed_query" if should_condense else "original_query"), "should_condense": should_condense, "condensed_query": q_eff if should_condense else None, "allowed_sources": sorted(allowed_sources) if allowed_sources else None, "source_types": orchestration_plan.source_types if orchestration_plan else [], "temporal_constraints": [item.model_dump(mode="json") for item in orchestration_plan.temporal_constraints] if orchestration_plan else [], "metadata_constraints": orchestration_plan.metadata_constraints.model_dump() if orchestration_plan else {}})
    _log_event(request_id, {"event": "rag_search_start", "q_eff": q_eff, "should_condense": should_condense, "hist_len": len(hist), "orchestrated": orchestration_plan is not None})

    retrieval_started = time.perf_counter()
    prelim, ce_scores = idx.search(
        q_eff,
        retrieve_k=RETRIEVE_K,
        top_k_faiss=TOP_K_FAISS,
        hybrid_alpha=HYBRID_ALPHA,
        use_rerank=ENABLE_RERANKER,
        allowed_sources=allowed_sources,
    )
    before_constraints = len(prelim)
    rejected_candidates = []
    if orchestration_plan:
        rejected_candidates = [{"filename": meta.get("file"), "source": meta.get("source"), "score": score, **reason} for score, meta in prelim if (reason := explain_candidate_rejection(meta, orchestration_plan))]
        prelim = filter_retrieval_candidates(prelim, orchestration_plan)
    trace("retrieval", {"retrieval_ms": round((time.perf_counter() - retrieval_started) * 1000, 1), "idx_search_candidates": before_constraints, "after_constraint_filter": len(prelim), "initial_candidates": [{"filename": meta.get("file"), "source": meta.get("source"), "score": score, "document_metadata": meta.get("document_metadata") or {}} for score, meta in prelim[:20]], "rejected_candidates": rejected_candidates[:20]})
    prelim_hits = len(prelim)
    fused = fuse_contiguous_passages(prelim, gap=FUSE_ADJACENT_GAP)
    blocks = clip_context_blocks(fused, max_chars=MAX_CONTEXT_CHARS, keep=FINAL_K)

    # Bounded evidence enrichment happens after the stable first retrieval and
    # before evidence_mode/generation. It follows only existing corpus links.
    retrieval_rounds = []
    if orchestration_plan is not None:
        blocks, retrieval_rounds, iterative_sufficiency = run_iterative_evidence_retrieval(
            candidate_pool=prelim,
            initial_evidence=blocks,
            corpus=list(getattr(idx, "corpus", []) or []),
            query=q_eff,
            semantics=orchestration_plan.query_semantics,
            max_rounds=3,
            max_expanded_chunks=4,
        )
        blocks = clip_context_blocks(blocks, max_chars=MAX_CONTEXT_CHARS, keep=FINAL_K)
        trace("retrieval_rounds", {
            "rounds": retrieval_rounds,
            "stop_reason": iterative_sufficiency.reason,
            "evidence_sufficient": iterative_sufficiency.sufficient,
            "final_evidence_chunk_uids": [meta.get("chunk_uid") for _, meta in blocks],
        })

    gated_ok = answerability_guard(ce_scores, threshold=ANS_THRESHOLD)
    context_local = format_context_for_llm(blocks) if blocks else ""
    overlap = keyword_overlap_count(q, context_local)

    # --- NEW: Vérification de cohérence thématique du contexte ---
    context_is_relevant = _check_context_relevance(q, context_local)
    evidence_decision = _evaluate_evidence(
        q, blocks, guard_ok=bool(gated_ok), context_is_relevant=context_is_relevant, overlap=overlap,
    )
    evidence_mode = evidence_decision.mode
    trace("evidence", {"evidence_mode": evidence_mode, "evidence_mode_reason": evidence_decision.reason, "context_is_relevant": evidence_decision.context_is_relevant, "context_relevance_reason": evidence_decision.context_relevance_reason, "morphological_topic_overlap": evidence_decision.morphological_topic_overlap, "guard_ok": bool(gated_ok), "strict_local_ok": evidence_mode in {"direct", "related"}, "final_context_blocks": [{"metadata": meta, "score": score, "text": meta.get("text")} for score, meta in blocks]})

    # Only the evidence decision controls whether local blocks are usable.
    strict_local_ok = evidence_mode in {"direct", "related"}

    # TIER 0 amélioration: multi-query expansion si peu de hits ou faible overlap OU contexte hors sujet
    if ENABLE_EXPANSION and (not strict_local_ok or not context_is_relevant) and prelim_hits < max(4, RETRIEVE_K // 2):
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
        except Exception:
            # Fallback silencieux: on garde les résultats initiaux
            pass

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
    }

    # --- Choix du mode (règle déterministe actu ⇒ web_live, hors sujet ⇒ general) ---
    if orchestration_plan is not None:
        # In this path the plan is the only authority deciding to retrieve.
        # evidence_mode still determines whether the retrieved blocks are usable.
        route_mode = "strict_local"
    elif mode_in in {"local", "web_index", "web_live"}:
        route_mode = {
            "local": "strict_local",
            "web_index": "strict_local",
            "web_live": "web_live"
        }[mode_in]
    else:
        # Si contexte hors sujet ET pas de hits solides => GENERAL direct
        if not context_is_relevant and prelim_hits < max(3, RETRIEVE_K // 3):
            route_mode = "general"
        elif _looks_fresh_news(q):
            route_mode = "web_live"
        else:
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
        "guard_ok": bool(gated_ok),
        "hits": len(prelim),
        "final_context_block_count": len(blocks),
        "top_hit_filenames": [str(meta.get("file") or Path(str(meta.get("path") or "")).name) for _, meta in prelim[:3]],
        "top_hit_scores": [round(float(score), 4) for score, _ in prelim[:3]],
    })

    # ------ Cache court (intègre le route_mode) ------
    ck = _cache_key(q, mode_in, route_mode, body.reply_to.content if body.reply_to else None)
    cached = response_cache.get(ck)
    if cached:
        cached["request_id"] = request_id
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
        try:
            web_text = _with_timeout(
                web_search_context,
                q,
                max_chars=WEB_MAX_CHARS,
                k=WEB_RESULT_K,
                timeout=WEB_TIMEOUT_SEC,
            ) or ""
        except FuturesTimeout:
            out = {"answer": "Recherche web trop longue. Réessaie ou passe en mode local.", "sources": [], "mode": "STRICT(web_live)", "ctx_len": 0}
            response_cache.set(ck, out)
            return finalize_result(_result(**out, request_id=request_id, route_mode=route_mode, validations=validations))
        except Exception:
            web_text = ""

        web_sources_list = _parse_web_links(web_text)
        if not web_text.strip():
            out = {"answer": "Je n’ai rien trouvé via la **recherche web en direct**.", "sources": [], "mode": "STRICT(web_live)", "ctx_len": 0}
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
            response_cache.set(ck, out)
            return finalize_result(_result(**out, request_id=request_id, route_mode=route_mode, validations=validations))

        if evidence_mode == "none":
            msg = "J’ai parcouru tes **sources**, mais rien de suffisamment pertinent."
            out = {"answer": msg, "sources": [], "mode": "STRICT(local)", "ctx_len": len(context_local)}
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
        trace("generation_input", {"generation_mode": mode_label, "question": q, "context_length": len(context_for_llm or ""), "context": context_for_llm, "provider": _RUNTIME_SETTINGS.generation.provider, "model": _RUNTIME_SETTINGS.generation.model, "reasoning_effort": _RUNTIME_SETTINGS.generation.reasoning_effort, "system_prompt": "constructed by rag_core.llm from generation mode and evidence mode", "evidence_mode": evidence_mode if use_strict else "none"})
        answer = _generate_answer(
            q, context_for_llm, history=hist, token_sink=token_sink,
            evidence_mode=evidence_mode if use_strict else "none",
        )
        trace("generation", {"generation_total_ms": round((time.perf_counter() - generation_started) * 1000, 1), "first_token_ms": None if token_sink is None else None})
        # Parse des citations éventuelles
        citations_idx = []
        try:
            m = re.search(r"<CITATIONS>\s*\[?([\d,\s]*)\]?\s*</CITATIONS>", answer, flags=re.IGNORECASE)
            if m:
                raw = m.group(1) or ""
                citations_idx = [int(x) for x in re.findall(r"\d+", raw)]
                answer = re.sub(r"\s*<CITATIONS>.*?</CITATIONS>\s*", " ", answer, flags=re.IGNORECASE).strip()
        except Exception:
            citations_idx = []
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
                    answer,
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
        sources = [{"path": s["url"], "chunk": -1} for s in web_sources_list]
    elif use_strict and blocks:
        # local strict
        if citations_idx:
            chosen = []
            for i in citations_idx:
                if 1 <= i <= len(blocks):
                    chosen.append(blocks[i - 1])
            chosen = chosen or (blocks or [])
        else:
            chosen = (blocks or [])
        tmp = [{"path": b[1]["path"], "chunk": b[1]["chunk_id"]} for b in chosen]
        seen, uniq = set(), []
        for s in tmp:
            p = s.get("path")
            if p and p not in seen:
                seen.add(p)
                uniq.append(s)
        sources = [{"path": s["path"], "chunk": -1} for s in uniq]
    else:
        sources = []

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
        "orchestration_use_history": orchestration_plan.use_history if orchestration_plan else None,
        "orchestration_reuse_previous_subject": orchestration_plan.reuse_previous_subject if orchestration_plan else None,
        "orchestration_ms": orchestration_ms,
        "evidence_mode": evidence_mode,
        "evidence_mode_reason": evidence_decision.reason,
        "context_is_relevant": evidence_decision.context_is_relevant,
        "context_relevance_reason": evidence_decision.context_relevance_reason,
        "morphological_topic_overlap": evidence_decision.morphological_topic_overlap,
        "strict_local_ok": strict_local_ok,
        "source_count": len(sources),
    })

    out = {
        "answer": answer,
        "sources": sources,
        "mode": mode_label,
        "ctx_len": len(context_for_llm or ""),
        "review": review.to_dict(),
        "faithfulness_review": claim_review.to_dict() if claim_review else None,
    }
    if not body.reply_to:
        response_cache.set(ck, out)
    return finalize_result(_result(
        **out,
        request_id=request_id,
        route_mode=route_mode,
        validations=validations,
    ))
