# -*- coding: utf-8 -*-
import re, json, uuid, hashlib
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

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
from .config import rag_params, timeouts
from rag_core.faithfulness import check_faithfulness, should_fallback_to_general
from rag_core.llm_stream import ask_mistral_with_context_stream
from .chats_db import create_chat, append_message
from auth_ms import verify_ms_token

# --- Anti-429: concurrence + retries ---
import threading, random, time

router = APIRouter()
EXEC = ThreadPoolExecutor(max_workers=8)

LLM_TIMEOUT_SEC = int(timeouts["llm_sec"])
WEB_TIMEOUT_SEC = int(timeouts["web_sec"])
MATH_TIMEOUT_SEC = int(timeouts["math_sec"])
ANS_THRESHOLD = float(rag_params.get("answerability_threshold", -0.5))
OVERLAP_MIN = int(rag_params.get("overlap_min", 1))
CONTEXT_RELEVANCE_THRESHOLD = float(rag_params.get("context_relevance_threshold", 0.3))
ENABLE_CONDENSATION = bool(rag_params.get("enable_query_condensation", True))
ENABLE_EXPANSION = bool(rag_params.get("enable_query_expansion", True))
ENABLE_FAITHFULNESS_CHECK = bool(rag_params.get("enable_faithfulness_check", True))
FAITHFULNESS_THRESHOLD = float(rag_params.get("faithfulness_threshold", 0.5))
FAITHFULNESS_STRICT_ONLY = bool(rag_params.get("faithfulness_strict_only", True))

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
               for p in (rag_params.get("fresh_news_keywords") or [])] \
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

@router.post("/ask", response_model=AskOut)
def ask(body: AskIn, request: Request):
    request_id = str(uuid.uuid4())
    q = (body.q or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Champ 'q' vide")

    # --- Historique & thread ---
    if body.reply_history:
        raw_hist: List[Dict] = [{"role": m.role, "content": m.content} for m in body.reply_history]
        hist = trim_history(raw_hist, max_turns=12)
    else:
        raw_hist: List[Dict] = [{"role": m.role, "content": m.content} for m in (body.history or [])]
        hist = trim_history(raw_hist, max_turns=6)

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
    trig = detect_roleplay_trigger(hist, q)
    if trig is True:
        _set_roleplay(thread_id, True)
    elif trig is False:
        _set_roleplay(thread_id, False)
    roleplay_active = _get_roleplay(thread_id)

    # Roleplay actif (et pas factuel) -> réponse courte roleplay
    if roleplay_active and (not looks_factual(q)):
        try:
            ans = _safe_llm(
                ask_mistral_with_context,
                q, context_text="", history=hist, roleplay_mode=True, max_tokens=220,
                timeout=LLM_TIMEOUT_SEC,
            )
            return AskOut(answer=ans, sources=[], request_id=request_id)
        except FuturesTimeout:
            raise HTTPException(status_code=504, detail="LLM timeout (roleplay)")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # --- Small talk rapide ---
    try:
        skind = classify_smalltalk_semantic(q, idx.embed_model)
    except Exception:
        skind = ""
    if skind:
        try:
            ans = _safe_llm(
                ask_mistral_with_context,
                q, context_text=reply_preamble, history=hist,
                smalltalk_mode=True, smalltalk_kind=skind, max_tokens=200,
                timeout=LLM_TIMEOUT_SEC,
            )
            return AskOut(answer=ans, sources=[], request_id=request_id)
        except FuturesTimeout:
            raise HTTPException(status_code=504, detail="LLM timeout (smalltalk)")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # --- Suivi "explique" après équation précédente ---
    if _is_explain_followup(q) and sess.get("last_math"):
        math_ans = _solve_math(sess["last_math"], detailed=True, timeout_sec=MATH_TIMEOUT_SEC)
        if math_ans:
            sess["no_context_once"] = True
            SESSIONS[thread_id] = sess
            return AskOut(answer=math_ans, sources=[], request_id=request_id)

    # --- Maths directes ---
    if _looks_like_equation(q):
        math_ans = _solve_math(q, detailed=True, timeout_sec=MATH_TIMEOUT_SEC)
        if math_ans:
            sess["last_math"] = q
            sess["no_context_once"] = True
            SESSIONS[thread_id] = sess
            return AskOut(answer=math_ans, sources=[], request_id=request_id)

    # --- Mode "general" forcé par l'utilisateur ---
    if mode_in == "general":
        try:
            ans = _safe_llm(ask_mistral_with_context, q, context_text=reply_preamble, history=hist, timeout=LLM_TIMEOUT_SEC)
            return AskOut(answer=ans, sources=[], mode="GENERAL(no-context)", ctx_len=len(reply_preamble), request_id=request_id)
        except FuturesTimeout:
            raise HTTPException(status_code=504, detail="LLM timeout")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # ===================== RAG (pré-recherche pour signaux) =====================
    # TIER 0 amélioration: condensation de question avec historique
    # Désactiver si historique trop court (risque de dérive) - compter messages utilisateur uniquement
    user_msg_count_for_condense = len([m for m in hist if m.get("role") == "user"])
    should_condense = (hist and user_msg_count_for_condense >= 2 and ENABLE_CONDENSATION)
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
    if is_vague_question and user_msg_count < 2:
        # Pas assez de contexte conversationnel pour ancrer la recherche => GENERAL direct
        _log_event(request_id, {"event": "vague_question_fallback", "q": q, "user_msg_count": user_msg_count})
        try:
            ans = _safe_llm(ask_mistral_with_context, q, context_text=reply_preamble, history=hist, timeout=LLM_TIMEOUT_SEC)
            return AskOut(answer=ans, sources=[], mode="GENERAL(vague-no-history)", ctx_len=len(reply_preamble), request_id=request_id)
        except FuturesTimeout:
            raise HTTPException(status_code=504, detail="LLM timeout")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM error: {e}")
    
    _log_event(request_id, {"event": "rag_search_start", "q_eff": q_eff, "should_condense": should_condense, "hist_len": len(hist)})
    
    prelim, ce_scores = idx.search(
        q_eff,
        retrieve_k=RETRIEVE_K,
        top_k_faiss=TOP_K_FAISS,
        hybrid_alpha=HYBRID_ALPHA,
        use_rerank=True,
        allowed_sources=allowed_sources,
    )
    prelim_hits = len(prelim)
    fused = fuse_contiguous_passages(prelim, gap=FUSE_ADJACENT_GAP)
    blocks = clip_context_blocks(fused, max_chars=MAX_CONTEXT_CHARS, keep=FINAL_K)

    gated_ok = answerability_guard(ce_scores, threshold=ANS_THRESHOLD)
    context_local = format_context_for_llm(blocks) if blocks else ""
    overlap = keyword_overlap_count(q, context_local)

    # --- NEW: Vérification de cohérence thématique du contexte ---
    context_is_relevant = _check_context_relevance(q, context_local)
    
    # --- Pertinence locale (PLUS souple): sémantique OU mots en commun ET cohérence ---
    strict_local_ok = bool(context_local) and context_is_relevant and (bool(gated_ok) or overlap >= OVERLAP_MIN)

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
                        use_rerank=True,
                        allowed_sources=allowed_sources,
                    )
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
            # Recalculer strict_local_ok avec le nouveau contexte fusionné
            strict_local_ok = bool(context_local) and context_is_relevant and (overlap >= OVERLAP_MIN)
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
    }

    # --- Choix du mode (règle déterministe actu ⇒ web_live, hors sujet ⇒ general) ---
    if mode_in in {"local", "web_index", "web_live"}:
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

    # ------ Cache court (intègre le route_mode) ------
    ck = _cache_key(q, mode_in, route_mode, body.reply_to.content if body.reply_to else None)
    cached = response_cache.get(ck)
    if cached:
        cached["request_id"] = request_id
        return AskOut(**cached)

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
            web_text = _with_timeout(web_search_context, q, max_chars=4000, k=5, timeout=WEB_TIMEOUT_SEC) or ""
        except FuturesTimeout:
            out = {"answer": "Recherche web trop longue. Réessaie ou passe en mode local.", "sources": [], "mode": "STRICT(web_live)", "ctx_len": 0}
            response_cache.set(ck, out)
            return AskOut(**out, request_id=request_id)
        except Exception:
            web_text = ""

        web_sources_list = _parse_web_links(web_text)
        if not web_text.strip():
            out = {"answer": "Je n’ai rien trouvé via la **recherche web en direct**.", "sources": [], "mode": "STRICT(web_live)", "ctx_len": 0}
            response_cache.set(ck, out)
            return AskOut(**out, request_id=request_id)

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
            return AskOut(**out, request_id=request_id)

        # double-check au cas où (pas obligatoire mais clair)
        if not strict_local_ok:
            msg = "J’ai parcouru tes **sources**, mais rien de suffisamment pertinent."
            out = {"answer": msg, "sources": [], "mode": "STRICT(local)", "ctx_len": len(context_local)}
            response_cache.set(ck, out)
            return AskOut(**out, request_id=request_id)

        context_for_llm = reply_preamble + context_local
        use_strict = True
        mode_label = "STRICT(local)"

    # C) general (par défaut) -> pas de contexte local

    # --- Sanity check : pas de contexte local en GENERAL ---
    if mode_label == "GENERAL(no-context)":
        assert context_for_llm == reply_preamble, "GENERAL ne doit pas embarquer de contexte local"

    # ===================== Appel LLM principal =====================
    try:
        answer = _safe_llm(
            ask_mistral_with_context,
            q, context_for_llm, history=hist,
            timeout=LLM_TIMEOUT_SEC
        )
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

    # ---------- NLI Faithfulness check (optionnel) ----------
    if ENABLE_FAITHFULNESS_CHECK:
        try:
            # Déterminer le contexte pour la vérification (local strict ou web strict)
            context_for_check = ""
            if route_mode == "strict_local":
                context_for_check = context_local or ""
            elif route_mode == "web_live":
                # web_text peut ne pas exister si pas de branche web; utiliser locals() pour vérifier
                context_for_check = locals().get("web_text", "") or ""

            # Si demandé en strict only, ne vérifier que quand on a un contexte strict
            should_check_here = (not FAITHFULNESS_STRICT_ONLY) or (route_mode in {"strict_local", "web_live"})
            if should_check_here and (context_for_check.strip()):
                fchk = check_faithfulness(answer, context_for_check, threshold=FAITHFULNESS_THRESHOLD)
                if should_fallback_to_general(fchk, strict_mode=True) and mode_in != "general":
                    _log_event(request_id, {
                        "event": "faithfulness_fallback",
                        "label": fchk.get("label"),
                        "score": fchk.get("score")
                    })
                    try:
                        alt = _safe_llm(
                            ask_mistral_with_context,
                            q, context_text=reply_preamble, history=hist,
                            timeout=LLM_TIMEOUT_SEC
                        )
                        if alt and alt.strip():
                            answer = alt
                            mode_label = "FALLBACK(faithfulness)"
                            use_strict = False
                            blocks = []
                            web_sources_list = []
                    except Exception:
                        pass
        except Exception:
            # Si le check échoue, on n'empêche pas la réponse
            pass

    # ---------- Filet "je ne sais pas" ⇒ tenter web si question d'actu ----------
    if _looks_fresh_news(q) and ("je ne sais pas" in answer.lower()):
        try:
            web_text = _with_timeout(web_search_context, q, max_chars=4000, k=5, timeout=WEB_TIMEOUT_SEC) or ""
            web_sources_list = _parse_web_links(web_text)
            if web_text.strip():
                context_for_llm = f"{reply_preamble}{web_text}"
                use_strict = True
                answer = _safe_llm(ask_mistral_with_context, q, context_for_llm, history=hist, timeout=LLM_TIMEOUT_SEC)
                mode_label = "STRICT(web_live)"
                blocks = []
        except Exception:
            pass  # on laisse la réponse telle quelle si le web échoue

    # ---------- NOUVEAU: Filet "je ne sais pas" en mode AUTO ⇒ retenter GENERAL ----------
    if (mode_in == "auto") and ("je ne sais pas" in (answer or "").lower()):
        try:
            alt = _safe_llm(
                ask_mistral_with_context,
                q, context_text=reply_preamble, history=hist,
                timeout=LLM_TIMEOUT_SEC
            )
            if alt and alt.strip():
                answer = alt
                mode_label = "FALLBACK(general)"
                use_strict = False
                blocks = []
                web_sources_list = []
        except Exception:
            # si l'appel échoue, on garde la réponse initiale
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

    # ------------------- Vérification post-réponse (léger) -------------------
    def _verify_answer(q: str, answer: str, strict: bool, sources_count: int) -> str:
        prompt = (
            "VERIFIEUR JSON:\n"
            "Réponds seulement JSON {\"ok\":true|false,\"action\":\"none|ask_web\"}.\n"
            "- Si strict=true et sources_count==0 => ask_web.\n"
            "- Si la question implique de l'actualité (aujourd'hui, en ce moment, 2024+) sans sources récentes => ask_web.\n"
            "- Sinon ok=true."
            f"\nQ:{q}\nstrict:{strict}\nsources_count:{sources_count}\nANSWER:\n{answer}"
        )
        try:
            raw = _safe_llm(ask_mistral_with_context, prompt, context_text="", history=[], timeout=min(LLM_TIMEOUT_SEC, 8))
            data = json.loads(raw.strip())
            return data.get("action", "none") or "none"
        except Exception:
            return "none"

    action = _verify_answer(q, answer, use_strict, len(sources))
    if action == "ask_web" and route_mode != "web_live":
        return AskOut(
            answer="Je préfère vérifier sur des sources à jour. Je lance une recherche web ?",
            sources=[],
            mode=mode_label,
            ctx_len=len(context_for_llm or ""),
            request_id=request_id,
        )

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
        "overlap": overlap,
        "guard_ok": bool(gated_ok),
    })

    out = {
        "answer": answer,
        "sources": sources,
        "mode": mode_label,
        "ctx_len": len(context_for_llm or "")
    }
    if not body.reply_to:
        response_cache.set(ck, out)
    # ------ Persistance (optionnelle si Authorization présent) ------
    tenant_id, user_id = _try_get_auth_ids(request)
    chat_id = (body.thread_id or "").strip()
    if tenant_id and user_id:
        try:
            # Crée la conversation si besoin
            title = (q[:60] + "…") if len(q) > 60 else q
            chat_id = chat_id or str(uuid.uuid4())
            create_chat(tenant_id, user_id, title=title or "Nouveau chat", chat_id=chat_id)
            # Ajoute les messages (user puis assistant)
            append_message(tenant_id, user_id, chat_id, "user", q, meta={"request_id": request_id})
            append_message(tenant_id, user_id, chat_id, "assistant", answer, meta={"mode": mode_label})
        except Exception:
            # Ne bloque pas la réponse si la persistance échoue
            pass
    return AskOut(**out, request_id=request_id, chat_id=(chat_id or None))


# ==================== STREAMING ENDPOINT ====================
@router.post("/ask/stream")
async def ask_stream(body: AskIn, request: Request):
    """
    Endpoint de streaming pour affichage progressif des réponses.
    Retourne un flux SSE (Server-Sent Events).
    """
    request_id = str(uuid.uuid4())
    q = (body.q or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Champ 'q' vide")

    # --- Historique & thread ---
    if body.reply_history:
        raw_hist: List[Dict] = [{"role": m.role, "content": m.content} for m in body.reply_history]
        hist = trim_history(raw_hist, max_turns=12)
    else:
        raw_hist: List[Dict] = [{"role": m.role, "content": m.content} for m in (body.history or [])]
        hist = trim_history(raw_hist, max_turns=6)

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
    trig = detect_roleplay_trigger(hist, q)
    if trig is True:
        _set_roleplay(thread_id, True)
    elif trig is False:
        _set_roleplay(thread_id, False)
    roleplay_active = _get_roleplay(thread_id)

    # Fonction génératrice pour le streaming
    def generate_stream():
        # Roleplay actif (et pas factuel) -> réponse courte roleplay
        if roleplay_active and (not looks_factual(q)):
            full_roleplay_answer = ""
            try:
                for chunk in ask_mistral_with_context_stream(
                    q, context_text="", history=hist, roleplay_mode=True, max_tokens=220
                ):
                    full_roleplay_answer += chunk
                    yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
                # Persistance AVANT le done
                tenant_id, user_id = _try_get_auth_ids(request)
                chat_id = (body.thread_id or "").strip()
                if tenant_id and user_id:
                    try:
                        title = (q[:60] + "…") if len(q) > 60 else q
                        chat_id = chat_id or str(uuid.uuid4())
                        create_chat(tenant_id, user_id, title=title or "Nouveau chat", chat_id=chat_id)
                        append_message(tenant_id, user_id, chat_id, "user", q, meta={"stream": True, "mode": "roleplay"})
                        append_message(tenant_id, user_id, chat_id, "assistant", full_roleplay_answer, meta={"mode": "roleplay", "stream": True})
                    except Exception:
                        pass
                yield f"data: {json.dumps({'type': 'done', 'sources': [], 'mode': 'roleplay', 'chat_id': (chat_id or None)})}\n\n"
                return
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
                return

        # --- Small talk rapide ---
        try:
            skind = classify_smalltalk_semantic(q, idx.embed_model)
        except Exception:
            skind = ""
        if skind:
            full_smalltalk_answer = ""
            try:
                for chunk in ask_mistral_with_context_stream(
                    q, context_text=reply_preamble, history=hist,
                    smalltalk_mode=True, smalltalk_kind=skind, max_tokens=200
                ):
                    full_smalltalk_answer += chunk
                    yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
                # Persistance AVANT le done
                tenant_id, user_id = _try_get_auth_ids(request)
                chat_id = (body.thread_id or "").strip()
                if tenant_id and user_id:
                    try:
                        title = (q[:60] + "…") if len(q) > 60 else q
                        chat_id = chat_id or str(uuid.uuid4())
                        create_chat(tenant_id, user_id, title=title or "Nouveau chat", chat_id=chat_id)
                        append_message(tenant_id, user_id, chat_id, "user", q, meta={"stream": True, "mode": "smalltalk"})
                        append_message(tenant_id, user_id, chat_id, "assistant", full_smalltalk_answer, meta={"mode": "smalltalk", "stream": True})
                    except Exception:
                        pass
                yield f"data: {json.dumps({'type': 'done', 'sources': [], 'mode': 'smalltalk', 'chat_id': (chat_id or None)})}\n\n"
                return
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
                return

        # --- Suivi "explique" après équation précédente ---
        if _is_explain_followup(q) and sess.get("last_math"):
            math_ans = _solve_math(sess["last_math"], detailed=True, timeout_sec=MATH_TIMEOUT_SEC)
            if math_ans:
                sess["no_context_once"] = True
                SESSIONS[thread_id] = sess
                yield f"data: {json.dumps({'type': 'content', 'content': math_ans})}\n\n"
                # Persistance AVANT le done
                tenant_id, user_id = _try_get_auth_ids(request)
                chat_id = (body.thread_id or "").strip()
                if tenant_id and user_id:
                    try:
                        title = (q[:60] + "…") if len(q) > 60 else q
                        chat_id = chat_id or str(uuid.uuid4())
                        create_chat(tenant_id, user_id, title=title or "Nouveau chat", chat_id=chat_id)
                        append_message(tenant_id, user_id, chat_id, "user", q, meta={"stream": True, "mode": "math"})
                        append_message(tenant_id, user_id, chat_id, "assistant", math_ans, meta={"mode": "math", "stream": True})
                    except Exception:
                        pass
                yield f"data: {json.dumps({'type': 'done', 'sources': [], 'mode': 'math', 'chat_id': (chat_id or None)})}\n\n"
                return

        # --- Maths directes ---
        if _looks_like_equation(q):
            math_ans = _solve_math(q, detailed=True, timeout_sec=MATH_TIMEOUT_SEC)
            if math_ans:
                sess["last_math"] = q
                sess["no_context_once"] = True
                SESSIONS[thread_id] = sess
                yield f"data: {json.dumps({'type': 'content', 'content': math_ans})}\n\n"
                # Persistance AVANT le done
                tenant_id, user_id = _try_get_auth_ids(request)
                chat_id = (body.thread_id or "").strip()
                if tenant_id and user_id:
                    try:
                        title = (q[:60] + "…") if len(q) > 60 else q
                        chat_id = chat_id or str(uuid.uuid4())
                        create_chat(tenant_id, user_id, title=title or "Nouveau chat", chat_id=chat_id)
                        append_message(tenant_id, user_id, chat_id, "user", q, meta={"stream": True, "mode": "math"})
                        append_message(tenant_id, user_id, chat_id, "assistant", math_ans, meta={"mode": "math", "stream": True})
                    except Exception:
                        pass
                yield f"data: {json.dumps({'type': 'done', 'sources': [], 'mode': 'math', 'chat_id': (chat_id or None)})}\n\n"
                return

        # --- Mode "general" forcé par l'utilisateur ---
        if mode_in == "general":
            full_general_answer = ""
            try:
                for chunk in ask_mistral_with_context_stream(q, context_text=reply_preamble, history=hist):
                    full_general_answer += chunk
                    yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
                # Persistance AVANT le done
                tenant_id, user_id = _try_get_auth_ids(request)
                chat_id = (body.thread_id or "").strip()
                if tenant_id and user_id:
                    try:
                        title = (q[:60] + "…") if len(q) > 60 else q
                        chat_id = chat_id or str(uuid.uuid4())
                        create_chat(tenant_id, user_id, title=title or "Nouveau chat", chat_id=chat_id)
                        append_message(tenant_id, user_id, chat_id, "user", q, meta={"stream": True, "mode": "general"})
                        append_message(tenant_id, user_id, chat_id, "assistant", full_general_answer, meta={"mode": "GENERAL(no-context)", "stream": True})
                    except Exception:
                        pass
                yield f"data: {json.dumps({'type': 'done', 'sources': [], 'mode': 'GENERAL(no-context)', 'chat_id': (chat_id or None)})}\n\n"
                return
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
                return

        # ===================== RAG (pré-recherche pour signaux) =====================
        user_msg_count_for_condense = len([m for m in hist if m.get("role") == "user"])
        should_condense = (hist and user_msg_count_for_condense >= 2 and ENABLE_CONDENSATION)
        q_eff = _condense_question(hist, q) if should_condense else q
        
        q_words = [w for w in q.strip().split() if len(w) > 2]
        is_vague_question = (
            len(q_words) <= 5 and
            any(w in q.lower() for w in ["comment", "pourquoi", "quoi", "how", "why", "what", "explain", "explique", "c'est quoi", "ça marche"])
        )
        user_msg_count = len([m for m in hist if m.get("role") == "user"])
        if is_vague_question and user_msg_count < 2:
            try:
                for chunk in ask_mistral_with_context_stream(q, context_text=reply_preamble, history=hist):
                    yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'sources': [], 'mode': 'GENERAL(vague-no-history)'})}\n\n"
                return
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
                return
        
        prelim, ce_scores = idx.search(
            q_eff,
            retrieve_k=RETRIEVE_K,
            top_k_faiss=TOP_K_FAISS,
            hybrid_alpha=HYBRID_ALPHA,
            use_rerank=True,
            allowed_sources=allowed_sources,
        )
        prelim_hits = len(prelim)
        fused = fuse_contiguous_passages(prelim, gap=FUSE_ADJACENT_GAP)
        blocks = clip_context_blocks(fused, max_chars=MAX_CONTEXT_CHARS, keep=FINAL_K)

        gated_ok = answerability_guard(ce_scores, threshold=ANS_THRESHOLD)
        context_local = format_context_for_llm(blocks) if blocks else ""
        overlap = keyword_overlap_count(q, context_local)

        context_is_relevant = _check_context_relevance(q, context_local)
        strict_local_ok = bool(context_local) and context_is_relevant and (bool(gated_ok) or overlap >= OVERLAP_MIN)

        if ENABLE_EXPANSION and (not strict_local_ok or not context_is_relevant) and prelim_hits < max(4, RETRIEVE_K // 2):
            try:
                variants = _multi_query_expand(q_eff, n=3)
                all_prelims = [prelim]
                for vq in variants[1:]:
                    try:
                        p2, _ = idx.search(
                            vq,
                            retrieve_k=RETRIEVE_K,
                            top_k_faiss=TOP_K_FAISS,
                            hybrid_alpha=HYBRID_ALPHA,
                            use_rerank=True,
                            allowed_sources=allowed_sources,
                        )
                        all_prelims.append(p2)
                    except Exception:
                        continue
                merged = _merge_prelims(all_prelims)
                fused = fuse_contiguous_passages(merged, gap=FUSE_ADJACENT_GAP)
                blocks = clip_context_blocks(fused, max_chars=MAX_CONTEXT_CHARS, keep=FINAL_K)
                context_local = format_context_for_llm(blocks) if blocks else ""
                overlap = keyword_overlap_count(q, context_local)
                context_is_relevant = _check_context_relevance(q, context_local)
                strict_local_ok = bool(context_local) and context_is_relevant and (overlap >= OVERLAP_MIN)
            except Exception:
                pass

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
        }

        if mode_in in {"local", "web_index", "web_live"}:
            route_mode = {
                "local": "strict_local",
                "web_index": "strict_local",
                "web_live": "web_live"
            }[mode_in]
        else:
            if not context_is_relevant and prelim_hits < max(3, RETRIEVE_K // 3):
                route_mode = "general"
            elif _looks_fresh_news(q):
                route_mode = "web_live"
            else:
                route_mode = _llm_route(q, signals, hist)

        if route_mode == "general" and strict_local_ok and mode_in != "web_live":
            route_mode = "strict_local"

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
                context_for_llm = reply_preamble
                use_strict = True
            else:
                context_for_llm = reply_preamble
                use_strict = False

        web_sources_list: List[Dict[str, str]] = []
        mode_label = "GENERAL(no-context)"

        # A) web_live
        if route_mode == "web_live":
            try:
                web_text = _with_timeout(web_search_context, q, max_chars=4000, k=5, timeout=WEB_TIMEOUT_SEC) or ""
            except FuturesTimeout:
                error_msg = 'Recherche web trop longue. Réessaie ou passe en mode local.'
                yield f"data: {json.dumps({'type': 'content', 'content': error_msg})}\n\n"
                # Persistance AVANT le done
                tenant_id, user_id = _try_get_auth_ids(request)
                chat_id = (body.thread_id or "").strip()
                if tenant_id and user_id:
                    try:
                        title = (q[:60] + "…") if len(q) > 60 else q
                        chat_id = chat_id or str(uuid.uuid4())
                        create_chat(tenant_id, user_id, title=title or "Nouveau chat", chat_id=chat_id)
                        append_message(tenant_id, user_id, chat_id, "user", q, meta={"stream": True, "mode": "web_live"})
                        append_message(tenant_id, user_id, chat_id, "assistant", error_msg, meta={"mode": "STRICT(web_live)", "stream": True, "error": True})
                    except Exception:
                        pass
                yield f"data: {json.dumps({'type': 'done', 'sources': [], 'mode': 'STRICT(web_live)', 'chat_id': (chat_id or None)})}\n\n"
                return
            except Exception:
                web_text = ""

            web_sources_list = _parse_web_links(web_text)
            if not web_text.strip():
                msg = "Je n'ai rien trouvé via la **recherche web en direct**."
                yield f"data: {json.dumps({'type': 'content', 'content': msg})}\n\n"
                # Persistance AVANT le done
                tenant_id, user_id = _try_get_auth_ids(request)
                chat_id = (body.thread_id or "").strip()
                if tenant_id and user_id:
                    try:
                        title = (q[:60] + "…") if len(q) > 60 else q
                        chat_id = chat_id or str(uuid.uuid4())
                        create_chat(tenant_id, user_id, title=title or "Nouveau chat", chat_id=chat_id)
                        append_message(tenant_id, user_id, chat_id, "user", q, meta={"stream": True, "mode": "web_live"})
                        append_message(tenant_id, user_id, chat_id, "assistant", msg, meta={"mode": "STRICT(web_live)", "stream": True})
                    except Exception:
                        pass
                yield f"data: {json.dumps({'type': 'done', 'sources': [], 'mode': 'STRICT(web_live)', 'chat_id': (chat_id or None)})}\n\n"
                return

            context_for_llm = f"{reply_preamble}{web_text}"
            use_strict = True
            blocks = []
            mode_label = "STRICT(web_live)"

        # B) strict_local
        elif route_mode == "strict_local":
            if not blocks:
                msg = "Je n'ai rien trouvé de pertinent dans les **sources autorisées**."
                yield f"data: {json.dumps({'type': 'content', 'content': msg})}\n\n"
                # Persistance AVANT le done
                tenant_id, user_id = _try_get_auth_ids(request)
                chat_id = (body.thread_id or "").strip()
                if tenant_id and user_id:
                    try:
                        title = (q[:60] + "…") if len(q) > 60 else q
                        chat_id = chat_id or str(uuid.uuid4())
                        create_chat(tenant_id, user_id, title=title or "Nouveau chat", chat_id=chat_id)
                        append_message(tenant_id, user_id, chat_id, "user", q, meta={"stream": True, "mode": "strict_local"})
                        append_message(tenant_id, user_id, chat_id, "assistant", msg, meta={"mode": "STRICT(local)", "stream": True})
                    except Exception:
                        pass
                yield f"data: {json.dumps({'type': 'done', 'sources': [], 'mode': 'STRICT(local)', 'chat_id': (chat_id or None)})}\n\n"
                return

            if not strict_local_ok:
                msg = "J'ai parcouru tes **sources**, mais rien de suffisamment pertinent."
                yield f"data: {json.dumps({'type': 'content', 'content': msg})}\n\n"
                # Persistance AVANT le done
                tenant_id, user_id = _try_get_auth_ids(request)
                chat_id = (body.thread_id or "").strip()
                if tenant_id and user_id:
                    try:
                        title = (q[:60] + "…") if len(q) > 60 else q
                        chat_id = chat_id or str(uuid.uuid4())
                        create_chat(tenant_id, user_id, title=title or "Nouveau chat", chat_id=chat_id)
                        append_message(tenant_id, user_id, chat_id, "user", q, meta={"stream": True, "mode": "strict_local"})
                        append_message(tenant_id, user_id, chat_id, "assistant", msg, meta={"mode": "STRICT(local)", "stream": True})
                    except Exception:
                        pass
                yield f"data: {json.dumps({'type': 'done', 'sources': [], 'mode': 'STRICT(local)', 'chat_id': (chat_id or None)})}\n\n"
                return

            context_for_llm = reply_preamble + context_local
            use_strict = True
            mode_label = "STRICT(local)"

        # ===================== Appel LLM principal avec streaming =====================
        full_answer = ""
        try:
            for chunk in ask_mistral_with_context_stream(
                q, context_for_llm, history=hist
            ):
                full_answer += chunk
                yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
            
            # Parse des citations éventuelles
            citations_idx = []
            try:
                m = re.search(r"<CITATIONS>\s*\[?([\d,\s]*)\]?\s*</CITATIONS>", full_answer, flags=re.IGNORECASE)
                if m:
                    raw = m.group(1) or ""
                    citations_idx = [int(x) for x in re.findall(r"\d+", raw)]
            except Exception:
                citations_idx = []

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
            return

        # ===================== Sources à renvoyer =====================
        if use_strict and context_for_llm and web_sources_list:
            sources = [{"path": s["url"], "chunk": -1} for s in web_sources_list]
        elif use_strict and blocks:
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

        # ------ Persistance (optionnelle si Authorization présent) ------
        tenant_id, user_id = _try_get_auth_ids(request)
        chat_id = (body.thread_id or "").strip()
        if tenant_id and user_id:
            try:
                title = (q[:60] + "…") if len(q) > 60 else q
                chat_id = chat_id or str(uuid.uuid4())
                create_chat(tenant_id, user_id, title=title or "Nouveau chat", chat_id=chat_id)
                append_message(tenant_id, user_id, chat_id, "user", q, meta={"stream": True})
                append_message(tenant_id, user_id, chat_id, "assistant", full_answer, meta={"mode": mode_label, "stream": True})
            except Exception:
                pass

        # Envoyer les métadonnées finales
        yield f"data: {json.dumps({'type': 'done', 'sources': sources, 'mode': mode_label, 'chat_id': (chat_id or None)})}\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")
