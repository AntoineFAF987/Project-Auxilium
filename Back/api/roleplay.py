# -*- coding: utf-8 -*-
import re
from typing import List, Optional, Dict

# --------- Heuristiques roleplay / factuel ---------
ROLEPLAY_ON_PATTERNS = [
    r"\b(role\s*play|roleplay)\b",
    r"\b(fais|faites)\s+semblant\b",
    r"\b(comme\s+si)\b",
    r"\b(joue(?:r)?\s+le\s+rôle|incarne(?:r)?)\b",
    r"\(fait\s+semblant\)",
]
ROLEPLAY_OFF_PATTERNS = [
    r"\b(arr(ê|e)te\s+de\s+faire\s+semblant)\b",
    r"\b(reprenons|repasse|rev(i|e)ens?)\s+(au|en)\s+(mode\s+)?(s(é|e)rieux|normal|strict)\b",
    r"\b(stop\s+role\s*play|fin\s+du\s+jeu)\b",
]
FACTUAL_HINTS = [
    r"\b(qui|quoi|qu'est-ce|quels?|quelle|comment|pourquoi|où|ou|quand|combien)\b",
    r"\b(define|what\s+is|who\s+is|how\s+to|when\s+did|where\s+is)\b",
    r"\b(diff(é|e)rence|proc(é|e)dure|processus|r(è|e)gles?|politique|politics?)\b",
]

def _hit_any(patterns: List[str], text: str) -> bool:
    s = (text or "").lower()
    return any(re.search(p, s, flags=re.IGNORECASE) for p in patterns)

def detect_roleplay_trigger(history_msgs: List[Dict], q: str) -> Optional[bool]:
    if _hit_any(ROLEPLAY_ON_PATTERNS, q):
        return True
    if _hit_any(ROLEPLAY_OFF_PATTERNS, q):
        return False
    for m in reversed(history_msgs[-8:]):
        if m.get("role") != "user":
            continue
        t = m.get("content") or ""
        if _hit_any(ROLEPLAY_OFF_PATTERNS, t):
            return False
        if _hit_any(ROLEPLAY_ON_PATTERNS, t):
            return True
    return None

def looks_factual(q: str) -> bool:
    if not q or len(q) > 220:
        return True
    if _hit_any(FACTUAL_HINTS, q):
        return True
    if _hit_any(ROLEPLAY_ON_PATTERNS, q):
        return False
    return False
