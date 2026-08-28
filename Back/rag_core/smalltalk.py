# -*- coding: utf-8 -*-
"""
Small talk — détection sémantique (embeddings) — inchangé.
"""
from typing import Dict, List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from .constants import DEVICE, EMBED_MODEL_NAME, NORMALIZE_EMBED

_SMALLTALK_PROTOS: Dict[str, List[str]] = {
    "greeting": ["bonjour", "salut", "bonsoir", "hello", "hi", "hey", "hola", "ola", "ciao", "👋"],
    "wellbeing_q": ["ça va ?", "comment ça va ?", "tu vas bien ?", "how are you?", "how's it going?"],
    "wellbeing_ok": ["ça va", "je vais bien", "nickel", "super", "au top", "i'm fine", "all good"],
    "insult": ["t'es con", "connard", "abruti", "idiot", "débile", "stupid", "dumb", "moron"],
    "chitchat": ["merci", "ok", "cool", "nice", "thanks", "parfait", "impeccable", "great"],
}
_SMALLTALK_CACHE = None  # {label: np.array[n_protos, d]}

def _prepare_smalltalk_cache(embed_model: SentenceTransformer):
    global _SMALLTALK_CACHE
    if _SMALLTALK_CACHE is not None:
        return _SMALLTALK_CACHE
    labels = {}
    for lab, phrases in _SMALLTALK_PROTOS.items():
        vecs = embed_model.encode(
            phrases,
            convert_to_tensor=False,
            normalize_embeddings=NORMALIZE_EMBED,
        )
        labels[lab] = np.array(vecs, dtype=np.float32)
    _SMALLTALK_CACHE = labels
    return _SMALLTALK_CACHE

def classify_smalltalk_semantic(q: str, embed_model: SentenceTransformer, threshold: float = 0.48) -> str:
    """
    Retourne: 'greeting' | 'wellbeing_q' | 'wellbeing_ok' | 'insult' | 'chitchat' | ''
    """
    if not q: return ""
    s = q.strip()
    if len(s) > 140:  # small talk = court
        return ""
    cache = _prepare_smalltalk_cache(embed_model)
    qv = embed_model.encode(
        [s],
        convert_to_tensor=False,
        normalize_embeddings=NORMALIZE_EMBED,
    )[0].astype(np.float32)
    best_label, best_sim = "", -1.0
    for lab, mat in cache.items():
        sim = float(np.max(mat @ qv))
        if sim > best_sim:
            best_sim, best_label = sim, lab
    return best_label if best_sim >= threshold else ""

def is_smalltalk(q: str, embed_model: Optional[SentenceTransformer] = None) -> bool:
    try:
        model = embed_model or SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)
        return classify_smalltalk_semantic(q, model) != ""
    except Exception:
        t = q.lower().strip()
        return len(t) <= 20 and any(tok in t for tok in ["salut", "bonjour", "hello", "hi", "hola", "👋"])
