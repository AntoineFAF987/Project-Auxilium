# -*- coding: utf-8 -*-
"""Exports de compatibilité dérivés de la configuration runtime typée."""
import os
import torch
from runtime_settings import get_runtime_settings

# =========================
# Constantes exportées
# =========================
_SETTINGS = get_runtime_settings()
_RETRIEVAL = _SETTINGS.retrieval

EMBED_MODEL_NAME = _RETRIEVAL.embedding_model
RERANK_MODEL_NAME = _RETRIEVAL.reranker_model
RERANK_MODEL_FALLBACK = _RETRIEVAL.reranker_fallback
DEVICE = (
    ("cuda" if torch.cuda.is_available() else "cpu")
    if _RETRIEVAL.device == "auto"
    else _RETRIEVAL.device
)

TOP_K_FAISS = _RETRIEVAL.top_k_faiss
TOP_K_BM25 = _RETRIEVAL.top_k_bm25
RETRIEVE_K = _RETRIEVAL.retrieve_k
FINAL_K = _RETRIEVAL.final_k
HYBRID_ALPHA = _RETRIEVAL.hybrid_alpha
EXACT_MATCH_BONUS = _RETRIEVAL.exact_match_bonus
EXACT_MATCH_MIN_CHARS = _RETRIEVAL.exact_match_min_chars
MMR_LAMBDA = _RETRIEVAL.mmr_lambda
PRELIMINARY_POOL_MULTIPLIER = _RETRIEVAL.preliminary_pool_multiplier
NORMALIZE_EMBED = _RETRIEVAL.normalize_embeddings

MAX_CONTEXT_CHARS = _RETRIEVAL.max_context_chars
FUSE_ADJACENT_GAP = _RETRIEVAL.fuse_adjacent_gap
ANSWER_MIN_CE = _SETTINGS.thresholds.answerability

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
HISTORY_MAX_TURNS = _SETTINGS.conversation.history_max_turns
