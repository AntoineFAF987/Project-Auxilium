# -*- coding: utf-8 -*-
"""
Constantes exportées — reprises strictement de rag_core.py.
"""
import os
import torch

# =========================
# Constantes exportées
# =========================
EMBED_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
# Reranker multilingue FR/EN pour meilleur classement en français
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
# Fallback si indisponible
RERANK_MODEL_FALLBACK = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOP_K_FAISS = 100
TOP_K_BM25 = 100
RETRIEVE_K = 12
FINAL_K = 10
HYBRID_ALPHA = 0.80
NORMALIZE_EMBED = True

MAX_CONTEXT_CHARS = 12000
FUSE_ADJACENT_GAP = 1
ANSWER_MIN_CE = -0.50

ENABLE_WEB_SEARCH = True
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
HISTORY_MAX_TURNS = 6