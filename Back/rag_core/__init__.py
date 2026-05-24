# -*- coding: utf-8 -*-
# Ré-export de l’API publique afin de conserver les mêmes imports qu’avant.
from .constants import (
    EMBED_MODEL_NAME, RERANK_MODEL_NAME, RERANK_MODEL_FALLBACK, DEVICE,
    TOP_K_FAISS, TOP_K_BM25, RETRIEVE_K, FINAL_K, HYBRID_ALPHA, NORMALIZE_EMBED,
    MAX_CONTEXT_CHARS, FUSE_ADJACENT_GAP, ANSWER_MIN_CE,
    ENABLE_WEB_SEARCH, TAVILY_API_KEY, HISTORY_MAX_TURNS,
)
from .utils import (
    FRENCH_STOPWORDS, _strip_accents, tokenize_for_bm25, keyword_overlap_count,
    ensure_dir, file_fingerprint, _is_under, _safe_walk, SUPPORTED_EXTS,
)
from .indexer import RAGIndexer
from .smalltalk import (
    _SMALLTALK_PROTOS, _prepare_smalltalk_cache, classify_smalltalk_semantic, is_smalltalk,
)
from .retrieval import (
    fuse_contiguous_passages, clip_context_blocks, format_context_for_llm,
    answerability_guard, trim_history,
)
from .web_live import web_search_context
from .llm import ask_mistral_with_context