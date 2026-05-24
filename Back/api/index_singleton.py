# -*- coding: utf-8 -*-
"""
Singleton d'index — version SANS réindex automatique au démarrage.
- On charge l'index s'il existe.
- Sinon, on initialise l'objet vide (et on attend un appel /reindex).
- Si la sélection "Dossiers" en DB est vide, on N'UTILISE PAS config.json en fallback ici non plus.
"""

import threading
from pathlib import Path as _P
from .config import index_dir, exclude_globs, follow_symlinks
from .directories_db import get_selected_directories
import os

from rag_core import (
    RAGIndexer,
    format_context_for_llm,
    fuse_contiguous_passages,
    clip_context_blocks,
    ask_mistral_with_context,
    answerability_guard,
    keyword_overlap_count,
    trim_history,
    classify_smalltalk_semantic,
    web_search_context,
    RETRIEVE_K, TOP_K_FAISS, HYBRID_ALPHA, FUSE_ADJACENT_GAP, MAX_CONTEXT_CHARS, FINAL_K,
)

# Détermine les roots initiaux UNIQUEMENT via la DB (pas de fallback auto)
try:
    _selected_dirs = get_selected_directories()
except Exception:
    _selected_dirs = []

_initial_roots: list[str] = []
for _item in _selected_dirs or []:
    if not isinstance(_item, dict):
        continue
    if not _item.get("enabled", True):
        continue
    _path = _item.get("path")
    if not _path:
        continue
    try:
        _abs = os.path.abspath(str(_path).strip())
        if _abs:
            _initial_roots.append(_abs)
    except Exception:
        continue

idx = RAGIndexer(_initial_roots, index_dir, exclude_globs, follow_symlinks)
idx_lock = threading.Lock()

# Pas de reconstruction auto. On charge seulement si des fichiers existent déjà.
if _P(idx.corpus_path).exists():
    _ = idx._load_existing()
# sinon: on attend un POST /reindex
