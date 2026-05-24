# -*- coding: utf-8 -*-
import time
from fastapi import APIRouter, HTTPException
from .config import index_dir
from .directories_db import get_selected_directories
from .extensions_db import get_allowed_extensions  # ✅
from .index_singleton import idx, idx_lock
from .sessions import SESSIONS

# pour modifier la constante dynamiquement
from rag_core import utils as rag_utils

router = APIRouter()
_start = time.time()

@router.get("/status")
def status():
    current_roots = []
    try:
        current_roots = getattr(idx, "roots", None) or []
    except Exception:
        current_roots = []
    return {
        "roots": current_roots,
        "index_dir": index_dir,
        "chunks": len(getattr(idx, "metas", []) or []),
        "sessions": len(SESSIONS),
        "uptime_sec": int(time.time() - _start),
        "needs_rebuild": bool(idx.needs_rebuild() if hasattr(idx, "needs_rebuild") else False),
    }

@router.post("/reindex")
def reindex():
    """
    Réindexation INCRÉMENTALE basée sur:
      1) Dossiers autorisés (Paramètres → Dossiers)
      2) Extensions autorisées (Paramètres → Extensions)
    """
    import os as _os

    try:
        selected = get_selected_directories() or []
    except Exception:
        selected = []

    roots: list[str] = []
    for d in selected:
        if not d.get("enabled", True):
            continue
        p = str(d.get("path") or "").strip()
        if p:
            roots.append(_os.path.abspath(p))

    # Extensions autorisées (peut être vide -> on garde un fallback minimal côté rag_core.utils)
    exts = get_allowed_extensions() or []
    if exts:
        # ✅ on force l’indexeur à ne considérer que ces extensions
        rag_utils.SUPPORTED_EXTS = set(exts)

    # Pas de fallback de roots: liste explicite des autorisations
    idx.roots = roots

    with idx_lock:
        idx.rebuild_incremental()

    return {
        "ok": True,
        "indexed_roots": roots,
        "extensions": list(rag_utils.SUPPORTED_EXTS),
        "note": "Incrémental avec extensions filtrées.",
    }
