# -*- coding: utf-8 -*-
import time
from fastapi import APIRouter, HTTPException, Request
from .config import index_dir, CONFIG_PATH
from .directories_db import get_selected_directories
from .email_db import get_selected_folders
from .extensions_db import get_allowed_extensions  # ✅
from .index_singleton import idx, idx_lock
from .routes_ingest import _recompute_roots_with_flat_emails
from .sessions import SESSIONS

try:
    from ingest_emails import ingest_emails, ingest_emails_with_access_token
except Exception:
    ingest_emails = None
    ingest_emails_with_access_token = None

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
        "index_schema_version": getattr(idx, "loaded_schema_version", 1),
        "index_generation": (getattr(idx, "index_manifest", {}) or {}).get("generation"),
        "index_summary": (getattr(idx, "index_manifest", {}) or {}).get("summary"),
    }

@router.post("/reindex")
def reindex(request: Request):
    """
    Réindexation INCRÉMENTALE basée sur:
      1) Dossiers autorisés (Paramètres → Dossiers)
      2) Extensions autorisées (Paramètres → Extensions)
    """
    auth = request.headers.get("Authorization") or ""
    bearer = auth.split(" ", 1)[1].strip() if auth.startswith("Bearer ") else None

    folders = get_selected_folders() or []
    if folders:
        if bearer and ingest_emails_with_access_token is not None:
            ingest_emails_with_access_token(str(CONFIG_PATH), bearer, override_folders=folders)
        elif ingest_emails is not None:
            # Non-browser callers use the persistent delegated cache. Device
            # flow is reached only after silent acquisition has been exhausted.
            ingest_emails(str(CONFIG_PATH), override_folders=folders)
        else:
            raise HTTPException(status_code=400, detail="Authentification email non disponible")

    roots = _recompute_roots_with_flat_emails()

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
        "index_schema_version": getattr(idx, "loaded_schema_version", 1),
        "index_generation": (getattr(idx, "index_manifest", {}) or {}).get("generation"),
        "index_summary": (getattr(idx, "index_manifest", {}) or {}).get("summary"),
    }
