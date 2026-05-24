# -*- coding: utf-8 -*-
"""
Persistence des extensions autorisées pour l’indexation locale.
Stockage : table allowed_extensions (1 ligne par (tenant_id, user_id)).
Valeur : exts_json = JSON array de strings, ex: [".pdf",".md",".py"]
"""

from __future__ import annotations
import json
from typing import List, Optional
from .db import get_connection, init_db

def _clean_ext(e: str) -> str:
    s = (e or "").strip().lower()
    if not s:
        return ""
    if not s.startswith("."):
        s = "." + s
    # interdites: chaînes trop longues, caractères spaces exotiques → on garde simple
    if len(s) > 16:
        return ""
    return s

def get_allowed_extensions(tenant_id: Optional[str] = None, user_id: Optional[str] = None) -> List[str]:
    """Retourne la liste nettoyée d'extensions autorisées (sans doublons)."""
    init_db()
    with get_connection() as conn:
        if tenant_id is None and user_id is None:
            row = conn.execute(
                "SELECT exts_json FROM allowed_extensions WHERE tenant_id IS NULL AND user_id IS NULL LIMIT 1"
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT exts_json FROM allowed_extensions WHERE tenant_id = ? AND user_id = ? LIMIT 1",
                (tenant_id, user_id),
            ).fetchone()
        if not row:
            return []  # par défaut: vide => on prendra le fallback côté index
        try:
            data = json.loads(row["exts_json"] or "[]")
        except Exception:
            data = []
        if not isinstance(data, list):
            return []
        seen = set()
        out: List[str] = []
        for e in data:
            ce = _clean_ext(str(e))
            if ce and ce not in seen:
                seen.add(ce)
                out.append(ce)
        return out

def set_allowed_extensions(exts: List[str], tenant_id: Optional[str] = None, user_id: Optional[str] = None) -> List[str]:
    """Remplace la sélection et renvoie la liste nettoyée effectivement sauvegardée."""
    init_db()
    seen = set()
    clean: List[str] = []
    for e in exts or []:
        ce = _clean_ext(str(e))
        if ce and ce not in seen:
            seen.add(ce)
            clean.append(ce)
    payload = json.dumps(clean, ensure_ascii=False)
    with get_connection() as conn:
        if tenant_id is None and user_id is None:
            conn.execute(
                """
                INSERT INTO allowed_extensions (tenant_id, user_id, exts_json)
                VALUES (NULL, NULL, ?)
                ON CONFLICT(tenant_id, user_id) DO UPDATE SET
                  exts_json=excluded.exts_json,
                  updated_at=CURRENT_TIMESTAMP
                """,
                (payload,),
            )
        else:
            conn.execute(
                """
                INSERT INTO allowed_extensions (tenant_id, user_id, exts_json)
                VALUES (?, ?, ?)
                ON CONFLICT(tenant_id, user_id) DO UPDATE SET
                  exts_json=excluded.exts_json,
                  updated_at=CURRENT_TIMESTAMP
                """,
                (tenant_id, user_id, payload),
            )
        conn.commit()
    return clean
