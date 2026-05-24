# -*- coding: utf-8 -*-
import json
from typing import List, Optional
from .db import get_connection, init_db

def get_selected_folders(tenant_id: Optional[str] = None,
                         user_id: Optional[str] = None) -> List[str]:
    # garantit l’existence de la table
    init_db()
    with get_connection() as conn:
        if tenant_id is None and user_id is None:
            row = conn.execute(
                "SELECT folders_json FROM email_folder_selection "
                "WHERE tenant_id IS NULL AND user_id IS NULL LIMIT 1"
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT folders_json FROM email_folder_selection "
                "WHERE tenant_id = ? AND user_id = ? LIMIT 1",
                (tenant_id, user_id)
            ).fetchone()
        if not row:
            return []
        try:
            out = json.loads(row["folders_json"] or "[]")
            return out if isinstance(out, list) else []
        except Exception:
            return []

def set_selected_folders(folders: List[str],
                         tenant_id: Optional[str] = None,
                         user_id: Optional[str] = None) -> List[str]:
    init_db()
    # dédoublonnage & nettoyage
    clean = [str(x).strip() for x in (folders or []) if str(x).strip()]
    seen, uniq = set(), []
    for f in clean:
        if f not in seen:
            seen.add(f); uniq.append(f)
    payload = json.dumps(uniq, ensure_ascii=False)

    with get_connection() as conn:
        if tenant_id is None and user_id is None:
            row = conn.execute(
                "SELECT id FROM email_folder_selection "
                "WHERE tenant_id IS NULL AND user_id IS NULL LIMIT 1"
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE email_folder_selection "
                    "SET folders_json=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    (payload, row["id"])
                )
            else:
                conn.execute(
                    "INSERT INTO email_folder_selection (tenant_id, user_id, folders_json) "
                    "VALUES (NULL, NULL, ?)",
                    (payload,)
                )
        else:
            row = conn.execute(
                "SELECT id FROM email_folder_selection WHERE tenant_id=? AND user_id=? LIMIT 1",
                (tenant_id, user_id)
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE email_folder_selection "
                    "SET folders_json=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    (payload, row["id"])
                )
            else:
                conn.execute(
                    "INSERT INTO email_folder_selection (tenant_id, user_id, folders_json) "
                    "VALUES (?, ?, ?)",
                    (tenant_id, user_id, payload)
                )
        conn.commit()
    return uniq
