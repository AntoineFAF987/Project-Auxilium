# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import uuid
from typing import List, Dict, Optional, Tuple
from .db import get_connection
from .crypto import encrypt_text, decrypt_text

# --------- Data Access for chats/messages ---------

def _ensure_chat_owner(chat_id: str, tenant_id: str, user_id: str) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id FROM chats WHERE id=? AND tenant_id=? AND user_id=? AND deleted=0",
            (chat_id, tenant_id, user_id),
        ).fetchone()
        return bool(row)


def create_chat(tenant_id: str, user_id: str, title: Optional[str] = None, *, chat_id: Optional[str] = None) -> str:
    cid = chat_id or str(uuid.uuid4())
    t = (title or "Nouveau chat").strip() or "Nouveau chat"
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO chats(id, tenant_id, user_id, title)
            VALUES(?, ?, ?, ?)
            """,
            (cid, tenant_id, user_id, t),
        )
        conn.commit()
    return cid


def rename_chat(tenant_id: str, user_id: str, chat_id: str, title: str) -> bool:
    with get_connection() as conn:
        cur = conn.execute(
            """
            UPDATE chats SET title=?, updated_at=CURRENT_TIMESTAMP
            WHERE id=? AND tenant_id=? AND user_id=? AND deleted=0
            """,
            (title, chat_id, tenant_id, user_id),
        )
        conn.commit()
        return cur.rowcount > 0


def soft_delete_chat(tenant_id: str, user_id: str, chat_id: str) -> bool:
    with get_connection() as conn:
        cur = conn.execute(
            """
            UPDATE chats SET deleted=1, updated_at=CURRENT_TIMESTAMP
            WHERE id=? AND tenant_id=? AND user_id=? AND deleted=0
            """,
            (chat_id, tenant_id, user_id),
        )
        conn.commit()
        return cur.rowcount > 0


def list_chats(tenant_id: str, user_id: str) -> List[Dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM chats
            WHERE tenant_id=? AND user_id=? AND deleted=0
            ORDER BY updated_at DESC, created_at DESC
            """,
            (tenant_id, user_id),
        ).fetchall()
    return [dict(r) for r in rows]


def append_message(tenant_id: str, user_id: str, chat_id: str, role: str, content: str, meta: Optional[Dict] = None) -> int:
    if not _ensure_chat_owner(chat_id, tenant_id, user_id):
        raise PermissionError("Chat introuvable ou non autorisé")
    enc = encrypt_text(content or "")
    m = json.dumps(meta or {}, ensure_ascii=False)
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO chat_messages(chat_id, role, content_enc, meta_json)
            VALUES(?, ?, ?, ?)
            """,
            (chat_id, role, enc, m),
        )
        conn.execute("UPDATE chats SET updated_at=CURRENT_TIMESTAMP WHERE id=?", (chat_id,))
        conn.commit()
        return int(cur.lastrowid)


def list_messages(tenant_id: str, user_id: str, chat_id: str, limit: int = 200) -> List[Dict]:
    if not _ensure_chat_owner(chat_id, tenant_id, user_id):
        raise PermissionError("Chat introuvable ou non autorisé")
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, role, content_enc, meta_json, created_at
            FROM chat_messages
            WHERE chat_id=?
            ORDER BY id ASC
            LIMIT ?
            """,
            (chat_id, int(limit)),
        ).fetchall()
    out: List[Dict] = []
    for r in rows:
        out.append({
            "id": r["id"],
            "role": r["role"],
            "content": decrypt_text(r["content_enc"]),
            "meta": json.loads(r["meta_json"] or "{}"),
            "created_at": r["created_at"],
        })
    return out
