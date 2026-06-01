# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from .crypto import decrypt_text, encrypt_text

CHAT_DB_PATH = Path(__file__).resolve().parent / "chats.db"


def get_connection() -> sqlite3.Connection:
    CHAT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CHAT_DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA busy_timeout=5000;")
    except Exception:
        pass
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chats (
                id          TEXT PRIMARY KEY,
                tenant_id   TEXT,
                user_id     TEXT,
                title       TEXT NOT NULL DEFAULT 'Nouveau chat',
                deleted     INTEGER NOT NULL DEFAULT 0,
                created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chats_owner
            ON chats(tenant_id, user_id, deleted);
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id     TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                role        TEXT NOT NULL CHECK (role IN ('user','assistant','system')),
                content_enc TEXT NOT NULL,
                meta_json   TEXT NOT NULL DEFAULT '{}',
                created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chat_messages_chat
            ON chat_messages(chat_id, created_at);
            """
        )
        conn.commit()


def _ensure_chat_owner(chat_id: str, tenant_id: str, user_id: str) -> bool:
    init_db()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id FROM chats WHERE id=? AND tenant_id=? AND user_id=? AND deleted=0",
            (chat_id, tenant_id, user_id),
        ).fetchone()
        return bool(row)


def create_chat(tenant_id: str, user_id: str, title: Optional[str] = None, *, chat_id: Optional[str] = None) -> str:
    init_db()
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
    init_db()
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
    init_db()
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
    init_db()
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
        raise PermissionError("Chat introuvable ou non autorise")
    enc = encrypt_text(content or "")
    payload = json.dumps(meta or {}, ensure_ascii=False)
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO chat_messages(chat_id, role, content_enc, meta_json)
            VALUES(?, ?, ?, ?)
            """,
            (chat_id, role, enc, payload),
        )
        conn.execute("UPDATE chats SET updated_at=CURRENT_TIMESTAMP WHERE id=?", (chat_id,))
        conn.commit()
        return int(cur.lastrowid)


def list_messages(tenant_id: str, user_id: str, chat_id: str, limit: int = 200) -> List[Dict]:
    if not _ensure_chat_owner(chat_id, tenant_id, user_id):
        raise PermissionError("Chat introuvable ou non autorise")
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
    for row in rows:
        out.append(
            {
                "id": row["id"],
                "role": row["role"],
                "content": decrypt_text(row["content_enc"]),
                "meta": json.loads(row["meta_json"] or "{}"),
                "created_at": row["created_at"],
            }
        )
    return out
