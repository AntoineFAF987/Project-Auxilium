# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from .crypto import decrypt_text, encrypt_text
from .diagnostics import json_safe

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
                title_generated INTEGER NOT NULL DEFAULT 0,
                title_is_manual INTEGER NOT NULL DEFAULT 0,
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
        # Migration retrocompatible : les conversations existantes restent hors projet.
        chat_columns = {row["name"] for row in conn.execute("PRAGMA table_info(chats)").fetchall()}
        if "project_id" not in chat_columns:
            conn.execute("ALTER TABLE chats ADD COLUMN project_id TEXT NULL")
        if "pinned" not in chat_columns:
            conn.execute("ALTER TABLE chats ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0")
        if "title_generated" not in chat_columns:
            conn.execute("ALTER TABLE chats ADD COLUMN title_generated INTEGER NOT NULL DEFAULT 0")
        if "title_is_manual" not in chat_columns:
            conn.execute("ALTER TABLE chats ADD COLUMN title_is_manual INTEGER NOT NULL DEFAULT 0")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id          TEXT PRIMARY KEY,
                tenant_id   TEXT NOT NULL,
                user_id     TEXT NOT NULL,
                name        TEXT NOT NULL,
                pinned      INTEGER NOT NULL DEFAULT 0,
                created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        project_columns = {row["name"] for row in conn.execute("PRAGMA table_info(projects)").fetchall()}
        if "pinned" not in project_columns:
            conn.execute("ALTER TABLE projects ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_projects_owner ON projects(tenant_id, user_id, updated_at);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chats_project_owner ON chats(tenant_id, user_id, project_id, deleted);")
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


def _project_belongs_to_user(conn: sqlite3.Connection, project_id: str, tenant_id: str, user_id: str) -> bool:
    return bool(conn.execute("SELECT id FROM projects WHERE id=? AND tenant_id=? AND user_id=?", (project_id, tenant_id, user_id)).fetchone())


def create_chat(tenant_id: str, user_id: str, title: Optional[str] = None, *, chat_id: Optional[str] = None, project_id: Optional[str] = None) -> str:
    init_db()
    cid = chat_id or str(uuid.uuid4())
    t = (title or "Nouveau chat").strip() or "Nouveau chat"
    with get_connection() as conn:
        if project_id and not _project_belongs_to_user(conn, project_id, tenant_id, user_id):
            raise ValueError("Projet introuvable ou non autorise")
        conn.execute(
            """
            INSERT OR IGNORE INTO chats(id, tenant_id, user_id, title, project_id)
            VALUES(?, ?, ?, ?, ?)
            """,
            (cid, tenant_id, user_id, t, project_id),
        )
        conn.commit()
    return cid


def rename_chat(tenant_id: str, user_id: str, chat_id: str, title: str) -> bool:
    init_db()
    with get_connection() as conn:
        cur = conn.execute(
            """
            UPDATE chats SET title=?, title_is_manual=1, updated_at=CURRENT_TIMESTAMP
            WHERE id=? AND tenant_id=? AND user_id=? AND deleted=0
            """,
            (title, chat_id, tenant_id, user_id),
        )
        conn.commit()
        return cur.rowcount > 0


def set_generated_chat_title(tenant_id: str, user_id: str, chat_id: str, title: str) -> bool:
    """Persist a generated title once, without ever replacing a manual title."""
    init_db()
    with get_connection() as conn:
        cur = conn.execute(
            """
            UPDATE chats SET title=?, title_generated=1, updated_at=CURRENT_TIMESTAMP
            WHERE id=? AND tenant_id=? AND user_id=? AND deleted=0
              AND title_is_manual=0 AND title_generated=0
            """,
            (title, chat_id, tenant_id, user_id),
        )
        conn.commit()
        return cur.rowcount > 0


def should_generate_chat_title(tenant_id: str, user_id: str, chat_id: str) -> bool:
    init_db()
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT 1 FROM chats
            WHERE id=? AND tenant_id=? AND user_id=? AND deleted=0
              AND title_is_manual=0 AND title_generated=0
            """,
            (chat_id, tenant_id, user_id),
        ).fetchone()
        return bool(row)


def mark_chat_title_generation_attempted(tenant_id: str, user_id: str, chat_id: str) -> None:
    """Avoid retrying a failed best-effort title generation on every later turn."""
    init_db()
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE chats SET title_generated=1
            WHERE id=? AND tenant_id=? AND user_id=? AND deleted=0
              AND title_is_manual=0 AND title_generated=0
            """,
            (chat_id, tenant_id, user_id),
        )
        conn.commit()


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
            SELECT id, title, project_id, pinned, created_at, updated_at
            FROM chats
            WHERE tenant_id=? AND user_id=? AND deleted=0
            ORDER BY updated_at DESC, created_at DESC
            """,
            (tenant_id, user_id),
        ).fetchall()
    return [dict(r) for r in rows]


def set_chat_project(tenant_id: str, user_id: str, chat_id: str, project_id: Optional[str]) -> bool:
    init_db()
    with get_connection() as conn:
        if project_id and not _project_belongs_to_user(conn, project_id, tenant_id, user_id):
            raise ValueError("Projet introuvable ou non autorise")
        cur = conn.execute("UPDATE chats SET project_id=?, updated_at=CURRENT_TIMESTAMP WHERE id=? AND tenant_id=? AND user_id=? AND deleted=0", (project_id, chat_id, tenant_id, user_id))
        conn.commit()
        return cur.rowcount > 0


def set_chat_pinned(tenant_id: str, user_id: str, chat_id: str, pinned: bool) -> bool:
    init_db()
    with get_connection() as conn:
        cur = conn.execute(
            "UPDATE chats SET pinned=?, updated_at=CURRENT_TIMESTAMP WHERE id=? AND tenant_id=? AND user_id=? AND deleted=0",
            (int(pinned), chat_id, tenant_id, user_id),
        )
        conn.commit()
        return cur.rowcount > 0


def list_projects(tenant_id: str, user_id: str) -> List[Dict]:
    init_db()
    with get_connection() as conn:
        rows = conn.execute("SELECT id, name, pinned, created_at, updated_at FROM projects WHERE tenant_id=? AND user_id=? ORDER BY updated_at DESC, created_at DESC", (tenant_id, user_id)).fetchall()
    return [dict(row) for row in rows]


def get_project(tenant_id: str, user_id: str, project_id: str) -> Optional[Dict]:
    init_db()
    with get_connection() as conn:
        row = conn.execute("SELECT id, name, pinned, created_at, updated_at FROM projects WHERE id=? AND tenant_id=? AND user_id=?", (project_id, tenant_id, user_id)).fetchone()
    return dict(row) if row else None


def create_project(tenant_id: str, user_id: str, name: str) -> Dict:
    init_db()
    project_id = str(uuid.uuid4())
    with get_connection() as conn:
        conn.execute("INSERT INTO projects(id, tenant_id, user_id, name) VALUES(?, ?, ?, ?)", (project_id, tenant_id, user_id, name))
        conn.commit()
    project = get_project(tenant_id, user_id, project_id)
    assert project is not None
    return project


def rename_project(tenant_id: str, user_id: str, project_id: str, name: str) -> Optional[Dict]:
    init_db()
    with get_connection() as conn:
        cur = conn.execute("UPDATE projects SET name=?, updated_at=CURRENT_TIMESTAMP WHERE id=? AND tenant_id=? AND user_id=?", (name, project_id, tenant_id, user_id))
        conn.commit()
    return get_project(tenant_id, user_id, project_id) if cur.rowcount else None


def set_project_pinned(tenant_id: str, user_id: str, project_id: str, pinned: bool) -> Optional[Dict]:
    init_db()
    with get_connection() as conn:
        cur = conn.execute(
            "UPDATE projects SET pinned=?, updated_at=CURRENT_TIMESTAMP WHERE id=? AND tenant_id=? AND user_id=?",
            (int(pinned), project_id, tenant_id, user_id),
        )
        conn.commit()
    return get_project(tenant_id, user_id, project_id) if cur.rowcount else None


def delete_project(tenant_id: str, user_id: str, project_id: str) -> bool:
    """Detach owned chats before removing their project; chats are never deleted."""
    init_db()
    with get_connection() as conn:
        if not _project_belongs_to_user(conn, project_id, tenant_id, user_id):
            return False
        conn.execute("UPDATE chats SET project_id=NULL, updated_at=CURRENT_TIMESTAMP WHERE tenant_id=? AND user_id=? AND project_id=? AND deleted=0", (tenant_id, user_id, project_id))
        conn.execute("DELETE FROM projects WHERE id=? AND tenant_id=? AND user_id=?", (project_id, tenant_id, user_id))
        conn.commit()
    return True


def append_message(tenant_id: str, user_id: str, chat_id: str, role: str, content: str, meta: Optional[Dict] = None) -> int:
    if not _ensure_chat_owner(chat_id, tenant_id, user_id):
        raise PermissionError("Chat introuvable ou non autorise")
    enc = encrypt_text(content or "")
    payload = json.dumps(json_safe(meta or {}), ensure_ascii=False)
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


def update_email_draft_artifact(tenant_id: str, user_id: str, chat_id: str, message_id: int, artifact_index: int, artifact: Dict) -> bool:
    """Persist one email artifact only; assistant prose and other artifacts stay immutable."""
    if not _ensure_chat_owner(chat_id, tenant_id, user_id):
        raise PermissionError("Chat introuvable ou non autorise")
    with get_connection() as conn:
        row = conn.execute("SELECT meta_json FROM chat_messages WHERE id=? AND chat_id=? AND role='assistant'", (message_id, chat_id)).fetchone()
        if not row:
            return False
        meta = json.loads(row["meta_json"] or "{}")
        artifacts = meta.get("artifacts")
        if not isinstance(artifacts, list) or artifact_index < 0 or artifact_index >= len(artifacts):
            return False
        if not isinstance(artifacts[artifact_index], dict) or artifacts[artifact_index].get("type") != "email_draft":
            return False
        artifacts[artifact_index] = artifact
        conn.execute("UPDATE chat_messages SET meta_json=? WHERE id=? AND chat_id=?", (json.dumps(json_safe(meta), ensure_ascii=False), message_id, chat_id))
        conn.commit()
    return True


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
