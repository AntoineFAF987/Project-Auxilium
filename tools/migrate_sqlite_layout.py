from __future__ import annotations

import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD_DB = ROOT / "Back" / "api" / "app.db"
SETTINGS_DB = ROOT / "Back" / "data" / "settings.db"
CHATS_DB = ROOT / "Back" / "api" / "chats.db"


def _copy_table(src: sqlite3.Connection, dst: sqlite3.Connection, table: str, columns: str) -> None:
    rows = src.execute(f"SELECT {columns} FROM {table}").fetchall()
    dst.execute(f"DELETE FROM {table}")
    if not rows:
        return
    placeholders = ",".join(["?"] * len(rows[0]))
    dst.executemany(
        f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
        rows,
    )


def migrate() -> None:
    if not OLD_DB.exists():
        raise SystemExit(f"Missing source database: {OLD_DB}")

    SETTINGS_DB.parent.mkdir(parents=True, exist_ok=True)

    old_conn = sqlite3.connect(OLD_DB)
    settings_conn = sqlite3.connect(SETTINGS_DB)

    settings_conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS email_folder_selection (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tenant_id    TEXT,
            user_id      TEXT,
            folders_json TEXT NOT NULL DEFAULT '[]',
            created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(tenant_id, user_id)
        );
        CREATE TABLE IF NOT EXISTS directory_selection (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tenant_id    TEXT,
            user_id      TEXT,
            dirs_json    TEXT NOT NULL DEFAULT '[]',
            created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(tenant_id, user_id)
        );
        CREATE TABLE IF NOT EXISTS allowed_extensions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tenant_id    TEXT,
            user_id      TEXT,
            exts_json    TEXT NOT NULL DEFAULT '[]',
            created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(tenant_id, user_id)
        );
        """
    )

    _copy_table(
        old_conn,
        settings_conn,
        "email_folder_selection",
        "tenant_id,user_id,folders_json,created_at,updated_at",
    )
    _copy_table(
        old_conn,
        settings_conn,
        "directory_selection",
        "tenant_id,user_id,dirs_json,created_at,updated_at",
    )
    _copy_table(
        old_conn,
        settings_conn,
        "allowed_extensions",
        "tenant_id,user_id,exts_json,created_at,updated_at",
    )
    settings_conn.commit()
    settings_conn.close()

    chat_tables = old_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('chats', 'chat_messages')"
    ).fetchall()
    if chat_tables:
        chats_conn = sqlite3.connect(CHATS_DB)
        chats_conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chats (
                id          TEXT PRIMARY KEY,
                tenant_id   TEXT,
                user_id     TEXT,
                title       TEXT NOT NULL DEFAULT 'Nouveau chat',
                deleted     INTEGER NOT NULL DEFAULT 0,
                created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_chats_owner
            ON chats(tenant_id, user_id, deleted);
            CREATE TABLE IF NOT EXISTS chat_messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id     TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                role        TEXT NOT NULL CHECK (role IN ('user','assistant','system')),
                content_enc TEXT NOT NULL,
                meta_json   TEXT NOT NULL DEFAULT '{}',
                created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_chat_messages_chat
            ON chat_messages(chat_id, created_at);
            """
        )
        _copy_table(
            old_conn,
            chats_conn,
            "chats",
            "id,tenant_id,user_id,title,deleted,created_at,updated_at",
        )
        _copy_table(
            old_conn,
            chats_conn,
            "chat_messages",
            "id,chat_id,role,content_enc,meta_json,created_at",
        )
        chats_conn.commit()
        chats_conn.close()

    old_conn.close()


if __name__ == "__main__":
    migrate()
