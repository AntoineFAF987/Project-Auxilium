# -*- coding: utf-8 -*-
import sqlite3
from pathlib import Path

# Base de configuration versionnable avec les données de démo.
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "settings.db"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=DELETE;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA busy_timeout=5000;")
    except Exception:
        pass
    return conn


def init_db() -> None:
    """Cree les tables de configuration persistante."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS email_folder_selection (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id    TEXT,
                user_id      TEXT,
                folders_json TEXT NOT NULL DEFAULT '[]',
                created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(tenant_id, user_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS directory_selection (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id    TEXT,
                user_id      TEXT,
                dirs_json    TEXT NOT NULL DEFAULT '[]',
                created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(tenant_id, user_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS allowed_extensions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id    TEXT,
                user_id      TEXT,
                exts_json    TEXT NOT NULL DEFAULT '[]',
                created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(tenant_id, user_id)
            )
            """
        )
        conn.commit()
