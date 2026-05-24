# -*- coding: utf-8 -*-
import sqlite3
from pathlib import Path

# DB stockée dans le dossier api/ (fichier: app.db)
DB_PATH = Path(__file__).resolve().parent / "app.db"

def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    # Enterprise-friendly SQLite settings
    try:
        conn.execute("PRAGMA journal_mode=WAL;")            # better concurrency
        conn.execute("PRAGMA synchronous=NORMAL;")          # durability vs speed balance
        conn.execute("PRAGMA foreign_keys=ON;")             # enforce FK constraints
        conn.execute("PRAGMA busy_timeout=5000;")           # wait up to 5s on locks
    except Exception:
        # non-fatal if PRAGMAs not supported
        pass
    return conn

def init_db() -> None:
    """Crée les tables nécessaires à la configuration persistante.

    Tables gérées ici :
      - email_folder_selection (sélection de dossiers mails)
      - directory_selection     (dossiers locaux autorisés)
      - allowed_extensions      (extensions autorisées pour l’indexation)
            - chats                   (conversations persistantes par utilisateur)
            - chat_messages           (messages chiffrés appartenant à une conversation)
    """
    with get_connection() as conn:
        # Table pour les sélections de dossiers e-mails (pré-existante)
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
        # Table pour la sélection des répertoires (dossiers locaux) autorisés
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
        # ✅ NOUVEAU : table des extensions autorisées
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS allowed_extensions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id    TEXT,
                user_id      TEXT,
                exts_json    TEXT NOT NULL DEFAULT '[]',  -- ex: [".pdf",".md",".py"]
                created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(tenant_id, user_id)
            )
            """
        )
        # ✅ Conversations (par utilisateur / tenant)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chats (
                id          TEXT PRIMARY KEY,             -- UUID string
                tenant_id   TEXT,
                user_id     TEXT,
                title       TEXT NOT NULL DEFAULT 'Nouveau chat',
                deleted     INTEGER NOT NULL DEFAULT 0,   -- soft delete
                created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chats_owner ON chats(tenant_id, user_id, deleted);
            """
        )
        # ✅ Messages (contenu chiffré au repos)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id     TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                role        TEXT NOT NULL CHECK (role IN ('user','assistant','system')),
                content_enc TEXT NOT NULL,               -- contenu chiffré (base64)
                meta_json   TEXT NOT NULL DEFAULT '{}',  -- métadonnées (non sensibles)
                created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chat_messages_chat ON chat_messages(chat_id, created_at);
            """
        )
        conn.commit()
