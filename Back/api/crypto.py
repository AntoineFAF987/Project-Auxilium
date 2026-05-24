# -*- coding: utf-8 -*-
"""
Symmetric encryption utilities for at-rest protection of chat messages.
- Uses Fernet (AES-128-CBC + HMAC, time-stamped tokens) from cryptography.
- Key is sourced from env CHAT_ENC_KEY when provided.
- Otherwise, a per-installation key is created under api/chat_secret.key.

NOTE: For production, prefer providing CHAT_ENC_KEY via environment or a
secrets manager. Rotating keys requires re-encryption, not implemented here.
"""
import os
from pathlib import Path
from typing import Optional
from cryptography.fernet import Fernet, InvalidToken

_KEY_ENV = "CHAT_ENC_KEY"
_KEY_PATH = Path(__file__).resolve().parent / "chat_secret.key"

_cached: Optional[Fernet] = None

def _load_or_create_key() -> bytes:
    env = os.getenv(_KEY_ENV, "").strip()
    if env:
        # Expect urlsafe base64 32-byte key (as produced by Fernet.generate_key())
        try:
            return env.encode("utf-8")
        except Exception:
            raise RuntimeError("CHAT_ENC_KEY invalide: fournis une clé Fernet valide (urlsafe base64).")
    # Fallback: file-based key for on-prem install
    if _KEY_PATH.exists():
        return _KEY_PATH.read_bytes()
    key = Fernet.generate_key()
    try:
        _KEY_PATH.write_bytes(key)
    except Exception:
        # If file write fails, still return the in-memory key (non-persistent)
        pass
    return key


def _fernet() -> Fernet:
    global _cached
    if _cached is None:
        _cached = Fernet(_load_or_create_key())
    return _cached


def encrypt_text(plain: str) -> str:
    if plain is None:
        plain = ""
    token = _fernet().encrypt(plain.encode("utf-8"))
    return token.decode("utf-8")


def decrypt_text(token: str) -> str:
    if not token:
        return ""
    try:
        data = _fernet().decrypt(token.encode("utf-8"))
        return data.decode("utf-8", errors="replace")
    except InvalidToken:
        # Backward compatibility or corrupted data
        return ""
