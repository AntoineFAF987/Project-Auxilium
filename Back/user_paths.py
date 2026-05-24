# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 09:54:20 2025

@author: aejau
"""

# user_paths.py
import os
from dataclasses import dataclass

BASE_DATA = os.path.abspath(os.getenv("DATA_DIR", "./data"))  # override possible via .env

@dataclass
class UserPaths:
    email: str
    root: str
    emails_cache: str
    rag_index: str
    messages_json: str

def _safe(s: str) -> str:
    return s.replace("/", "_").replace("\\", "_").replace("..", "_")

def _ensure(path: str):
    os.makedirs(path, exist_ok=True)

def paths_for(email: str) -> UserPaths:
    e = _safe(email.lower())
    root = os.path.join(BASE_DATA, e)
    p = UserPaths(
        email=e,
        root=root,
        emails_cache=os.path.join(root, "emails_cache"),
        rag_index=os.path.join(root, "index_rag"),
        messages_json=os.path.join(root, "emails_cache", "messages.json"),
    )
    _ensure(p.root); _ensure(p.emails_cache); _ensure(p.rag_index)
    return p
