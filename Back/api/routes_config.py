# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 13:57:54 2025

@author: aejau
"""

# -*- coding: utf-8 -*-
import json, os
from pathlib import Path
from fastapi import APIRouter, HTTPException

router = APIRouter()

APP_ROOT = Path(__file__).resolve().parent.parent  # racine projet
CONFIG_PATH = APP_ROOT / "config.json"

def _read() -> dict:
    if CONFIG_PATH.exists():
        try:
            raw = CONFIG_PATH.read_text(encoding="utf-8")
            return json.loads(raw or "{}")
        except Exception:
            return {}
    return {}

def _write(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

@router.get("/config")
def get_config():
    return _read()

@router.put("/config")
def put_config(body: dict):
    cfg = _read()
    cfg.update(body.get("config") or {})
    _write(cfg)
    return cfg

@router.post("/config/test/llm")
def test_llm(llm: dict):
    return {"ok": True, "provider": llm.get("provider"), "model": llm.get("model"), "sample": "Hello"}

@router.post("/config/test/directory")
def test_directory(body: dict):
    p = (body.get("path") or "").strip()
    if not p:
        raise HTTPException(status_code=400, detail="Chemin vide.")
    if not os.path.isdir(p):
        raise HTTPException(status_code=400, detail="Dossier introuvable.")
    return {"ok": True, "message": "Accès OK"}

@router.post("/config/test/email")
def test_email(body: dict):
    return {"ok": True, "message": "Test e-mail OK (stub)"}
