# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 13:57:54 2025

@author: aejau
"""

import json
from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from rag_core.llm import ask_mistral_with_context
from runtime_settings import (
    CONFIG_PATH,
    get_runtime_settings,
    load_config_document,
    load_runtime_settings,
    reload_runtime_settings,
)

router = APIRouter()

def _read() -> dict:
    return load_config_document(CONFIG_PATH)

def _write(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged

@router.get("/config")
def get_config():
    return _read()

@router.put("/config")
def put_config(body: dict):
    cfg = _deep_merge(_read(), body.get("config") or {})
    try:
        load_runtime_settings(config_data=cfg, env={})
        candidate = load_runtime_settings(config_data=cfg)
    except (ValidationError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if candidate.requires_restart_compared_to(get_runtime_settings()):
        raise HTTPException(
            status_code=409,
            detail=(
                "Cette modification touche une configuration initialisée au "
                "démarrage. Modifiez config.json ou l'environnement puis "
                "redémarrez Auxilium."
            ),
        )
    _write(cfg)
    reload_runtime_settings()
    return cfg

@router.post("/config/test/llm")
def test_llm(llm: dict):
    candidate = _deep_merge(_read(), {"llm": llm})
    try:
        settings = load_runtime_settings(config_data=candidate)
        generation = settings.generation
        sample = ask_mistral_with_context(
            "Réponds uniquement par OK.",
            context_text="",
            history=[],
            model=generation.model,
            temperature=generation.temperature,
            max_tokens=16,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        "ok": True,
        "provider": generation.provider,
        "model": generation.model,
        "sample": sample,
    }

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
