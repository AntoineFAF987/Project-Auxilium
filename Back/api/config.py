# -*- coding: utf-8 -*-
from pathlib import Path
import json
from typing import Dict, Tuple, List, Any

# --------- Config générale ---------
APP_DIR = Path(__file__).resolve().parent.parent  # racine du projet
CONFIG_PATH = APP_DIR / "config.json"

def _load_cfg() -> dict:
    if not CONFIG_PATH.exists():
        raise RuntimeError(f"config.json introuvable: {CONFIG_PATH}")
    raw = CONFIG_PATH.read_text(encoding="utf-8").strip()
    return json.loads(raw) if raw else {}

def _core_from_cfg(cfg: dict):
    roots = cfg.get("roots") or [str(APP_DIR)]
    index_dir = cfg.get("index_dir") or str(APP_DIR / "index_rag")
    exclude_globs = cfg.get("exclude_globs") or []
    follow_symlinks = bool(cfg.get("follow_symlinks", False))
    return roots, index_dir, exclude_globs, follow_symlinks

def _rag_params(cfg: dict) -> Dict[str, Any]:
    rp = cfg.get("rag_params") or {}
    return {
        "answerability_threshold": float(rp.get("answerability_threshold", -0.50)),
        "final_k": int(rp.get("final_k", 12)),
        "max_context_chars": int(rp.get("max_context_chars", 7000)),
        "fresh_news_keywords": list(rp.get("fresh_news_keywords", [])),
        "overlap_min": int(rp.get("overlap_min", 1)),
        "context_relevance_threshold": float(rp.get("context_relevance_threshold", 0.3)),
        "enable_query_condensation": bool(rp.get("enable_query_condensation", True)),
        "enable_query_expansion": bool(rp.get("enable_query_expansion", True)),
        # NLI / faithfulness
        "enable_faithfulness_check": bool(rp.get("enable_faithfulness_check", True)),
        "faithfulness_threshold": float(rp.get("faithfulness_threshold", 0.5)),
        # Si true: n'exécute le check que pour STRICT(local) / STRICT(web_live)
        "faithfulness_strict_only": bool(rp.get("faithfulness_strict_only", True)),
    }

def _timeouts(cfg: dict) -> Dict[str, int]:
    t = cfg.get("timeouts") or {}
    return {
        "llm_sec": int(t.get("llm_sec", 18)),
        "web_sec": int(t.get("web_sec", 8)),
        "math_sec": int(t.get("math_sec", 8)),
    }

def _rate_limit(cfg: dict) -> Dict[str, int]:
    rl = cfg.get("rate_limit") or {}
    return {
        "max_req": int(rl.get("max_req", 12)),
        "per_sec": int(rl.get("per_sec", 10)),
    }

cfg = _load_cfg()
roots, index_dir, exclude_globs, follow_symlinks = _core_from_cfg(cfg)
rag_params = _rag_params(cfg)
timeouts = _timeouts(cfg)
rate_limit = _rate_limit(cfg)
