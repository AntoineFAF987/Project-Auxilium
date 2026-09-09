# -*- coding: utf-8 -*-
from fastapi import APIRouter, HTTPException, Request
from .config import CONFIG_PATH
from .email_db import get_selected_folders
from .directories_db import get_selected_directories
from .index_singleton import idx, idx_lock

import os, json
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from ingest_emails import ingest_emails, ingest_emails_with_access_token  # selon ton mode
except Exception:
    ingest_emails = None
    ingest_emails_with_access_token = None

router = APIRouter()

# ---------- Helpers ----------
def _read_json(path: str, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _write_json(path: str, data: Any):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _email_paths_from_config(config_path: str) -> Dict[str, str]:
    cfg = _read_json(str(config_path), default={})
    E = (cfg or {}).get("email_ingest") or {}
    return {
        "out_dir": E.get("output_dir") or "",
        "flat_dir": E.get("flatten_for_rag_dir") or "",
        "att_dir": E.get("attachments_dir") or "",
    }

def _purge_emails_not_in_selected(config_path: str, keep_folders: List[str]) -> Dict[str, int]:
    """
    Supprime localement tous les e-mails (JSON + TXT + pièces jointes) dont le champ 'folder'
    n'est pas dans la liste des dossiers sélectionnés.
    """
    paths = _email_paths_from_config(config_path)
    out_dir, flat_dir, att_dir = paths["out_dir"], paths["flat_dir"], paths["att_dir"]
    if not out_dir or not flat_dir or not att_dir:
        return {"json_removed": 0, "txt_removed": 0, "att_removed": 0}

    out = Path(out_dir)
    flat = Path(flat_dir)
    att  = Path(att_dir)

    keep = {k.strip().lower() for k in (keep_folders or [])}
    json_removed = txt_removed = att_removed = 0

    for js in out.glob("*.json"):
        remove_this = False
        data = None
        try:
            data = json.loads(js.read_text(encoding="utf-8", errors="ignore"))
            folder = (data or {}).get("folder")
            if folder is not None and folder.strip().lower() not in keep:
                remove_this = True
        except Exception:
            # JSON corrompu -> on supprime
            remove_this = True

        if not remove_this:
            continue

        try:
            js.unlink()
            json_removed += 1
        except Exception:
            pass

        base = js.stem
        txt = flat / f"{base}.txt"
        if txt.exists():
            try:
                txt.unlink(); txt_removed += 1
            except Exception:
                pass

        # Supprimer aussi les pièces jointes listées dans le JSON
        try:
            for p in (data.get("attachment_paths") or []):
                pth = Path(p)
                if att in pth.parents or pth.parent == att:
                    if pth.exists():
                        pth.unlink(); att_removed += 1
        except Exception:
            pass

    return {"json_removed": json_removed, "txt_removed": txt_removed, "att_removed": att_removed}

def _prune_sync_state_for_unselected(config_path: str, keep_folders: List[str]) -> Dict[str, int]:
    """Retire de sync_state.json toutes les entrées dont le dossier n'est pas sélectionné.
    Clés concernées: 'legacy:<user>:<folder>' et 'front:<upn>:<folder>'.
    """
    paths = _email_paths_from_config(config_path)
    state_path = os.path.join(paths["out_dir"], "sync_state.json")
    state = _read_json(state_path, default={}) or {}

    keep = {k.strip().lower() for k in (keep_folders or [])}
    removed = 0
    new_state = {}

    for k, v in state.items():
        if isinstance(k, str) and (k.startswith("legacy:") or k.startswith("front:")):
            folder = k.rsplit(":", 1)[-1] if ":" in k else ""
            if folder not in keep:
                removed += 1
                continue
        new_state[k] = v

    if removed:
        _write_json(state_path, new_state)

    return {"state_removed": removed, "state_total": len(new_state)}

def _ensure_backfill_if_folder_empty(config_path: str, keep_folders: List[str]) -> Dict[str, int]:
    """Si un dossier est sélectionné mais qu'aucun JSON ne lui correspond (après purge manuelle, par ex.),
    on supprime son entrée de sync_state pour forcer le backfill lors de la prochaine ingestion.
    """
    paths = _email_paths_from_config(config_path)
    out_dir = paths["out_dir"]
    state_path = os.path.join(out_dir, "sync_state.json")
    state = _read_json(state_path, default={}) or {}

    # Comptage JSON par folder
    counts = {f.lower(): 0 for f in (keep_folders or [])}
    for js in Path(out_dir).glob("*.json"):
        try:
            data = json.loads(js.read_text(encoding="utf-8", errors="ignore"))
            folder = (data or {}).get("folder")
            if folder:
                f = folder.strip().lower()
                if f in counts:
                    counts[f] += 1
        except Exception:
            continue

    removed = 0
    for folder, nb in counts.items():
        if nb == 0:
            # Effacer toutes les clés de ce folder (front/legacy)
            keys = [k for k in list(state.keys())
                    if isinstance(k, str) and (k.endswith(f":{folder}")) and (k.startswith("legacy:") or k.startswith("front:"))]
            if keys:
                for k in keys:
                    state.pop(k, None); removed += 1

    if removed:
        _write_json(state_path, state)

    return {"empty_folders_reset": removed}

# ---- NEW: recalculer les roots (dossiers autorisés + emails aplatis si sélection mail) ----
def _recompute_roots_with_flat_emails() -> List[str]:
    roots: List[str] = []
    # 1) Dossiers locaux autorisés (DB)
    try:
        selected_dirs = get_selected_directories() or []
    except Exception:
        selected_dirs = []
    for d in selected_dirs:
        if not d.get("enabled", True):
            continue
        p = (d.get("path") or "").strip()
        if p:
            roots.append(os.path.abspath(p))

    # 2) Emails aplatis et leurs pièces jointes si au moins un dossier mail est sélectionné.
    try:
        mail_folders = get_selected_folders() or []
    except Exception:
        mail_folders = []
    if mail_folders:
        flat_dir = _email_paths_from_config(str(CONFIG_PATH))["flat_dir"]
        if flat_dir:
            flat_abs = os.path.abspath(flat_dir)
            if os.path.isdir(flat_abs):
                roots.append(flat_abs)
        att_dir = _email_paths_from_config(str(CONFIG_PATH))["att_dir"]
        if att_dir:
            att_abs = os.path.abspath(att_dir)
            if os.path.isdir(att_abs):
                roots.append(att_abs)

    # Dédoublonner
    seen, uniq = set(), []
    for r in roots:
        if r not in seen:
            seen.add(r); uniq.append(r)
    return uniq

# ---------- Endpoints ----------
@router.post("/ingest_emails")
def ingest_mails():
    folders = get_selected_folders() or []

    purged = _purge_emails_not_in_selected(str(CONFIG_PATH), folders)
    pruned = _prune_sync_state_for_unselected(str(CONFIG_PATH), folders)
    backfill_reset = _ensure_backfill_if_folder_empty(str(CONFIG_PATH), folders)

    if not folders:
        # Rien d'autorisé : purge + reset du cache + MAJ de l'index (les mails disparaissent)
        with idx_lock:
            idx.roots = _recompute_roots_with_flat_emails()
            idx.rebuild_incremental()

        return {
            "ok": True,
            "purged": purged,
            "pruned_state": pruned,
            "backfill_reset": backfill_reset,
            "note": "Aucun dossier e-mail sélectionné : purge + reset cache. Pas d’ingestion."
        }

    if ingest_emails is None:
        raise HTTPException(status_code=400, detail="ingest_emails non disponible")

    # Ingestion (nouveaux e-mails ou backfill si cache reset)
    ingest_emails(str(CONFIG_PATH), override_folders=folders)

    # 🔁 Réindexation incrémentale du RAG **en ré-évaluant les roots** (ajout du flat_dir)
    with idx_lock:
        idx.roots = _recompute_roots_with_flat_emails()
        idx.rebuild_incremental()

    return {
        "ok": True,
        "purged": purged,
        "pruned_state": pruned,
        "backfill_reset": backfill_reset,
        "note": "Ingestion e-mails (nouveaux) + purge dossiers désélectionnés + cache nettoyé + réindex local (mails inclus)."
    }

@router.post("/ingest_emails_only")
def ingest_mails_only(request: Request):
    """
    Variante via access token FRONT :
      - purge + nettoyage cache identiques
      - ingestion via le token
      - puis réindex local en recalculant les roots (pour inclure le flat_dir)
    """
    auth = request.headers.get("Authorization") or ""
    bearer = auth.split(" ", 1)[1].strip() if auth.startswith("Bearer ") else None

    folders = get_selected_folders() or []
    purged = _purge_emails_not_in_selected(str(CONFIG_PATH), folders)
    pruned = _prune_sync_state_for_unselected(str(CONFIG_PATH), folders)
    backfill_reset = _ensure_backfill_if_folder_empty(str(CONFIG_PATH), folders)

    if not bearer and ingest_emails is None:
        raise HTTPException(status_code=400, detail="Token manquant et fallback ingest_emails indisponible")

    if not folders:
        # Rien d'autorisé : pas d’ingestion → mais on reflète l’état (purge) dans l’index
        with idx_lock:
            idx.roots = _recompute_roots_with_flat_emails()
            idx.rebuild_incremental()

        return {
            "ok": True,
            "purged": purged,
            "pruned_state": pruned,
            "backfill_reset": backfill_reset,
            "mode": "front_token" if bearer else "legacy",
            "note": "Aucun dossier sélectionné : purge + reset cache. Réindex local (mails exclus)."
        }

    try:
        if bearer and ingest_emails_with_access_token is not None:
            ingest_emails_with_access_token(str(CONFIG_PATH), bearer, override_folders=folders)
            mode = "front_token"
        else:
            ingest_emails(str(CONFIG_PATH), override_folders=folders)
            mode = "legacy"

        with idx_lock:
            idx.roots = _recompute_roots_with_flat_emails()
            idx.rebuild_incremental()

        # 🔁 Réindexation locale **avec** recalcul des roots (inclure flat_dir)
        return {
            "ok": True,
            "purged": purged,
            "pruned_state": pruned,
            "backfill_reset": backfill_reset,
            "mode": mode,
            "note": "Emails ingérés. Réindex local avec dossiers autorisés + emails aplatis."
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ingestion KO: {e}")
