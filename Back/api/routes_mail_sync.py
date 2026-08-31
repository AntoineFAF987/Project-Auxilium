# -*- coding: utf-8 -*-
"""
Mail sync endpoints (foundation to move toward a Copilot-like model):
- POST /u/sync_mails: triggers a background incremental ingest for the currently authenticated user.
- GET  /u/sync_status: returns latest sync status for the user.

Current implementation uses existing ingest_emails/ingest_emails_with_access_token
to avoid duplicating logic. It runs ingestion based on the folders selected in the
local DB (email_folder_selection). If an access token is provided, it will use the
front token path; otherwise it falls back to legacy ingestion.

Later, this can be swapped to Graph delta queries + webhooks without changing
the API surface.
"""
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from typing import Dict, Any, Optional, List
import os, json, time

from .email_db import get_selected_folders
from .config import CONFIG_PATH
from .index_singleton import idx, idx_lock
from .routes_ingest import (
    _recompute_roots_with_flat_emails,
    _purge_emails_not_in_selected,
    _prune_sync_state_for_unselected,
    _ensure_backfill_if_folder_empty,
)
from auth_ms import verify_ms_token

try:
    from ingest_emails import ingest_emails, ingest_emails_with_access_token
except Exception:
    ingest_emails = None
    ingest_emails_with_access_token = None

router = APIRouter()

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

def _status_path(tenant_id: str, user_id: str) -> str:
    cfg = _read_json(str(CONFIG_PATH), default={}) or {}
    E = (cfg.get("email_ingest") or {})
    out_dir = E.get("output_dir") or "."
    os.makedirs(out_dir, exist_ok=True)
    safe_t = (tenant_id or "-").replace(":", "_")
    safe_u = (user_id or "-").replace(":", "_")
    return os.path.join(out_dir, f"sync_status_{safe_t}_{safe_u}.json")

def _set_status(tenant_id: str, user_id: str, **kw):
    p = _status_path(tenant_id, user_id)
    cur = _read_json(p, default={}) or {}
    cur.update(kw)
    cur.setdefault("updated_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    _write_json(p, cur)
    return cur

def _get_status(tenant_id: str, user_id: str) -> Dict[str, Any]:
    p = _status_path(tenant_id, user_id)
    return _read_json(p, default={}) or {}

@router.post("/u/sync_mails")
async def sync_mails(request: Request, bt: BackgroundTasks):
    # Extract tenant/user from Authorization (ID token preferred)
    auth = request.headers.get("Authorization") or ""
    token = auth.split(" ", 1)[1].strip() if auth.startswith("Bearer ") else None
    if not token:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    try:
        payload = verify_ms_token(token)
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

    tenant_id = str(payload.get("tid") or "")
    user_id = str(payload.get("oid") or payload.get("sub") or "")
    if not tenant_id or not user_id:
        raise HTTPException(status_code=401, detail="Missing tenant/user claims in token")

    folders = get_selected_folders(tenant_id=tenant_id, user_id=user_id) or []

    # Persist initial status
    _set_status(tenant_id, user_id, status="queued", folders=folders, started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    # Queue background job
    bt.add_task(_do_sync_job, tenant_id, user_id, token, folders)
    return {"ok": True, "queued": True, "folders": folders}

def _rebuild_index_after_sync():
    # Recompute roots to include the flattened emails directory when folders are selected
    print("[AUTO_SYNC] index_rebuild_start")
    with idx_lock:
        idx.roots = _recompute_roots_with_flat_emails()
        idx.rebuild_incremental()
    print("[AUTO_SYNC] index_rebuild_end")

def _do_sync_job(tenant_id: str, user_id: str, bearer_token: Optional[str], folders: List[str]):
    _set_status(tenant_id, user_id, status="running")
    # Toujours purger/adapter le cache local en fonction de la sélection
    try:
        purged = _purge_emails_not_in_selected(str(CONFIG_PATH), folders)
        pruned = _prune_sync_state_for_unselected(str(CONFIG_PATH), folders)
        backfill_reset = _ensure_backfill_if_folder_empty(str(CONFIG_PATH), folders)
        _set_status(tenant_id, user_id, cache_cleanup={"purged": purged, "pruned": pruned, "backfill_reset": backfill_reset})
    except Exception:
        pass

    if not folders:
        # Nothing selected: ensure index excludes emails, then mark finished
        _rebuild_index_after_sync()
        _set_status(tenant_id, user_id, status="done", finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), processed=0)
        return

    processed = 0
    try:
        # Prefer front access token path if available (works with /me); fallback to legacy ingest
        if bearer_token and ingest_emails_with_access_token is not None:
            ingest_emails_with_access_token(str(CONFIG_PATH), bearer_token, override_folders=folders)
        elif ingest_emails is not None:
            ingest_emails(str(CONFIG_PATH), override_folders=folders)
        else:
            raise RuntimeError("No ingestion function available")
        # We don't have an exact processed count from the ingest module; leave as None
        processed = None  # type: ignore
        _rebuild_index_after_sync()
        _set_status(tenant_id, user_id, status="done", finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), processed=processed)
    except Exception as e:
        _set_status(tenant_id, user_id, status="error", error=str(e))

@router.get("/u/sync_status")
async def sync_status(request: Request):
    auth = request.headers.get("Authorization") or ""
    token = auth.split(" ", 1)[1].strip() if auth.startswith("Bearer ") else None
    if not token:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    try:
        payload = verify_ms_token(token)
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

    tenant_id = str(payload.get("tid") or "")
    user_id = str(payload.get("oid") or payload.get("sub") or "")
    if not tenant_id or not user_id:
        raise HTTPException(status_code=401, detail="Missing tenant/user claims in token")

    return _get_status(tenant_id, user_id)
