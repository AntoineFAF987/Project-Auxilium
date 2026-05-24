# -*- coding: utf-8 -*-
"""
Auto-sync scheduler (Copilot-like continuous mail sync).

This module runs a background thread that periodically triggers email sync
for all users who have selected email folders. It mimics the Copilot experience
where mail ingestion happens automatically without user intervention.

The scheduler:
- Runs every N minutes (configurable via SYNC_INTERVAL_MINUTES env var, default 15).
- Scans the email_folder_selection DB to find all tenant/user combinations with selected folders.
- For each user, calls _do_sync_job (from routes_mail_sync) in a thread pool.
- Logs sync activity to the console (visible in backend logs).

To enable: call start_auto_sync_scheduler() from the main application startup.
To disable: set environment variable AUTO_SYNC_ENABLED=false.
"""
import os
import time
import threading
import json
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor

from .db import get_connection
from .routes_mail_sync import _do_sync_job
from .config import CONFIG_PATH

SYNC_INTERVAL_MINUTES = int(os.getenv("SYNC_INTERVAL_MINUTES", "15"))
AUTO_SYNC_ENABLED = os.getenv("AUTO_SYNC_ENABLED", "true").lower() == "true"
MAX_WORKERS = int(os.getenv("AUTO_SYNC_MAX_WORKERS", "4"))

_scheduler_thread = None
_stop_flag = threading.Event()

def _get_all_users_with_folders() -> List[Tuple[str, str, List[str]]]:
    """
    Returns a list of (tenant_id, user_id, [folder_names]) tuples for all users
    who have at least one selected folder.
    """
    with get_connection() as conn:
        cur = conn.execute("""
            SELECT tenant_id, user_id, folders_json
            FROM email_folder_selection
        """)
        rows = cur.fetchall()
    
    # Parse folders_json and filter users with at least one folder
    result = []
    for row in rows:
        tenant_id = row["tenant_id"] or ""
        user_id = row["user_id"] or ""
        try:
            folders = json.loads(row["folders_json"] or "[]")
            if isinstance(folders, list) and folders:
                result.append((tenant_id, user_id, folders))
        except Exception:
            continue
    
    return result

def _sync_user(tenant_id: str, user_id: str, folders: List[str]):
    """
    Triggers a sync job for a single user.
    Note: we don't have a bearer token in the scheduler context, so we rely on
    the legacy ingest path (no front access token).
    """
    try:
        print(f"[AUTO_SYNC] Starting sync for tenant={tenant_id}, user={user_id}, folders={folders}")
        _do_sync_job(tenant_id, user_id, bearer_token=None, folders=folders)
        print(f"[AUTO_SYNC] Finished sync for tenant={tenant_id}, user={user_id}")
    except Exception as e:
        print(f"[AUTO_SYNC] Error syncing tenant={tenant_id}, user={user_id}: {e}")

def _scheduler_loop():
    """
    Main loop: every SYNC_INTERVAL_MINUTES, scan all users and trigger sync.
    """
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    print(f"[AUTO_SYNC] Scheduler started (interval={SYNC_INTERVAL_MINUTES}min, workers={MAX_WORKERS})")
    
    while not _stop_flag.is_set():
        try:
            users = _get_all_users_with_folders()
            if users:
                print(f"[AUTO_SYNC] Found {len(users)} user(s) with selected folders; queueing sync jobs...")
                for tenant_id, user_id, folders in users:
                    if _stop_flag.is_set():
                        break
                    executor.submit(_sync_user, tenant_id, user_id, folders)
            else:
                print("[AUTO_SYNC] No users with selected folders; skipping this cycle.")
        except Exception as e:
            print(f"[AUTO_SYNC] Scheduler cycle error: {e}")
        
        # Wait for next cycle (or until stop)
        _stop_flag.wait(timeout=SYNC_INTERVAL_MINUTES * 60)
    
    executor.shutdown(wait=True)
    print("[AUTO_SYNC] Scheduler stopped.")

def start_auto_sync_scheduler():
    """
    Starts the auto-sync background thread (if enabled).
    Call this once from the main application startup (e.g., in server.py).
    """
    global _scheduler_thread
    if not AUTO_SYNC_ENABLED:
        print("[AUTO_SYNC] Auto-sync scheduler disabled (AUTO_SYNC_ENABLED=false)")
        return
    
    if _scheduler_thread is not None and _scheduler_thread.is_alive():
        print("[AUTO_SYNC] Scheduler already running; skipping start.")
        return
    
    _stop_flag.clear()
    _scheduler_thread = threading.Thread(target=_scheduler_loop, daemon=True, name="auto-sync-scheduler")
    _scheduler_thread.start()
    print("[AUTO_SYNC] Scheduler thread started.")

def stop_auto_sync_scheduler():
    """
    Stops the auto-sync background thread gracefully.
    """
    global _scheduler_thread
    if _scheduler_thread is None or not _scheduler_thread.is_alive():
        print("[AUTO_SYNC] Scheduler not running; nothing to stop.")
        return
    
    print("[AUTO_SYNC] Stopping scheduler...")
    _stop_flag.set()
    _scheduler_thread.join(timeout=10)
    _scheduler_thread = None
    print("[AUTO_SYNC] Scheduler stopped.")
