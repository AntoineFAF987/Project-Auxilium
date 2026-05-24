"""Utility functions for persisting and retrieving authorised directories.

This module mirrors the behaviour of ``email_db.py`` but for local file
directories. Selected directories are stored in a single table
(``directory_selection``) with one row per tenant/user combination.  The
directories are persisted as a JSON list of objects with the following
structure::

    {
        "path": "C:\\Users\\me\\Documents",
        "label": "Mes docs",
        "enabled": true
    }

When persisting, duplicate paths are deduplicated (the last value wins)
and invalid entries are ignored.  When retrieving, each item is normalised
so that:

* ``path`` is a non‑empty string
* ``label`` is either ``None`` or a string
* ``enabled`` is a boolean (defaulting to ``True``)

Tenant and user identifiers are optional; if omitted the configuration
applies globally.
"""

from __future__ import annotations

import json
from typing import List, Optional, Dict, Any

from .db import get_connection, init_db


def get_selected_directories(
    tenant_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return the list of authorised directories for a given tenant/user.

    The return value is always a list.  Each element is a dict with keys
    ``path`` (str), ``label`` (str or None) and ``enabled`` (bool).  If no
    record exists for the specified tenant/user, an empty list is
    returned.  Any malformed JSON in the database is treated as empty.
    """
    init_db()  # ensure table exists
    with get_connection() as conn:
        if tenant_id is None and user_id is None:
            row = conn.execute(
                "SELECT dirs_json FROM directory_selection WHERE tenant_id IS NULL AND user_id IS NULL LIMIT 1"
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT dirs_json FROM directory_selection WHERE tenant_id = ? AND user_id = ? LIMIT 1",
                (tenant_id, user_id),
            ).fetchone()
        if not row:
            return []
        raw = row["dirs_json"] or "[]"
        try:
            data = json.loads(raw)
        except Exception:
            return []
        if not isinstance(data, list):
            return []
        result: List[Dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path", "")).strip()
            if not path:
                continue
            label = item.get("label")
            if label is not None and not isinstance(label, str):
                label = str(label)
            label = label.strip() if isinstance(label, str) and label.strip() else None
            enabled = item.get("enabled")
            if not isinstance(enabled, bool):
                enabled = True
            result.append({"path": path, "label": label, "enabled": enabled})
        return result


def set_selected_directories(
    dirs: List[Dict[str, Any]],
    tenant_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Persist a list of authorised directories for a tenant/user.

    ``dirs`` should be a list of objects with at least a ``path`` field.
    Duplicate paths are deduplicated, with the last occurrence winning.
    Items without a valid ``path`` are ignored.  The function returns the
    cleaned list that was actually saved, with keys ``path``, ``label`` and
    ``enabled``.
    """
    init_db()
    # deduplicate and normalise entries
    ordered: Dict[str, Dict[str, Any]] = {}
    for item in dirs or []:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip()
        if not path:
            continue
        label = item.get("label")
        if label is not None and not isinstance(label, str):
            label = str(label)
        label = label.strip() if isinstance(label, str) and label.strip() else None
        enabled = item.get("enabled")
        if not isinstance(enabled, bool):
            enabled = True
        # update or insert with path as key; last occurrence wins
        ordered[path] = {"path": path, "label": label, "enabled": enabled}
    # preserve insertion order
    clean_list = list(ordered.values())
    payload = json.dumps(clean_list, ensure_ascii=False)
    with get_connection() as conn:
        if tenant_id is None and user_id is None:
            row = conn.execute(
                "SELECT id FROM directory_selection WHERE tenant_id IS NULL AND user_id IS NULL LIMIT 1"
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE directory_selection SET dirs_json = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (payload, row["id"]),
                )
            else:
                conn.execute(
                    "INSERT INTO directory_selection (tenant_id, user_id, dirs_json) VALUES (NULL, NULL, ?)",
                    (payload,),
                )
        else:
            row = conn.execute(
                "SELECT id FROM directory_selection WHERE tenant_id = ? AND user_id = ? LIMIT 1",
                (tenant_id, user_id),
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE directory_selection SET dirs_json = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (payload, row["id"]),
                )
            else:
                conn.execute(
                    "INSERT INTO directory_selection (tenant_id, user_id, dirs_json) VALUES (?, ?, ?)",
                    (tenant_id, user_id, payload),
                )
        conn.commit()
    return clean_list