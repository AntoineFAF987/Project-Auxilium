"""API endpoints for managing authorised directories (local folders).

This router exposes two endpoints to manage the set of local file
directories that the RAG engine is authorised to read:

* ``GET /settings/directories`` – return the currently authorised
  directories for the given tenant/user.
* ``PUT /settings/directories`` – persist a new list of authorised
  directories for the tenant/user.  The request body should include
  ``directories`` (list of objects) and optional ``tenant_id`` and
  ``user_id`` fields.

Each directory object must contain at least a ``path``.  Optional fields
``label`` (a friendly name) and ``enabled`` (boolean) are also accepted.
Invalid entries are ignored during persistence.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Query

from .directories_db import get_selected_directories, set_selected_directories


router = APIRouter()


@router.get("/settings/directories")
def read_directories(
    tenant_id: Optional[str] = Query(None), user_id: Optional[str] = Query(None)
) -> Dict[str, Any]:
    """Return the authorised directories for the given tenant/user.

    The result is a JSON object with a single key ``directories``.
    If no configuration exists, an empty list is returned.
    """
    dirs = get_selected_directories(tenant_id, user_id)
    return {"directories": dirs}


@router.put("/settings/directories")
def write_directories(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Persist a new list of authorised directories for the tenant/user.

    The payload should contain:

    * ``directories`` – list of objects with at least a ``path`` key.
    * ``tenant_id`` (optional)
    * ``user_id`` (optional)

    Malformed inputs result in a 400 error.  The response echoes the
    cleaned list of directories actually saved.
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Payload must be a JSON object")
    dirs = payload.get("directories")
    tenant_id: Optional[str] = payload.get("tenant_id")
    user_id: Optional[str] = payload.get("user_id")
    if not isinstance(dirs, list):
        raise HTTPException(status_code=400, detail="'directories' doit être une liste")
    # Each element should be a dict with at least a 'path'; detailed validation
    # happens in set_selected_directories.
    saved = set_selected_directories(dirs, tenant_id, user_id)
    return {"directories": saved}