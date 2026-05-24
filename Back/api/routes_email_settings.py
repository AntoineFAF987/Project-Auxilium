"""
routes_email_settings.py – API endpoints for mail folder configuration.

This router exposes three endpoints:

* ``GET /emails/available`` – return a list of mail folders the user can
  potentially authorise.  Initially this is a static list; later you can
  replace it with a call to Microsoft Graph or another mail API to fetch
  real folder names.
* ``GET /settings/email-folders`` – return the currently authorised mail
  folders for the tenant/user.
* ``PUT /settings/email-folders`` – persist a new list of authorised mail
  folders for the tenant/user.

All tenant/user parameters are optional.  If omitted, the configuration
applies globally.
"""

from __future__ import annotations

from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query

from .email_db import get_selected_folders, set_selected_folders


router = APIRouter()


@router.get("/emails/available")
def emails_available(
    tenant_id: Optional[str] = Query(None), user_id: Optional[str] = Query(None)
):
    """List possible mail folders.

    For now this is a static list.  To integrate with Microsoft Graph or
    another mail service, replace this implementation with a call that
    retrieves the actual folder names.
    """
    folders = [
        "Boîte de réception",
        "Éléments envoyés",
        "Brouillons",
        "Éléments supprimés",
        "Courrier indésirable",
        "Archive",
    ]
    return {"folders": folders}


@router.get("/settings/email-folders")
def read_email_folders(
    tenant_id: Optional[str] = Query(None), user_id: Optional[str] = Query(None)
):
    """Return the authorised mail folders for the given tenant/user."""
    return {"folders": get_selected_folders(tenant_id, user_id)}


@router.put("/settings/email-folders")
def write_email_folders(payload: dict):
    """Persist a new list of authorised mail folders.

    ``payload`` should be a JSON object containing:
    - ``folders``: list of folder names (strings)
    - ``tenant_id`` (optional)
    - ``user_id`` (optional)
    """
    folders: List[str] = payload.get("folders") or []
    tenant_id: Optional[str] = payload.get("tenant_id")
    user_id: Optional[str] = payload.get("user_id")
    # Validate folder list
    if not isinstance(folders, list) or not all(isinstance(x, str) for x in folders):
        raise HTTPException(status_code=400, detail="'folders' doit être une liste de chaînes")
    return {"folders": set_selected_folders(folders, tenant_id, user_id)}