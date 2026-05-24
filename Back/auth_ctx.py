# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 09:51:48 2025

@author: aejau
"""

# auth_ctx.py
from typing import Dict, Any
from fastapi import Depends, HTTPException, Request
from auth_ms import require_access_token  # ← déjà créé avant

def get_user_info(request: Request, user: Dict[str, Any] = Depends(require_access_token)) -> Dict[str, Any]:
    """
    Standardise les infos utilisateur à partir du token Microsoft vérifié.
    On récupère aussi le token brut depuis le header Authorization pour proxifier vers Graph.
    """
    email = user.get("preferred_username") or user.get("email")
    sub   = user.get("sub") or user.get("oid")
    if not email:
        raise HTTPException(status_code=400, detail="Email utilisateur introuvable dans le token.")
    # Token brut depuis l'en-tête (format 'Bearer <token>')
    auth = request.headers.get("authorization") or request.headers.get("Authorization") or ""
    raw_token = auth.split(" ", 1)[1] if " " in auth else ""
    return {
        "email": email,
        "sub": sub,
        "name": user.get("name"),
        "access_token": raw_token,
        "claims": user,
    }
