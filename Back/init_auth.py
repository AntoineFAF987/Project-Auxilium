# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 17:57:52 2025

@author: aejau
"""

# init_auth.py
from fastapi.middleware.cors import CORSMiddleware
from users_routes import router as users_router

def wire_auth(app):
    """
    Branche CORS + routes d'authentification sur l'app FastAPI existante.
    À appeler depuis api.py après avoir créé 'app'.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(users_router, prefix="")
