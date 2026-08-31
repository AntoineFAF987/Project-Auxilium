# -*- coding: utf-8 -*-
from typing import Optional

from fastapi import APIRouter, HTTPException, Path, Request
from pydantic import BaseModel, Field

from .chats_db import create_project, delete_project, get_project, list_projects, rename_project, set_project_pinned
from .routes_chats import _try_get_auth_ids

router = APIRouter()


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)


class ProjectUpdate(BaseModel):
    name: Optional[str] = Field(default=None, max_length=120)
    pinned: Optional[bool] = None


class ProjectResponse(ProjectCreate):
    id: str
    pinned: bool = False
    created_at: str
    updated_at: str


def _project_name(name: str) -> str:
    if not (name := name.strip()):
        raise HTTPException(status_code=400, detail="Le nom du projet est obligatoire")
    if len(name) > 120:
        raise HTTPException(status_code=400, detail="Le nom du projet est trop long")
    return name


@router.get("/projects")
def api_list_projects(request: Request):
    tenant_id, user_id = _try_get_auth_ids(request)
    return {"items": list_projects(tenant_id, user_id)}


@router.post("/projects", status_code=201, response_model=ProjectResponse)
def api_create_project(body: ProjectCreate, request: Request = None):
    tenant_id, user_id = _try_get_auth_ids(request)
    return create_project(tenant_id, user_id, _project_name(body.name))


@router.get("/projects/{project_id}", response_model=ProjectResponse)
def api_get_project(project_id: str = Path(..., min_length=1), request: Request = None):
    tenant_id, user_id = _try_get_auth_ids(request)
    project = get_project(tenant_id, user_id, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Projet introuvable")
    return project


@router.patch("/projects/{project_id}", response_model=ProjectResponse)
def api_update_project(project_id: str, body: ProjectUpdate, request: Request = None):
    tenant_id, user_id = _try_get_auth_ids(request)
    if body.name is None and body.pinned is None:
        raise HTTPException(status_code=400, detail="Aucune modification demandee")
    project = get_project(tenant_id, user_id, project_id)
    if body.name is not None:
        project = rename_project(tenant_id, user_id, project_id, _project_name(body.name))
    if project and body.pinned is not None:
        project = set_project_pinned(tenant_id, user_id, project_id, body.pinned)
    if not project:
        raise HTTPException(status_code=404, detail="Projet introuvable")
    return project


@router.delete("/projects/{project_id}")
def api_delete_project(project_id: str, request: Request = None):
    tenant_id, user_id = _try_get_auth_ids(request)
    if not delete_project(tenant_id, user_id, project_id):
        raise HTTPException(status_code=404, detail="Projet introuvable")
    return {"ok": True}
