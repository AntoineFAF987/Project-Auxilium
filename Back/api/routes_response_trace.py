from fastapi import APIRouter, HTTPException

from .answer_pipeline import RESPONSE_TRACES

router = APIRouter(prefix="/debug/response-traces", tags=["debug"])


@router.get("")
def list_response_traces():
    return {"traces": RESPONSE_TRACES.list()}


@router.get("/{request_id}")
def get_response_trace(request_id: str):
    trace = RESPONSE_TRACES.get(request_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace introuvable")
    return trace
