"""HTTP routes.

  GET  /api/config     non-secret STT settings the browser needs
  GET  /api/stt-token  mint a single-use realtime-STT token
  POST /api/chat       proxy one LLM Gateway completion
  GET  /favicon.(ico|svg)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, Response

from .agent import run_agent
from .assemblyai import AssemblyAIError, mint_stt_token
from .schemas import ChatRequest
from .settings import Settings, get_settings

router = APIRouter()

_FAVICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#22d3ee"/><stop offset=".55" stop-color="#8b5cf6"/>'
    '<stop offset="1" stop-color="#ec4899"/></linearGradient></defs>'
    '<rect width="32" height="32" rx="8" fill="url(#g)"/>'
    '<rect x="12.5" y="6" width="7" height="12" rx="3.5" fill="#0b0e1a"/>'
    '<path d="M9 15a7 7 0 0 0 14 0M16 22v4M12 26h8" fill="none" stroke="#0b0e1a" '
    'stroke-width="2" stroke-linecap="round"/></svg>'
)


def _error_response(exc: AssemblyAIError) -> JSONResponse:
    content: dict = {"detail": exc.detail}
    if exc.retry_after is not None:
        content["retry_after"] = exc.retry_after
    return JSONResponse(status_code=exc.status_code, content=content)


@router.get("/favicon.ico", include_in_schema=False)
@router.get("/favicon.svg", include_in_schema=False)
async def favicon() -> Response:
    return Response(content=_FAVICON, media_type="image/svg+xml")


@router.get("/api/config")
async def get_config(settings: Settings = Depends(get_settings)) -> dict:
    return {
        "stt_ws_url": settings.stt_ws_url,
        "speech_model": settings.stt_speech_model,
        "sample_rate": settings.stt_sample_rate,
        "mode": settings.stt_mode,
        "llm_model": settings.llm_model,
    }


@router.get("/api/stt-token")
async def create_stt_token(settings: Settings = Depends(get_settings)) -> JSONResponse:
    try:
        return JSONResponse(await mint_stt_token(settings))
    except AssemblyAIError as exc:
        return _error_response(exc)


@router.post("/api/chat")
async def chat(req: ChatRequest) -> JSONResponse:
    """One conversational turn. Memory is kept server-side by session_id (LangGraph)."""
    try:
        reply = await run_agent(req.session_id, req.message)
    except AssemblyAIError as exc:
        return _error_response(exc)
    return JSONResponse({"reply": reply})
