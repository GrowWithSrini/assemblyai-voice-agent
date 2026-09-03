"""FastAPI app for the AssemblyAI realtime-STT voice agent.

The browser owns the orchestration loop (STT WebSocket → LLM → TTS). This server
only does what must not happen client-side:

  GET  /api/config     non-secret STT settings for the browser
  GET  /api/stt-token  mint a single-use realtime-STT token (raw-key auth)
  POST /api/chat       proxy the AssemblyAI LLM Gateway (raw-key auth)

...and serves the static frontend. The AssemblyAI API key never leaves the server.

Module layout:
  settings.py    env-driven configuration
  schemas.py     request models
  assemblyai.py  upstream calls to AssemblyAI (STT token + LLM Gateway)
  api.py         HTTP routes
  main.py        app assembly + static hosting
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .api import router
from .settings import STATIC_DIR


def create_app() -> FastAPI:
    app = FastAPI(title="AssemblyAI Realtime Voice Agent")
    app.include_router(router)
    # Mounted last so /api/* and /favicon.* win over the catch-all static handler.
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    from .settings import get_settings

    uvicorn.run(app, host="0.0.0.0", port=get_settings().port)
