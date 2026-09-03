"""Direct calls to AssemblyAI (not routed through LangChain).

Currently just the realtime-STT token mint. The LLM path lives in ``agent.py``
(LangGraph → LLM Gateway). Both use the RAW API key — no ``Bearer`` prefix.
"""

from __future__ import annotations

import httpx

from .settings import Settings


class AssemblyAIError(Exception):
    """An upstream call failed; carries the status to return to the client."""

    def __init__(self, status_code: int, detail: str, *, retry_after: int | None = None) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.retry_after = retry_after


def _require_key(settings: Settings) -> str:
    if not settings.api_key:
        raise AssemblyAIError(500, "ASSEMBLYAI_API_KEY is not set on the server")
    return settings.api_key


async def mint_stt_token(settings: Settings) -> dict:
    """Mint a fresh single-use realtime-STT token for the browser."""
    key = _require_key(settings)
    params = {
        "expires_in_seconds": settings.stt_token_expires_in_seconds,
        "max_session_duration_seconds": settings.stt_max_session_duration_seconds,
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{settings.stt_base}/v3/token", params=params, headers={"Authorization": key}
        )
    if resp.status_code != 200:
        raise AssemblyAIError(502, f"token request failed ({resp.status_code}): {resp.text}")
    return resp.json()
