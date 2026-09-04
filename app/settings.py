"""Environment-driven configuration.

Every value has a sensible default so the app boots with only ASSEMBLYAI_API_KEY set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# LLM Gateway models that reject a `tools` payload (HTTP 400). Tool calling is
# disabled by default on these; any other model gets tools unless LLM_ENABLE_TOOLS
# says otherwise.
_MODELS_WITHOUT_TOOLS = {"qwen3.5-4b-32k-fast"}


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    api_key: str | None  # same key is used for STT and the LLM Gateway

    stt_base: str
    stt_token_expires_in_seconds: int
    stt_max_session_duration_seconds: int
    stt_speech_model: str
    stt_sample_rate: int
    stt_mode: str

    llm_base: str
    llm_model: str
    llm_max_tokens: int
    llm_enable_tools: bool  # bind JSON-Schema tools to the model + add the ReAct ToolNode

    port: int

    @property
    def stt_ws_url(self) -> str:
        return (
            self.stt_base.replace("https://", "wss://").replace("http://", "ws://")
            + "/v3/ws"
        )


@lru_cache
def get_settings() -> Settings:
    llm_model = os.environ.get("LLM_MODEL", "qwen3.5-4b-32k-fast")
    return Settings(
        api_key=os.environ.get("ASSEMBLYAI_API_KEY"),
        stt_base=os.environ.get("STT_BASE", "https://streaming.assemblyai.com").rstrip("/"),
        stt_token_expires_in_seconds=int(os.environ.get("STT_TOKEN_EXPIRES_IN_SECONDS", "60")),
        stt_max_session_duration_seconds=int(os.environ.get("STT_MAX_SESSION_DURATION_SECONDS", "3600")),
        stt_speech_model=os.environ.get("STT_SPEECH_MODEL", "universal-3-5-pro"),
        stt_sample_rate=int(os.environ.get("STT_SAMPLE_RATE", "16000")),
        stt_mode=os.environ.get("STT_MODE", "balanced"),
        llm_base=os.environ.get("LLM_BASE", "https://llm-gateway.assemblyai.com").rstrip("/"),
        llm_model=llm_model,
        llm_max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "400")),
        llm_enable_tools=_bool_env("LLM_ENABLE_TOOLS", default=llm_model not in _MODELS_WITHOUT_TOOLS),
        # PaaS hosts (Azure App Service / Container Apps) inject the listen port here.
        port=int(os.environ.get("PORT") or os.environ.get("WEBSITES_PORT") or "8000"),
    )
