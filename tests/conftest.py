"""Shared fixtures.

`get_settings()` is `@lru_cache`, so any test that touches env must run with a clean
cache — the autouse fixture handles that and seeds a dummy API key.
"""

from __future__ import annotations

import pytest

from app.settings import get_settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "test-key-123")
    for var in ("PORT", "WEBSITES_PORT", "STT_MODE", "STT_SAMPLE_RATE", "STT_BASE", "LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)
