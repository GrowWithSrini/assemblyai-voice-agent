import httpx
import pytest
import respx

from app.assemblyai import AssemblyAIError, mint_stt_token
from app.settings import get_settings

TOKEN_URL = "https://streaming.assemblyai.com/v3/token"


@respx.mock
async def test_mint_returns_upstream_json():
    respx.get(TOKEN_URL).mock(
        return_value=httpx.Response(200, json={"token": "abc", "expires_in_seconds": 60})
    )
    out = await mint_stt_token(get_settings())
    assert out == {"token": "abc", "expires_in_seconds": 60}


@respx.mock
async def test_mint_sends_raw_key_without_bearer():
    route = respx.get(TOKEN_URL).mock(return_value=httpx.Response(200, json={"token": "x"}))
    await mint_stt_token(get_settings())
    assert route.calls.last.request.headers["authorization"] == "test-key-123"


@respx.mock
async def test_mint_maps_upstream_failure_to_502():
    respx.get(TOKEN_URL).mock(return_value=httpx.Response(403, text="forbidden"))
    with pytest.raises(AssemblyAIError) as excinfo:
        await mint_stt_token(get_settings())
    assert excinfo.value.status_code == 502
    assert "403" in excinfo.value.detail


async def test_mint_without_key_raises_500(monkeypatch):
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    get_settings.cache_clear()
    with pytest.raises(AssemblyAIError) as excinfo:
        await mint_stt_token(get_settings())
    assert excinfo.value.status_code == 500


def test_error_carries_retry_after():
    err = AssemblyAIError(429, "slow down", retry_after=50)
    assert err.status_code == 429
    assert err.retry_after == 50
    assert str(err) == "slow down"


def test_error_retry_after_defaults_none():
    assert AssemblyAIError(502, "boom").retry_after is None
