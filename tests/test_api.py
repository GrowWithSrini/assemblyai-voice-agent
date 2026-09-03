from unittest.mock import AsyncMock

from app import api
from app.assemblyai import AssemblyAIError


def test_config_shape(client):
    r = client.get("/api/config")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"stt_ws_url", "speech_model", "sample_rate", "mode", "llm_model"}
    assert body["stt_ws_url"] == "wss://streaming.assemblyai.com/v3/ws"
    assert body["sample_rate"] == 16000


def test_favicon_is_svg(client):
    r = client.get("/favicon.ico")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/svg+xml")
    assert "<svg" in r.text


def test_index_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "voice agent" in r.text.lower()


def test_stt_token_ok(client, monkeypatch):
    monkeypatch.setattr(api, "mint_stt_token", AsyncMock(return_value={"token": "tk", "expires_in_seconds": 60}))
    r = client.get("/api/stt-token")
    assert r.status_code == 200
    assert r.json()["token"] == "tk"


def test_stt_token_upstream_error_maps_status(client, monkeypatch):
    monkeypatch.setattr(api, "mint_stt_token", AsyncMock(side_effect=AssemblyAIError(502, "upstream boom")))
    r = client.get("/api/stt-token")
    assert r.status_code == 502
    assert r.json() == {"detail": "upstream boom"}


def test_chat_ok(client, monkeypatch):
    monkeypatch.setattr(api, "run_agent", AsyncMock(return_value="hello!"))
    r = client.post("/api/chat", json={"session_id": "s1", "message": "hi"})
    assert r.status_code == 200
    assert r.json() == {"reply": "hello!"}


def test_chat_forwards_session_and_message(client, monkeypatch):
    spy = AsyncMock(return_value="ok")
    monkeypatch.setattr(api, "run_agent", spy)
    client.post("/api/chat", json={"session_id": "abc", "message": "yo"})
    spy.assert_awaited_once_with("abc", "yo")


def test_chat_rate_limit_includes_retry_after(client, monkeypatch):
    monkeypatch.setattr(
        api, "run_agent", AsyncMock(side_effect=AssemblyAIError(429, "slow", retry_after=50))
    )
    r = client.post("/api/chat", json={"session_id": "s1", "message": "hi"})
    assert r.status_code == 429
    assert r.json() == {"detail": "slow", "retry_after": 50}


def test_chat_validation_error(client):
    r = client.post("/api/chat", json={"session_id": "s1"})
    assert r.status_code == 422
