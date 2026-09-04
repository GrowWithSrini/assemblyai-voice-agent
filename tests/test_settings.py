import dataclasses

import pytest

from app.settings import get_settings


def test_defaults():
    s = get_settings()
    assert s.api_key == "test-key-123"
    assert s.stt_base == "https://streaming.assemblyai.com"
    assert s.stt_speech_model == "universal-3-5-pro"
    assert s.stt_sample_rate == 16000
    assert s.stt_mode == "balanced"
    assert s.llm_model == "qwen3.5-4b-32k-fast"
    assert s.port == 8000


def test_stt_ws_url_derived_from_base():
    assert get_settings().stt_ws_url == "wss://streaming.assemblyai.com/v3/ws"


def test_stt_ws_url_http_base(monkeypatch):
    monkeypatch.setenv("STT_BASE", "http://localhost:9999")
    get_settings.cache_clear()
    assert get_settings().stt_ws_url == "ws://localhost:9999/v3/ws"


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("STT_MODE", "min_latency")
    monkeypatch.setenv("STT_SAMPLE_RATE", "8000")
    get_settings.cache_clear()
    s = get_settings()
    assert s.stt_mode == "min_latency"
    assert s.stt_sample_rate == 8000


def test_tools_disabled_by_default_for_free_model():
    # default LLM_MODEL is qwen3.5-4b-32k-fast, which rejects a tools payload
    assert get_settings().llm_enable_tools is False


def test_tools_enabled_by_default_for_other_models(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "claude-haiku-4-5-20251001")
    get_settings.cache_clear()
    assert get_settings().llm_enable_tools is True


def test_tools_env_override_wins(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "claude-haiku-4-5-20251001")
    monkeypatch.setenv("LLM_ENABLE_TOOLS", "false")
    get_settings.cache_clear()
    assert get_settings().llm_enable_tools is False

    monkeypatch.setenv("LLM_MODEL", "qwen3.5-4b-32k-fast")
    monkeypatch.setenv("LLM_ENABLE_TOOLS", "true")
    get_settings.cache_clear()
    assert get_settings().llm_enable_tools is True


def test_port_reads_WEBSITES_PORT(monkeypatch):
    monkeypatch.setenv("WEBSITES_PORT", "9000")
    get_settings.cache_clear()
    assert get_settings().port == 9000


def test_port_prefers_PORT_over_WEBSITES_PORT(monkeypatch):
    monkeypatch.setenv("PORT", "1234")
    monkeypatch.setenv("WEBSITES_PORT", "9000")
    get_settings.cache_clear()
    assert get_settings().port == 1234


def test_settings_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        get_settings().api_key = "mutated"
