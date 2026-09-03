import pytest
from pydantic import ValidationError

from app.schemas import ChatRequest


def test_valid_request():
    r = ChatRequest(session_id="s1", message="hello there")
    assert r.session_id == "s1"
    assert r.message == "hello there"


def test_empty_message_rejected():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="s1", message="")


def test_empty_session_id_rejected():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="", message="hi")


def test_session_id_length_capped():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="x" * 201, message="hi")


def test_missing_field_rejected():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="s1")
