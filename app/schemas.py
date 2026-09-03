"""Request/response models for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=200)  # LangGraph thread id; one per conversation
    message: str = Field(min_length=1)
