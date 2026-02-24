# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Pydantic models for session REST API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# =============================================================================
# Request Models
# =============================================================================


class SessionCreate(BaseModel):
    """Request model for creating/upserting a session."""

    session_id: str = Field(..., description="Unique session identifier (platform-provided)")
    platform: str = Field(..., description="Platform name (e.g., 'openclaw', 'claude-code')")
    channel: str | None = Field(None, description="Channel (e.g., 'discord', 'telegram', 'cli')")
    participants: list[str] | None = Field(None, description="List of participant names")
    metadata: dict[str, Any] | None = Field(None, description="Optional JSON metadata")
    parent_session_id: str | None = Field(None, description="Parent session ID for subagents")
    subagent_label: str | None = Field(None, description="Label for subagent sessions")
    subagent_model: str | None = Field(None, description="Model used for subagent")
    subagent_task: str | None = Field(None, description="Task description for subagent")


class SessionUpdate(BaseModel):
    """Request model for updating a session."""

    status: str | None = Field(None, description="Session status ('active', 'stale', 'completed')")
    metadata: dict[str, Any] | None = Field(None, description="Metadata to replace existing")
    participants: list[str] | None = Field(None, description="Participant list to set")
    ended_at: datetime | None = Field(None, description="End timestamp")


class MessageCreate(BaseModel):
    """Request model for creating a message."""

    speaker: str = Field(..., description="Speaker name (e.g., 'chris', 'jane', worker label)")
    role: str = Field(..., description="Message role ('user', 'assistant', 'system', 'tool')")
    content: str = Field(..., description="Message content")
    metadata: dict[str, Any] | None = Field(None, description="Optional message-specific metadata")


class MessageBatchCreate(BaseModel):
    """Request model for creating multiple messages."""

    messages: list[MessageCreate] = Field(..., description="List of messages to append")


# =============================================================================
# Response Models
# =============================================================================


class SessionResponse(BaseModel):
    """Response model for session data."""

    session_id: str
    platform: str
    channel: str | None
    participants: list[str]
    started_at: str
    last_activity_at: str
    ended_at: str | None
    status: str
    metadata: dict[str, Any]
    parent_session_id: str | None
    subagent_label: str | None
    subagent_model: str | None
    subagent_task: str | None
    current_chunk_index: int


class MessageResponse(BaseModel):
    """Response model for message data."""

    id: int
    session_id: str
    chunk_index: int
    timestamp: str
    speaker: str
    role: str
    content: str
    metadata: dict[str, Any]
    flushed_at: str | None


class FlushResponse(BaseModel):
    """Response model for flush operations."""

    session_id: str
    chunk_index: int | None = None
    message_count: int
    source_id: str | None = None
    flushed: bool


class FinalizeResponse(BaseModel):
    """Response model for session finalization."""

    session_id: str
    status: str
    flush: FlushResponse


class StaleFlushResponse(BaseModel):
    """Response model for stale session flush operation."""

    flushed: list[FlushResponse]
    count: int
