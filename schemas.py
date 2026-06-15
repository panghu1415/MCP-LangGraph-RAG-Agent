from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="用户自然语言出行需求")
    thread_id: str = Field("default", description="会话 ID，用于多轮对话隔离")
    user_id: str = Field("default-user", description="用户 ID，用于偏好持久化")
    auto_save: bool = Field(False, description="是否自动保存最终行程")


class ToolTrace(BaseModel):
    name: str
    input: dict[str, Any]
    output: Any


class ChatResponse(BaseModel):
    answer: str
    thread_id: str
    user_id: str = "default-user"
    tool_traces: list[ToolTrace] = Field(default_factory=list)
    saved_file: str | None = None


class ResetRequest(BaseModel):
    thread_id: str = "default"


class ContextResponse(BaseModel):
    user_id: str
    user_preferences: dict[str, Any]
    conversation_state: dict[str, Any] = Field(default_factory=dict)
