from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class MessageContent(BaseModel):
    type: str = "text"  # "text" | "image_url"
    text: Optional[str] = None
    image_url: Optional[str] = None


class Message(BaseModel):
    role: Role
    content: Union[str, List[MessageContent]]
    name: Optional[str] = None
    tool_call_id: Optional[str] = None


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]


class ProviderCapabilities(BaseModel):
    chat: bool = True
    tools: bool = False
    vision: bool = False
    json_mode: bool = False
    streaming: bool = True


class ProviderInfo(BaseModel):
    name: str
    capabilities: ProviderCapabilities
    confidence: float = 1.0
    detected_by: str = "url"


class RequestConfig(BaseModel):
    model: str
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: Optional[int] = None
    stream: bool = False
    json_mode: bool = False
    tools: Optional[List[ToolDefinition]] = None


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: str


class ResponseChunk(BaseModel):
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: Optional[str] = None
