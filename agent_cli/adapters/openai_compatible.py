from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from agent_cli.core.provider import BaseProvider
from agent_cli.core.schema import (
    Message,
    ProviderCapabilities,
    ProviderInfo,
    RequestConfig,
    ResponseChunk,
    ToolCall,
)


class OpenAICompatibleAdapter(BaseProvider):
    """Adapter for OpenAI-style Chat Completions endpoints.

    Works for many providers that implement the OpenAI API shape (OpenAI, Groq, OpenRouter, Together, etc.).
    """

    def __init__(self, base_url: str, api_key: str, provider_name: str = "openai_compatible"):
        self._provider_name = provider_name

        url = base_url.rstrip("/")
        # normalize: accept https://host OR https://host/v1
        if not url.endswith("/v1"):
            url = url + "/v1"
        self.base_url = url
        self.api_key = api_key

        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=60.0)

    @property
    def name(self) -> str:
        return self._provider_name

    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for m in messages:
            msg: Dict[str, Any] = {"role": m.role.value, "content": m.content}
            if m.name:
                msg["name"] = m.name
            if m.tool_call_id:
                msg["tool_call_id"] = m.tool_call_id
            out.append(msg)
        return out

    def _convert_tools(self, tools: Optional[List[Any]]) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None
        # OpenAI-style tool schema
        return [{"type": "function", "function": t.model_dump()} for t in tools]

    async def chat(self, messages: List[Message], config: RequestConfig) -> ResponseChunk:
        payload: Dict[str, Any] = {
            "model": config.model,
            "messages": self._convert_messages(messages),
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
            "stream": False,
        }

        tools = self._convert_tools(config.tools)
        if tools:
            payload["tools"] = tools

        if config.json_mode:
            payload["response_format"] = {"type": "json_object"}

        r = await self.client.post("/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()

        choice = data["choices"][0]
        msg = choice["message"]
        tool_calls = None
        if isinstance(msg, dict) and "tool_calls" in msg:
            tool_calls = [
                ToolCall(
                    id=tc.get("id", ""),
                    name=tc.get("function", {}).get("name", ""),
                    arguments=tc.get("function", {}).get("arguments", ""),
                )
                for tc in msg.get("tool_calls", [])
            ]

        return ResponseChunk(
            content=msg.get("content"),
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason"),
        )

    async def stream_chat(self, messages: List[Message], config: RequestConfig) -> AsyncGenerator[ResponseChunk, None]:
        payload: Dict[str, Any] = {
            "model": config.model,
            "messages": self._convert_messages(messages),
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
            "stream": True,
        }
        tools = self._convert_tools(config.tools)
        if tools:
            payload["tools"] = tools
        if config.json_mode:
            payload["response_format"] = {"type": "json_object"}

        async with self.client.stream("POST", "/chat/completions", json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    line = line[6:]
                if line.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                if not data.get("choices"):
                    continue
                delta = data["choices"][0].get("delta", {})
                if not delta:
                    continue
                tool_calls = None
                if "tool_calls" in delta:
                    tool_calls = [
                        ToolCall(
                            id=tc.get("id", ""),
                            name=tc.get("function", {}).get("name", ""),
                            arguments=tc.get("function", {}).get("arguments", ""),
                        )
                        for tc in delta.get("tool_calls", [])
                    ]

                yield ResponseChunk(
                    content=delta.get("content"),
                    tool_calls=tool_calls,
                    finish_reason=data["choices"][0].get("finish_reason"),
                )

    async def detect(self) -> ProviderInfo:
        # lightweight check: list models (may fail if key invalid)
        caps = ProviderCapabilities(chat=True, tools=True, json_mode=True, streaming=True)
        confidence = 0.6
        detected_by = "probe"

        try:
            r = await self.client.get("/models")
            if r.status_code == 200:
                confidence = 0.9
                detected_by = "probe:/models"
            elif r.status_code in (401, 403):
                confidence = 0.7
                detected_by = "probe:/models_auth"
        except Exception:
            confidence = 0.4
            detected_by = "probe:error"

        return ProviderInfo(name=self.name, capabilities=caps, confidence=confidence, detected_by=detected_by)

    async def aclose(self) -> None:
        await self.client.aclose()
