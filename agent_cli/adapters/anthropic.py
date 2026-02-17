from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict, List, Tuple

import httpx

from agent_cli.core.provider import BaseProvider
from agent_cli.core.schema import (
    Message,
    ProviderCapabilities,
    ProviderInfo,
    RequestConfig,
    ResponseChunk,
    Role,
)


class AnthropicAdapter(BaseProvider):
    def __init__(self, base_url: str, api_key: str, provider_name: str = "anthropic"):
        self._provider_name = provider_name
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/v1"):
            # anthropic uses /v1/messages
            pass
        self.api_key = api_key
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=60.0,
        )

    @property
    def name(self) -> str:
        return self._provider_name

    def _split_system(self, messages: List[Message]) -> Tuple[str, List[Dict[str, Any]]]:
        system_parts: List[str] = []
        out: List[Dict[str, Any]] = []
        for m in messages:
            if m.role == Role.SYSTEM:
                if isinstance(m.content, str):
                    system_parts.append(m.content)
                else:
                    # list of parts
                    for part in m.content:
                        if part.type == "text" and part.text:
                            system_parts.append(part.text)
                continue

            # anthropic expects role user/assistant
            content = m.content
            out.append({"role": m.role.value, "content": content})
        return "\n".join(system_parts).strip(), out

    async def chat(self, messages: List[Message], config: RequestConfig) -> ResponseChunk:
        system, msgs = self._split_system(messages)
        payload: Dict[str, Any] = {
            "model": config.model,
            "messages": msgs,
            "max_tokens": config.max_tokens or 1024,
            "temperature": config.temperature,
        }
        if system:
            payload["system"] = system

        r = await self.client.post("/v1/messages", json=payload)
        r.raise_for_status()
        data = r.json()
        text = ""
        if isinstance(data.get("content"), list) and data["content"]:
            # [{type:'text', text:'...'}]
            text = data["content"][0].get("text", "")
        return ResponseChunk(content=text, finish_reason=data.get("stop_reason"))

    async def stream_chat(self, messages: List[Message], config: RequestConfig) -> AsyncGenerator[ResponseChunk, None]:
        system, msgs = self._split_system(messages)
        payload: Dict[str, Any] = {
            "model": config.model,
            "messages": msgs,
            "max_tokens": config.max_tokens or 1024,
            "temperature": config.temperature,
            "stream": True,
        }
        if system:
            payload["system"] = system

        async with self.client.stream("POST", "/v1/messages", json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    line = line[6:]
                try:
                    data = json.loads(line)
                except Exception:
                    continue

                t = data.get("type")
                if t == "content_block_delta":
                    delta = data.get("delta", {})
                    if delta.get("text"):
                        yield ResponseChunk(content=delta["text"])
                elif t == "message_stop":
                    yield ResponseChunk(finish_reason="stop")
                    break

    async def detect(self) -> ProviderInfo:
        caps = ProviderCapabilities(chat=True, tools=True, vision=True, streaming=True)
        confidence = 0.6
        detected_by = "probe"
        try:
            # Anthropic has no cheap models list for all accounts; do a minimal invalid request to fingerprint.
            r = await self.client.post("/v1/messages", json={"model": "invalid", "max_tokens": 1, "messages": []})
            if r.status_code in (400, 401, 403):
                confidence = 0.8
                detected_by = "probe:/v1/messages"
        except Exception:
            confidence = 0.4
            detected_by = "probe:error"

        return ProviderInfo(name=self.name, capabilities=caps, confidence=confidence, detected_by=detected_by)

    async def aclose(self) -> None:
        await self.client.aclose()
