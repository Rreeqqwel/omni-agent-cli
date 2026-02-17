from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from agent_cli.core.provider import BaseProvider
from agent_cli.core.schema import Message, ProviderCapabilities, ProviderInfo, RequestConfig, ResponseChunk, Role


class GeminiAdapter(BaseProvider):
    """Minimal Google Gemini adapter (Generative Language API).

    Notes:
    - Expects base_url like https://generativelanguage.googleapis.com
    - Uses v1beta models endpoint: /v1beta/models/{model}:generateContent
    - Streaming is best-effort; if endpoint doesn't support SSE in your account/version, it falls back.
    """

    def __init__(self, base_url: str, api_key: str, provider_name: str = "google"):
        self._provider_name = provider_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0)

    @property
    def name(self) -> str:
        return self._provider_name

    def _to_contents(self, messages: List[Message]) -> List[Dict[str, Any]]:
        contents: List[Dict[str, Any]] = []
        for m in messages:
            if m.role == Role.SYSTEM:
                # Gemini supports systemInstruction; we'll handle elsewhere.
                continue
            role = "user" if m.role == Role.USER else "model"
            parts: List[Dict[str, Any]] = []
            if isinstance(m.content, str):
                parts.append({"text": m.content})
            else:
                for c in m.content:
                    if c.type == "text" and c.text:
                        parts.append({"text": c.text})
                    elif c.type in ("image_url", "image") and c.image_url:
                        # Gemini expects inline_data or file_data; URL support depends. Keep as text fallback.
                        parts.append({"text": f"[image] {c.image_url}"})
            contents.append({"role": role, "parts": parts})
        return contents

    def _system_instruction(self, messages: List[Message]) -> Optional[Dict[str, Any]]:
        sys_texts: List[str] = []
        for m in messages:
            if m.role != Role.SYSTEM:
                continue
            if isinstance(m.content, str):
                sys_texts.append(m.content)
            else:
                for c in m.content:
                    if c.type == "text" and c.text:
                        sys_texts.append(c.text)
        if not sys_texts:
            return None
        return {"parts": [{"text": "\n".join(sys_texts)}]}

    async def chat(self, messages: List[Message], config: RequestConfig) -> ResponseChunk:
        model = config.model
        path = f"/v1beta/models/{model}:generateContent"
        payload: Dict[str, Any] = {
            "contents": self._to_contents(messages),
            "generationConfig": {
                "temperature": config.temperature,
                "topP": config.top_p,
                **({"maxOutputTokens": config.max_tokens} if config.max_tokens else {}),
            },
        }
        sys_inst = self._system_instruction(messages)
        if sys_inst:
            payload["systemInstruction"] = sys_inst

        r = await self.client.post(path, params={"key": self.api_key}, json=payload)
        r.raise_for_status()
        data = r.json()

        text = ""
        cands = data.get("candidates") or []
        if cands:
            parts = ((cands[0].get("content") or {}).get("parts") or [])
            if parts:
                text = parts[0].get("text", "")
        return ResponseChunk(content=text, finish_reason=(cands[0].get("finishReason") if cands else None))

    async def stream_chat(self, messages: List[Message], config: RequestConfig) -> AsyncGenerator[ResponseChunk, None]:
        # Best-effort fallback: Gemini streaming varies by endpoint/version. We'll simulate by one-shot.
        chunk = await self.chat(messages, config)
        if chunk.content:
            yield ResponseChunk(content=chunk.content)
        yield ResponseChunk(finish_reason=chunk.finish_reason or "stop")

    async def detect(self) -> ProviderInfo:
        caps = ProviderCapabilities(chat=True, tools=True, vision=True, json_mode=True, streaming=True)
        confidence = 0.6
        detected_by = "probe"
        try:
            r = await self.client.get("/v1beta/models", params={"key": self.api_key})
            if r.status_code == 200:
                confidence = 0.9
                detected_by = "probe:/v1beta/models"
            elif r.status_code in (401, 403):
                confidence = 0.7
                detected_by = "probe:/v1beta/models_auth"
        except Exception:
            confidence = 0.4
            detected_by = "probe:error"
        return ProviderInfo(name=self.name, capabilities=caps, confidence=confidence, detected_by=detected_by)

    async def aclose(self) -> None:
        await self.client.aclose()
