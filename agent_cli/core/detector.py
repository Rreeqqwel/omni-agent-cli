from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple
from urllib.parse import urlparse

import httpx

from .schema import ProviderCapabilities, ProviderInfo


@dataclass
class DetectionResult:
    provider: str
    confidence: float
    detected_by: str
    capabilities: ProviderCapabilities


class ProviderDetector:
    """Detect provider type + basic capabilities.

    Strategy:
    1) URL hostname pattern matching (cheap, reliable)
    2) Safe probing (optional) to classify unknown endpoints as OpenAI-compatible
    """

    URL_PATTERNS = {
        "openai": [r"api\\.openai\\.com$"],
        "anthropic": [r"api\\.anthropic\\.com$"],
        "google": [r"generativelanguage\\.googleapis\\.com$"],
        "groq": [r"api\\.groq\\.com$"],
        "mistral": [r"api\\.mistral\\.ai$"],
        "openrouter": [r"openrouter\\.ai$"],
        "together": [r"api\\.together\\.xyz$"],
        "xai": [r"api\\.x\\.ai$", r"api\\.xai\\.com$"],
        "azure": [r"openai\\.azure\\.com$"],
        "huggingface": [r"api-inference\\.huggingface\\.co$"],
    }

    # These are typically OpenAI-compatible.
    OPENAI_COMPATIBLE = {"openai", "openrouter", "groq", "together", "azure", "xai", "mistral"}

    def detect_by_url(self, base_url: str) -> Optional[DetectionResult]:
        host = urlparse(base_url).hostname or ""
        host = host.lower()
        for provider, patterns in self.URL_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, host):
                    caps = self._default_caps(provider)
                    return DetectionResult(provider=provider, confidence=0.92, detected_by="url", capabilities=caps)
        return None

    def _default_caps(self, provider: str) -> ProviderCapabilities:
        if provider in self.OPENAI_COMPATIBLE:
            # baseline; real capabilities depend on model.
            return ProviderCapabilities(chat=True, tools=True, vision=False, json_mode=True, streaming=True)
        if provider == "anthropic":
            return ProviderCapabilities(chat=True, tools=True, vision=True, json_mode=False, streaming=True)
        if provider == "google":
            return ProviderCapabilities(chat=True, tools=True, vision=True, json_mode=True, streaming=True)
        if provider == "huggingface":
            return ProviderCapabilities(chat=True, tools=False, vision=False, json_mode=False, streaming=False)
        return ProviderCapabilities()

    async def probe_openai_compatible(self, base_url: str, api_key: str, timeout_s: float = 8.0) -> Optional[DetectionResult]:
        """Try to classify an unknown endpoint as OpenAI-compatible.

        Uses GET /v1/models (or /models if base_url already ends with /v1). This is usually cheap.
        Treats 200, 401, 403 with OpenAI-like error JSON as 'compatible'.
        """

        url = base_url.rstrip("/")
        # If user passes https://host/v1 already, keep it.
        if not url.endswith("/v1"):
            url_v1 = url + "/v1"
        else:
            url_v1 = url

        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            for path in ("/models",):
                try:
                    r = await client.get(url_v1 + path, headers=headers)
                except httpx.HTTPError:
                    continue

                if r.status_code == 200:
                    try:
                        j = r.json()
                    except Exception:
                        j = None
                    if isinstance(j, dict) and ("data" in j or "object" in j):
                        caps = ProviderCapabilities(chat=True, tools=True, json_mode=True, streaming=True)
                        return DetectionResult("openai_compatible", 0.75, "probe:/v1/models", caps)

                if r.status_code in (401, 403, 404):
                    # 401/403 often means key missing but endpoint exists.
                    # 404 can happen when /models disabled; still could be compatible.
                    try:
                        j = r.json()
                    except Exception:
                        j = None
                    if isinstance(j, dict):
                        # OpenAI-style: {"error": {...}}
                        if "error" in j and isinstance(j["error"], dict):
                            caps = ProviderCapabilities(chat=True, tools=True, json_mode=True, streaming=True)
                            return DetectionResult("openai_compatible", 0.6, "probe:error_fingerprint", caps)
        return None

    async def detect(self, base_url: str, api_key: str = "") -> ProviderInfo:
        by_url = self.detect_by_url(base_url)
        if by_url:
            return ProviderInfo(
                name=by_url.provider,
                capabilities=by_url.capabilities,
                confidence=by_url.confidence,
                detected_by=by_url.detected_by,
            )

        probe = await self.probe_openai_compatible(base_url, api_key)
        if probe:
            return ProviderInfo(
                name=probe.provider,
                capabilities=probe.capabilities,
                confidence=probe.confidence,
                detected_by=probe.detected_by,
            )

        return ProviderInfo(
            name="unknown",
            capabilities=ProviderCapabilities(chat=True, tools=False, vision=False, json_mode=False, streaming=False),
            confidence=0.2,
            detected_by="fallback",
        )
