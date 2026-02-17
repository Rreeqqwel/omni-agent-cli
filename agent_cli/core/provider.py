from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator, List

from .schema import Message, ProviderInfo, RequestConfig, ResponseChunk


class BaseProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:  # provider id
        raise NotImplementedError

    @abstractmethod
    async def chat(self, messages: List[Message], config: RequestConfig) -> ResponseChunk:
        """Non-streaming chat completion."""
        raise NotImplementedError

    @abstractmethod
    async def stream_chat(
        self, messages: List[Message], config: RequestConfig
    ) -> AsyncGenerator[ResponseChunk, None]:
        """Streaming chat completion."""
        raise NotImplementedError

    @abstractmethod
    async def detect(self) -> ProviderInfo:
        """Verify credentials and return detected capabilities for this configured endpoint."""
        raise NotImplementedError

    async def aclose(self) -> None:
        """Optional: close underlying http client."""
        return None
