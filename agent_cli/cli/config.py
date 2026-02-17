from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    name: str
    base_url: str
    api_key: str = ""
    model: str
    provider_type: Optional[str] = None  # openai/openrouter/groq/... or openai_compatible


class AppConfig(BaseModel):
    providers: Dict[str, ProviderConfig] = Field(default_factory=dict)
    default_provider: Optional[str] = None


CONFIG_DIR = Path.home() / ".config" / "agent-cli"
CONFIG_FILE = CONFIG_DIR / "config.json"


def load_config() -> AppConfig:
    if not CONFIG_FILE.exists():
        return AppConfig()
    with CONFIG_FILE.open("r", encoding="utf-8") as f:
        return AppConfig(**json.load(f))


def save_config(config: AppConfig) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with CONFIG_FILE.open("w", encoding="utf-8") as f:
        json.dump(config.model_dump(), f, indent=2, ensure_ascii=False)
