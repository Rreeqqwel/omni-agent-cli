from __future__ import annotations

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.table import Table

from agent_cli.adapters.anthropic import AnthropicAdapter
from agent_cli.adapters.gemini import GeminiAdapter
from agent_cli.adapters.openai_compatible import OpenAICompatibleAdapter
from agent_cli.cli.config import AppConfig, ProviderConfig, load_config, save_config
from agent_cli.core.detector import ProviderDetector
from agent_cli.core.schema import Message, RequestConfig, Role


app = typer.Typer(add_completion=False, help="Universal multi-provider AI Agent CLI")
console = Console()


def _pick_adapter(p: ProviderConfig, detected: Optional[str] = None):
    provider_type = (p.provider_type or detected or "openai_compatible").lower()
    if provider_type in {"anthropic"}:
        return AnthropicAdapter(base_url=p.base_url, api_key=p.api_key, provider_name=provider_type)
    if provider_type in {"google", "gemini"}:
        return GeminiAdapter(base_url=p.base_url, api_key=p.api_key, provider_name="google")
    # default: OpenAI compatible
    return OpenAICompatibleAdapter(base_url=p.base_url, api_key=p.api_key, provider_name=provider_type)


@app.command()
def init() -> None:
    """Initialize config file (~/.config/agent-cli/config.json)."""
    config = load_config()
    save_config(config)
    console.print("[bold green]OK[/bold green] Config ensured at ~/.config/agent-cli/config.json")


@app.command("providers")
def providers_list() -> None:
    """List configured providers."""
    cfg = load_config()
    if not cfg.providers:
        console.print("No providers configured. Use: agent config-add")
        raise typer.Exit(code=1)

    t = Table(title="Configured providers")
    t.add_column("name", style="cyan")
    t.add_column("type")
    t.add_column("base_url")
    t.add_column("model")
    t.add_column("default")
    for name, p in cfg.providers.items():
        t.add_row(name, p.provider_type or "(auto)", p.base_url, p.model, "✓" if name == cfg.default_provider else "")
    console.print(t)


@app.command("config-add")
def config_add(
    name: str = typer.Argument(..., help="Config name, e.g. openai, groq, work"),
    base_url: str = typer.Option("https://api.openai.com", help="Provider base URL"),
    api_key: str = typer.Option("", help="API key (or leave empty and use env/secret manager)"),
    model: str = typer.Option("gpt-4o-mini", help="Default model for this provider"),
    provider_type: Optional[str] = typer.Option(None, help="Force provider type (openai, openrouter, groq, anthropic, google, ...)"),
    make_default: bool = typer.Option(False, help="Set as default provider"),
) -> None:
    """Add (or update) a provider configuration."""
    cfg = load_config()
    cfg.providers[name] = ProviderConfig(
        name=name,
        base_url=base_url,
        api_key=api_key,
        model=model,
        provider_type=provider_type,
    )
    if make_default or not cfg.default_provider:
        cfg.default_provider = name
    save_config(cfg)
    console.print(f"[bold green]OK[/bold green] saved provider [bold]{name}[/bold]")


@app.command()
def detect(
    base_url: str = typer.Argument(..., help="Base URL"),
    api_key: str = typer.Option("", help="Optional key for probing"),
) -> None:
    """Detect provider type/capabilities for a URL."""

    async def _run() -> None:
        info = await ProviderDetector().detect(base_url=base_url, api_key=api_key)
        console.print(f"Provider: [bold cyan]{info.name}[/bold cyan]")
        console.print(f"Confidence: {info.confidence:.2f} (by {info.detected_by})")
        console.print(f"Capabilities: {info.capabilities.model_dump()}")

    asyncio.run(_run())


@app.command()
def doctor(provider: Optional[str] = typer.Option(None, help="Provider name from config")) -> None:
    """Diagnostics: detect + try a lightweight API call."""

    async def _run() -> None:
        cfg = load_config()
        p_name = provider or cfg.default_provider
        if not p_name or p_name not in cfg.providers:
            console.print("[red]No provider configured.[/red]")
            raise typer.Exit(code=1)
        p = cfg.providers[p_name]
        info = await ProviderDetector().detect(p.base_url, p.api_key)
        console.print(f"Detected: [bold cyan]{info.name}[/bold cyan]  (confidence {info.confidence:.2f})")

        adapter = _pick_adapter(p, detected=info.name)
        try:
            detected_info = await adapter.detect()
            console.print(
                f"Adapter probe: [bold]{detected_info.name}[/bold] (confidence {detected_info.confidence:.2f}, {detected_info.detected_by})"
            )
            console.print(f"Capabilities: {detected_info.capabilities.model_dump()}")
        finally:
            await adapter.aclose()

    asyncio.run(_run())


@app.command()
def ask(
    prompt: str = typer.Argument(..., help="Prompt"),
    provider: Optional[str] = typer.Option(None, help="Provider name from config"),
    model: Optional[str] = typer.Option(None, help="Override model"),
    no_stream: bool = typer.Option(False, help="Disable streaming"),
) -> None:
    """Ask the agent a question."""

    async def _run() -> None:
        cfg: AppConfig = load_config()
        p_name = provider or cfg.default_provider
        if not p_name or p_name not in cfg.providers:
            console.print("[red]Error:[/red] No provider configured. Use 'agent config-add' first.")
            raise typer.Exit(code=1)

        p = cfg.providers[p_name]
        detected = await ProviderDetector().detect(p.base_url, p.api_key)
        adapter = _pick_adapter(p, detected=detected.name)

        messages = [Message(role=Role.USER, content=prompt)]
        req = RequestConfig(model=model or p.model, stream=not no_stream)

        console.print(f"[bold]Agent ({p_name})[/bold] → {adapter.name} | model={req.model}")
        full = ""
        try:
            if req.stream:
                with Live(Markdown(""), refresh_per_second=10, console=console) as live:
                    async for chunk in adapter.stream_chat(messages, req):
                        if chunk.content:
                            full += chunk.content
                            live.update(Markdown(full))
            else:
                out = await adapter.chat(messages, req)
                if out.content:
                    console.print(Markdown(out.content))
        finally:
            await adapter.aclose()

    asyncio.run(_run())


if __name__ == "__main__":
    app()
