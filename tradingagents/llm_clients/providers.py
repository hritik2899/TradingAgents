"""Provider registry for TradingAgents LLM clients."""

from __future__ import annotations

from dataclasses import dataclass

from .anthropic_client import AnthropicClient
from .api_key_env import get_api_key_env
from .cloudflare_client import CloudflareClient
from .google_client import GoogleClient
from .openai_client import OpenAIClient


@dataclass(frozen=True)
class ProviderSpec:
    client_cls: type
    api_key_env: str | None
    models: tuple[str, ...]


PROVIDERS: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(OpenAIClient, get_api_key_env("openai"), ("gpt-5.4", "gpt-5.4-mini", "gpt-5.3")),
    "anthropic": ProviderSpec(AnthropicClient, get_api_key_env("anthropic"), ("claude-sonnet-4-6", "claude-opus-4-6")),
    "google": ProviderSpec(GoogleClient, get_api_key_env("google"), ("gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-3-flash-preview")),
    "gemini": ProviderSpec(GoogleClient, "GEMINI_API_KEY", ("gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-3-flash-preview")),
    "cloudflare": ProviderSpec(CloudflareClient, get_api_key_env("cloudflare"), ("@cf/zai-org/glm-4.7-flash", "@cf/zai-org/glm-5.2")),
}
