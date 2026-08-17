import os
from typing import Any

from .base_client import BaseLLMClient
from .openai_client import NormalizedChatOpenAI


class CloudflareClient(BaseLLMClient):
    """OpenAI-compatible client for Cloudflare Workers AI."""

    provider = "cloudflare"

    def get_llm(self) -> Any:
        account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        api_key = os.getenv("CLOUDFLARE_API_KEY")
        if not account_id:
            raise RuntimeError("CLOUDFLARE_ACCOUNT_ID is not configured")
        if not api_key:
            raise RuntimeError("CLOUDFLARE_API_KEY is not configured")

        base_url = self.base_url or (
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
        )
        kwargs = {
            "model": self.model,
            "api_key": api_key,
            "base_url": base_url,
        }
        for key in ("timeout", "max_retries", "temperature"):
            if key in self.kwargs:
                kwargs[key] = self.kwargs[key]
        return NormalizedChatOpenAI(**kwargs)

    def validate_model(self) -> bool:
        # Cloudflare model IDs are vendor-defined and may evolve independently
        # of this package's static catalog.
        return bool(self.model)
