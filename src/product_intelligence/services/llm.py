"""Provider-agnostic LLM client (extension: real copilot summaries).

A thin, dependency-free (stdlib ``urllib``) client supporting OpenAI- and
Anthropic-style chat APIs behind one ``generate(system, prompt)`` interface.
``get_llm_client`` returns ``None`` when no provider/key is configured (or the
mock is forced), so callers fall back to the deterministic template - the repo
stays fully runnable offline while the real integration is wired and ready.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

from product_intelligence.core.config import Settings, settings
from product_intelligence.core.logging import get_logger

logger = get_logger(__name__)


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str: ...


def _post_json(url: str, headers: dict, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str, base_url: str | None, timeout: float) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        body = _post_json(
            f"{self.base_url}/chat/completions",
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            {"model": self.model, "messages": messages, "temperature": 0.2},
            self.timeout,
        )
        return body["choices"][0]["message"]["content"].strip()


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str, model: str, base_url: str | None, timeout: float) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = (base_url or "https://api.anthropic.com/v1").rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str, system: str | None = None) -> str:
        payload = {
            "model": self.model,
            "max_tokens": 512,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        body = _post_json(
            f"{self.base_url}/messages",
            {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            payload,
            self.timeout,
        )
        return body["content"][0]["text"].strip()


def get_llm_client(cfg: Settings | None = None) -> LLMClient | None:
    cfg = cfg or settings
    if cfg.copilot_use_mock or cfg.llm_provider == "mock" or not cfg.llm_api_key:
        return None
    if cfg.llm_provider == "openai":
        return OpenAIClient(
            cfg.llm_api_key, cfg.llm_model, cfg.llm_base_url, cfg.llm_timeout_seconds
        )
    if cfg.llm_provider == "anthropic":
        return AnthropicClient(
            cfg.llm_api_key, cfg.llm_model, cfg.llm_base_url, cfg.llm_timeout_seconds
        )
    return None
