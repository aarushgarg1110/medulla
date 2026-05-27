"""LLM provider abstraction — Bedrock, Anthropic, Ollama.

All providers implement one method: generate(prompt) -> str.
Active provider is set in config.toml via `medulla use <provider>`.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: str | None = None,
                 on_token: "Callable[[str], None] | None" = None) -> str:
        """Generate text from a prompt. Returns the full response string.

        on_token: optional callback called with each text chunk as it arrives.
                  When provided, Bedrock uses streaming. Others ignore it.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for display."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Model ID for display."""


class BedrockProvider(LLMProvider):
    def __init__(self, model: str, aws_profile: str, aws_region: str):
        self._model = model
        self._profile = aws_profile
        self._region = aws_region

    @property
    def name(self) -> str:
        return "bedrock"

    @property
    def model(self) -> str:
        return self._model

    def generate(self, prompt: str, system: str | None = None, on_token=None) -> str:
        import boto3
        session = boto3.Session(profile_name=self._profile, region_name=self._region)
        client = session.client("bedrock-runtime")
        body: dict = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
            "anthropic_version": "bedrock-2023-05-31",
        }
        if system:
            body["system"] = system

        if on_token is not None:
            resp = client.invoke_model_with_response_stream(
                modelId=self._model, body=json.dumps(body)
            )
            parts: list[str] = []
            for event in resp["body"]:
                chunk = json.loads(event["chunk"]["bytes"])
                if chunk.get("type") == "content_block_delta":
                    text = chunk.get("delta", {}).get("text", "")
                    if text:
                        parts.append(text)
                        on_token(text)
            return "".join(parts)

        resp = client.invoke_model(modelId=self._model, body=json.dumps(body))
        result = json.loads(resp["body"].read())
        return result["content"][0]["text"]


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str):
        self._model = model

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def model(self) -> str:
        return self._model

    def generate(self, prompt: str, system: str | None = None, on_token=None) -> str:
        import anthropic
        client = anthropic.Anthropic()
        kwargs: dict = {"model": self._model, "max_tokens": 4096, "messages": [{"role": "user", "content": prompt}]}
        if system:
            kwargs["system"] = system
        msg = client.messages.create(**kwargs)
        return msg.content[0].text


class OllamaProvider(LLMProvider):
    def __init__(self, model: str, host: str):
        self._model = model
        self._host = host.rstrip("/")

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return self._model

    def generate(self, prompt: str, system: str | None = None, on_token=None) -> str:
        import httpx
        import json as _json
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        if on_token:
            # Streaming — emit tokens as they arrive, no timeout risk
            full = []
            with httpx.stream(
                "POST",
                f"{self._host}/api/chat",
                json={"model": self._model, "messages": messages, "stream": True},
                timeout=600.0,
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    chunk = _json.loads(line)
                    text = chunk.get("message", {}).get("content", "")
                    if text:
                        on_token(text)
                        full.append(text)
            return "".join(full)

        # Non-streaming (no on_token) — longer timeout for big models
        response = httpx.post(
            f"{self._host}/api/chat",
            json={"model": self._model, "messages": messages, "stream": False},
            timeout=600.0,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


def get_provider() -> LLMProvider:
    """Return the active LLM provider based on config."""
    from medulla.config import get_config
    cfg = get_config()
    active = cfg.llm.active

    if active == "bedrock":
        bc = cfg.llm.bedrock
        return BedrockProvider(bc.model, bc.aws_profile, bc.aws_region)
    elif active == "anthropic":
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not set. Export it or switch provider:\n"
                "  medulla use bedrock\n"
                "  medulla use ollama"
            )
        return AnthropicProvider(cfg.llm.anthropic.model)
    elif active == "ollama":
        ol = cfg.llm.ollama
        return OllamaProvider(ol.model, ol.host)
    else:
        raise ValueError(f"Unknown provider: {active!r}. Choose: bedrock | anthropic | ollama")


def check_provider() -> tuple[bool, str]:
    """Check if the active provider is reachable. Returns (ok, message)."""
    try:
        provider = get_provider()
        # Minimal smoke test
        result = provider.generate("Reply with just: ok")
        return True, f"{provider.name} ({provider.model}) — reachable ✓"
    except EnvironmentError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Provider error: {e}"
