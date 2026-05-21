"""Tests for LLM provider abstraction — no real API calls."""
import pytest


def test_get_provider_bedrock(monkeypatch):
    import medulla.config as cfg_module
    cfg_module._config = None
    monkeypatch.setattr(cfg_module, "CONFIG_FILE", __import__("pathlib").Path("/nonexistent/config.toml"))
    from medulla.config import get_config
    cfg = get_config()
    cfg.llm.active = "bedrock"
    from medulla.llm import get_provider, BedrockProvider
    provider = get_provider()
    assert isinstance(provider, BedrockProvider)
    assert provider.name == "bedrock"


def test_get_provider_anthropic_no_key(monkeypatch):
    import medulla.config as cfg_module
    cfg_module._config = None
    monkeypatch.setattr(cfg_module, "CONFIG_FILE", __import__("pathlib").Path("/nonexistent"))
    from medulla.config import get_config
    cfg = get_config()
    cfg.llm.active = "anthropic"
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from medulla.llm import get_provider
    with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
        get_provider()


def test_get_provider_anthropic_with_key(monkeypatch):
    import medulla.config as cfg_module
    cfg_module._config = None
    monkeypatch.setattr(cfg_module, "CONFIG_FILE", __import__("pathlib").Path("/nonexistent"))
    from medulla.config import get_config
    cfg = get_config()
    cfg.llm.active = "anthropic"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
    from medulla.llm import get_provider, AnthropicProvider
    provider = get_provider()
    assert isinstance(provider, AnthropicProvider)
    assert provider.name == "anthropic"


def test_get_provider_ollama(monkeypatch):
    import medulla.config as cfg_module
    cfg_module._config = None
    monkeypatch.setattr(cfg_module, "CONFIG_FILE", __import__("pathlib").Path("/nonexistent"))
    from medulla.config import get_config
    cfg = get_config()
    cfg.llm.active = "ollama"
    from medulla.llm import get_provider, OllamaProvider
    provider = get_provider()
    assert isinstance(provider, OllamaProvider)
    assert provider.name == "ollama"


def test_get_provider_unknown_raises(monkeypatch):
    import medulla.config as cfg_module
    cfg_module._config = None
    monkeypatch.setattr(cfg_module, "CONFIG_FILE", __import__("pathlib").Path("/nonexistent"))
    from medulla.config import get_config
    cfg = get_config()
    cfg.llm.active = "unknown-provider"
    from medulla.llm import get_provider
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider()


def test_provider_name_and_model():
    from medulla.llm import BedrockProvider, AnthropicProvider, OllamaProvider
    b = BedrockProvider("model-a", "profile", "us-east-1")
    assert b.name == "bedrock"
    assert b.model == "model-a"

    a = AnthropicProvider("claude-haiku-4-5")
    assert a.name == "anthropic"
    assert a.model == "claude-haiku-4-5"

    o = OllamaProvider("llama3.2", "http://localhost:11434")
    assert o.name == "ollama"
    assert o.model == "llama3.2"


def test_bedrock_provider_generate(monkeypatch):
    """Mock boto3 to test BedrockProvider.generate()."""
    import json
    from medulla.llm import BedrockProvider

    mock_body = json.dumps({"content": [{"text": "test response"}]}).encode()

    class MockBody:
        def read(self): return mock_body

    class MockClient:
        def invoke_model(self, **kwargs):
            return {"body": MockBody()}

    class MockSession:
        def client(self, service): return MockClient()

    monkeypatch.setattr("boto3.Session", lambda **kw: MockSession())
    provider = BedrockProvider("anthropic.claude-haiku", "profile", "us-east-1")
    result = provider.generate("Hello", system="You are helpful.")
    assert result == "test response"


def test_anthropic_provider_generate(monkeypatch):
    """Mock anthropic SDK to test AnthropicProvider.generate()."""
    import os
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    class MockContent:
        text = "anthropic response"

    class MockMsg:
        content = [MockContent()]

    class MockMessages:
        def create(self, **kwargs): return MockMsg()

    class MockAnthropic:
        messages = MockMessages()

    monkeypatch.setattr("anthropic.Anthropic", lambda: MockAnthropic())
    from medulla.llm import AnthropicProvider
    provider = AnthropicProvider("claude-haiku-4-5")
    result = provider.generate("Hello", system="system prompt")
    assert result == "anthropic response"


def test_ollama_provider_generate(monkeypatch):
    """Mock httpx to test OllamaProvider.generate()."""
    class MockResponse:
        def raise_for_status(self): pass
        def json(self): return {"message": {"content": "ollama response"}}

    monkeypatch.setattr("httpx.post", lambda url, **kw: MockResponse())
    from medulla.llm import OllamaProvider
    provider = OllamaProvider("llama3.2", "http://localhost:11434")
    result = provider.generate("Hello", system="system")
    assert result == "ollama response"


def test_ollama_provider_generate_no_system(monkeypatch):
    """OllamaProvider without system prompt."""
    class MockResponse:
        def raise_for_status(self): pass
        def json(self): return {"message": {"content": "response"}}

    monkeypatch.setattr("httpx.post", lambda url, **kw: MockResponse())
    from medulla.llm import OllamaProvider
    provider = OllamaProvider("llama3.2", "http://localhost:11434")
    result = provider.generate("Hello")
    assert result == "response"
