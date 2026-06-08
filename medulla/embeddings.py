"""Embedding provider abstraction.

Default: SentenceTransformersProvider with intfloat/e5-base-v2 (768-dim, local).
Model is lazy-loaded on first embed() call so startup stays fast.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    dimension: int
    model_name: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...


class SentenceTransformersProvider:
    """Local embedding model via sentence-transformers. Downloaded on first use."""

    dimension = 768
    model_name = "intfloat/e5-base-v2"

    def __init__(self, model: str = "intfloat/e5-base-v2") -> None:
        self.model_name = model
        self._model = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            import os
            ssl_cert = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
            if ssl_cert:
                # huggingface_hub creates httpx.Client without a verify= argument,
                # so it bypasses SSL_CERT_FILE and certifi patching. Inject our CA
                # bundle by patching httpx.Client.__init__ before the first download.
                import httpx
                _orig_init = httpx.Client.__init__
                def _patched_init(self_, *args, **kwargs):
                    kwargs.setdefault("verify", ssl_cert)
                    _orig_init(self_, *args, **kwargs)
                httpx.Client.__init__ = _patched_init
                os.environ.setdefault("REQUESTS_CA_BUNDLE", ssl_cert)
                os.environ.setdefault("CURL_CA_BUNDLE", ssl_cert)
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model.encode(texts, convert_to_numpy=True).tolist()


def get_embedding_provider() -> EmbeddingProvider:
    """Return the configured embedding provider (sentence-transformers by default)."""
    return SentenceTransformersProvider()
