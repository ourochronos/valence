"""Ollama local inference backend for Valence.

Ollama runs models locally (no API key, no cloud costs, full privacy).
Exposes an OpenAI-compatible API at ``/v1`` on the local host.

Install and start Ollama: https://ollama.ai/
Pull a model: ``ollama pull qwen3:30b``
"""

from __future__ import annotations


def create_ollama_backend(
    host: str = "http://localhost:11434",
    model: str = "qwen3:30b",
    timeout: float = 120.0,
) -> callable:
    """Return an async callable that routes inference through a local Ollama server.

    Args:
        host: Ollama server URL.  Defaults to ``"http://localhost:11434"``.
            For the derptop Ollama instance, use the Tailscale address:
            ``"http://100.127.143.21:11434"``.
        model: Model name as registered in Ollama.  Defaults to
            ``"qwen3:30b"``.  Check available models with ``ollama list``.
        timeout: Request timeout in seconds.  Defaults to 120 s — local
            models on GPU can be slow for compilation tasks.

    Returns:
        Async callable ``(prompt: str) -> str``.

    Example::

        from valence.core.backends.ollama import create_ollama_backend
        from valence.core.inference import provider

        # Local Ollama
        provider.configure(create_ollama_backend())

        # Remote Ollama on derptop
        provider.configure(create_ollama_backend(
            host="http://100.127.143.21:11434",
            model="qwen3:30b",
        ))
    """
    from .openai_compat import create_openai_backend

    backend = create_openai_backend(
        base_url=f"{host.rstrip('/')}/v1",
        api_key="ollama",  # Ollama doesn't validate the key
        model=model,
        timeout=timeout,
    )
    backend.__name__ = f"ollama_backend({model}@{host})"
    backend._model = model  # type: ignore[attr-defined]
    backend._host = host  # type: ignore[attr-defined]
    backend._provider = "ollama"  # type: ignore[attr-defined]
    return backend
