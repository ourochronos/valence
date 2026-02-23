"""Cerebras Cloud backend for Valence inference.

Cerebras offers extremely low latency (< 1 s for classification tasks) via
their custom Wafer-Scale Engine hardware.  Uses the OpenAI-compatible API.

Obtain an API key at https://cloud.cerebras.ai/.
"""

from __future__ import annotations

from typing import Callable, Coroutine, Any


def create_cerebras_backend(
    api_key: str,
    model: str = "llama-4-scout-17b-16e-instruct",
    timeout: float = 30.0,
) -> Callable[[str], Coroutine[Any, Any, str]]:
    """Return an async callable that routes inference through Cerebras Cloud.

    Args:
        api_key: Cerebras API key (from https://cloud.cerebras.ai/).
        model: Model to use.  Defaults to ``"llama-4-scout-17b-16e-instruct"``.
            Other options: ``"llama3.1-8b"``, ``"llama3.1-70b"``.
        timeout: Request timeout in seconds.  Defaults to 30 s (Cerebras is
            fast; 30 s is very generous even for large prompts).

    Returns:
        Async callable ``(prompt: str) -> str``.

    Example::

        import os
        from valence.core.backends.cerebras import create_cerebras_backend
        from valence.core.inference import provider

        provider.configure(create_cerebras_backend(api_key=os.environ["CEREBRAS_API_KEY"]))
    """
    from .openai_compat import create_openai_backend

    backend = create_openai_backend(
        base_url="https://api.cerebras.ai/v1",
        api_key=api_key,
        model=model,
        timeout=timeout,
    )
    backend.__name__ = f"cerebras_backend({model})"  # type: ignore[attr-defined]
    backend._model = model  # type: ignore[attr-defined]
    backend._provider = "cerebras"  # type: ignore[attr-defined]
    return backend
