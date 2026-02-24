# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""OpenAI-compatible HTTP API backend for Valence inference.

Works with any provider that implements the OpenAI chat-completions API:
Cerebras, Ollama (/v1 endpoint), Together AI, Fireworks, etc.

Requires the ``openai`` Python package (already installed in the venv).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from typing import Any

logger = logging.getLogger(__name__)


def create_openai_backend(
    base_url: str,
    api_key: str,
    model: str,
    timeout: float = 60.0,
    system_prompt: str | None = None,
) -> Callable[[str], Coroutine[Any, Any, str]]:
    """Return an async callable suitable for ``InferenceProvider.configure()``.

    Sends the prompt as the user message in a chat-completions request.  The
    response text is extracted from ``choices[0].message.content`` and returned
    as a string.

    Args:
        base_url: API root URL, e.g. ``"https://api.cerebras.ai/v1"`` or
            ``"http://localhost:11434/v1"`` for Ollama.
        api_key: API key.  For providers that don't require one (e.g. Ollama),
            pass any non-empty string such as ``"ollama"``.
        model: Model identifier, e.g. ``"llama-4-scout-17b-16e-instruct"``.
        timeout: Request timeout in seconds.  Defaults to 60 s.
        system_prompt: Optional system message prepended before the user
            prompt.  Useful for telling the model to respond only with JSON.

    Returns:
        Async callable ``(prompt: str) -> str``.

    Raises:
        ImportError: If the ``openai`` package is not installed.
        openai.APIError: At call time, if the provider returns an error.
        asyncio.TimeoutError: At call time, if the request exceeds *timeout*.
    """
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ImportError("The 'openai' package is required for the OpenAI-compatible backend. Install it with: pip install openai") from exc

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
    )

    default_system = "You are a precise knowledge-management assistant. Respond ONLY with valid JSON; no markdown, no commentary."
    effective_system = system_prompt if system_prompt is not None else default_system

    async def backend(prompt: str) -> str:
        """Async backend: send *prompt* via chat-completions, return response text."""
        messages = [
            {"role": "system", "content": effective_system},
            {"role": "user", "content": prompt},
        ]

        response = await client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
        )

        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError(f"OpenAI-compat backend ({base_url}) returned empty content for model {model!r}")

        logger.debug(
            "OpenAI-compat backend: received %d chars (model=%s url=%s)",
            len(content),
            model,
            base_url,
        )
        return content

    backend.__name__ = f"openai_compat_backend({model}@{base_url})"  # type: ignore[attr-defined]
    backend._model = model  # type: ignore[attr-defined]
    backend._base_url = base_url  # type: ignore[attr-defined]
    return backend
