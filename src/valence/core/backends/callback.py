# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""HTTP callback inference backend for Valence.

Delegates inference to an external HTTP endpoint. The callback URL receives
a JSON POST with the prompt and returns the LLM response. This enables
platforms like OpenClaw to provide inference through their own model
providers without Valence needing direct API key access.

Contract:
  Request:  POST callback_url {"prompt": "...", "system": "..."}
  Response: {"text": "..."}
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from typing import Any

logger = logging.getLogger(__name__)


def create_callback_backend(
    callback_url: str,
    token: str | None = None,
    timeout: float = 120.0,
) -> Callable[[str], Coroutine[Any, Any, str]]:
    """Return an async callable suitable for ``InferenceProvider.configure()``.

    Args:
        callback_url: HTTP(S) URL to POST inference requests to.
        token: Optional Bearer token for authentication.
        timeout: Request timeout in seconds. Defaults to 120s (compilation
            can be slow depending on the upstream model).

    Returns:
        Async callable ``(prompt: str) -> str``.

    Raises:
        RuntimeError: At call time, if the callback returns non-200 or
            missing ``text`` field.
        httpx.TimeoutException: At call time, if the request exceeds timeout.
    """
    import httpx

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    async def backend(prompt: str) -> str:
        """POST prompt to callback URL, return response text."""
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                callback_url,
                json={"prompt": prompt, "system": ""},
                headers=headers,
            )

        if response.status_code != 200:
            body = response.text[:500]
            raise RuntimeError(
                f"Callback backend ({callback_url}) returned HTTP {response.status_code}: {body}"
            )

        data = response.json()
        text = data.get("text")
        if text is None:
            raise RuntimeError(
                f"Callback backend ({callback_url}) response missing 'text' field: {data!r}"
            )

        logger.debug(
            "Callback backend: received %d chars from %s",
            len(text),
            callback_url,
        )
        return text

    backend.__name__ = f"callback_backend({callback_url})"  # type: ignore[attr-defined]
    backend._callback_url = callback_url  # type: ignore[attr-defined]
    return backend
