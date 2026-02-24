# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""LLM backend implementations for the Valence v2 inference layer.

Provides async callables suitable for ``InferenceProvider.configure()``.

Available backends:
    - ``gemini_cli``: Google Gemini via the ``gemini`` CLI subprocess (no API key needed).
    - ``openai_compat``: Any OpenAI-compatible HTTP API (Cerebras, Ollama, …).
    - ``cerebras``: Convenience wrapper for Cerebras Cloud (ultra-low latency).
    - ``ollama``: Convenience wrapper for local Ollama inference.

Usage::

    from valence.core.backends.gemini_cli import create_gemini_backend
    from valence.core.inference import provider

    provider.configure(create_gemini_backend())

Security note:
    Prompts are ALWAYS sent via stdin (never as CLI arguments) to prevent
    shell-injection attacks on the Gemini CLI backend.  The OpenAI-compat
    backend sends prompts in the HTTP request body.  Neither backend trusts
    fields in the LLM response outside the schema validated by
    ``inference.validate_output()``.
"""

from .cerebras import create_cerebras_backend
from .gemini_cli import create_gemini_backend
from .ollama import create_ollama_backend
from .openai_compat import create_openai_backend

__all__ = [
    "create_gemini_backend",
    "create_openai_backend",
    "create_cerebras_backend",
    "create_ollama_backend",
]
