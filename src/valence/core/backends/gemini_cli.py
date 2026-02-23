"""Gemini CLI subprocess backend for Valence inference.

Uses the ``gemini`` CLI tool (installed separately, already authenticated)
instead of API keys or SDK dependencies.

Security: prompts are piped via stdin — never passed as CLI arguments — to
prevent shell injection.
"""

from __future__ import annotations

import asyncio
import logging
import shutil

logger = logging.getLogger(__name__)


def create_gemini_backend(
    model: str = "gemini-2.5-flash",
    timeout: float = 60.0,
    gemini_bin: str = "gemini",
) -> callable:
    """Return an async callable suitable for ``InferenceProvider.configure()``.

    The returned backend sends prompts via **stdin** to the ``gemini`` CLI,
    preventing shell-injection attacks.

    Args:
        model: Gemini model name (e.g. ``"gemini-2.5-flash"``,
            ``"gemini-2.5-pro"``).
        timeout: Seconds to wait for the CLI to respond before raising
            ``asyncio.TimeoutError``.  Defaults to 60 s (generous for
            compilation tasks; classification tasks are typically much faster).
        gemini_bin: Path or name of the ``gemini`` binary.  Defaults to
            ``"gemini"`` (looked up on ``$PATH``).

    Returns:
        Async callable ``(prompt: str) -> str``.

    Raises:
        RuntimeError: At call time, if the CLI exits with a non-zero status.
        asyncio.TimeoutError: At call time, if the CLI takes longer than
            ``timeout`` seconds.
        FileNotFoundError: At call time, if the ``gemini`` binary is not found.
    """
    # Eagerly warn if binary isn't on PATH — fail-fast is friendlier than
    # mysterious RuntimeErrors at inference time.
    if shutil.which(gemini_bin) is None:
        logger.warning(
            "Gemini backend: binary %r not found on PATH. Install the Gemini CLI and ensure it is accessible.",
            gemini_bin,
        )

    async def backend(prompt: str) -> str:
        """Async backend: pipe *prompt* via stdin, return stdout text."""
        try:
            proc = await asyncio.create_subprocess_exec(
                gemini_bin,
                "-m",
                model,
                # No prompt arg — prevent shell injection.  The CLI reads from
                # stdin when no positional argument is provided.
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Gemini CLI binary {gemini_bin!r} not found. Install it with: pip install google-generativeai-cli") from exc

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=prompt.encode("utf-8")),
                timeout=timeout,
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise TimeoutError(f"Gemini CLI timed out after {timeout}s (model={model!r}, prompt_len={len(prompt)})")

        if proc.returncode != 0:
            err_msg = stderr.decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Gemini CLI failed (exit {proc.returncode}): {err_msg}")

        response = stdout.decode("utf-8", errors="replace")
        logger.debug("Gemini backend: received %d chars (model=%s)", len(response), model)
        return response

    # Attach metadata for introspection / repr
    backend.__name__ = f"gemini_backend({model})"
    backend._model = model
    backend._gemini_bin = gemini_bin
    return backend
