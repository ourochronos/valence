"""Tests for WU-18 LLM backends (Gemini CLI, OpenAI-compat, Cerebras, Ollama).

All tests use mocked subprocess / HTTP calls — no real LLM required.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Gemini CLI backend tests
# ---------------------------------------------------------------------------


class TestGeminiBackend:
    """Tests for create_gemini_backend()."""

    def test_create_returns_callable(self):
        from valence.core.backends.gemini_cli import create_gemini_backend

        backend = create_gemini_backend()
        assert callable(backend)

    def test_custom_model_stored(self):
        from valence.core.backends.gemini_cli import create_gemini_backend

        backend = create_gemini_backend(model="gemini-2.5-pro")
        assert backend._model == "gemini-2.5-pro"

    @pytest.mark.asyncio
    async def test_successful_call_returns_stdout(self):
        from valence.core.backends.gemini_cli import create_gemini_backend

        backend = create_gemini_backend()

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b'{"result": "ok"}', b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await backend("test prompt")

        assert result == '{"result": "ok"}'

    @pytest.mark.asyncio
    async def test_prompt_sent_via_stdin_not_arg(self):
        """Security test: prompt must go to stdin, not as a CLI argument."""
        from valence.core.backends.gemini_cli import create_gemini_backend

        backend = create_gemini_backend()
        captured_args: list = []
        captured_input: list = []

        mock_proc = AsyncMock()
        mock_proc.returncode = 0

        async def fake_communicate(input=None):
            captured_input.append(input)
            return (b"response text", b"")

        mock_proc.communicate = fake_communicate

        async def fake_create_subprocess(*args, **kwargs):
            captured_args.extend(args)
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_create_subprocess):
            await backend("sensitive user prompt with injection; rm -rf /")

        # The prompt must NOT appear in the subprocess args (shell injection risk)
        for arg in captured_args:
            assert "sensitive user prompt" not in str(arg), f"Prompt found in CLI argument {arg!r} — shell injection risk!"

        # The prompt MUST have been sent via stdin
        assert len(captured_input) == 1
        assert captured_input[0] is not None
        assert b"sensitive user prompt" in captured_input[0]

    @pytest.mark.asyncio
    async def test_nonzero_exit_raises_runtime_error(self):
        from valence.core.backends.gemini_cli import create_gemini_backend

        backend = create_gemini_backend()

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"Error: quota exceeded"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="Gemini CLI failed"):
                await backend("test prompt")

    @pytest.mark.asyncio
    async def test_timeout_kills_process_and_raises(self):
        from valence.core.backends.gemini_cli import create_gemini_backend

        backend = create_gemini_backend(timeout=0.001)

        mock_proc = AsyncMock()
        mock_proc.returncode = None
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        async def slow_communicate(input=None):
            await asyncio.sleep(10)
            return (b"", b"")

        mock_proc.communicate = slow_communicate

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(asyncio.TimeoutError):
                await backend("test prompt")

        mock_proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_gemini_bin_not_found_raises_file_not_found(self):
        from valence.core.backends.gemini_cli import create_gemini_backend

        backend = create_gemini_backend(gemini_bin="/nonexistent/gemini")

        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError, match="not found"):
                await backend("test prompt")

    @pytest.mark.asyncio
    async def test_unicode_prompt_encoded_as_utf8(self):
        from valence.core.backends.gemini_cli import create_gemini_backend

        backend = create_gemini_backend()
        received_input: list = []

        mock_proc = AsyncMock()
        mock_proc.returncode = 0

        async def capture_communicate(input=None):
            received_input.append(input)
            return (b"ok", b"")

        mock_proc.communicate = capture_communicate

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await backend("こんにちは世界")  # Japanese: "Hello World"

        assert received_input[0] == "こんにちは世界".encode()


# ---------------------------------------------------------------------------
# OpenAI-compatible backend tests
# ---------------------------------------------------------------------------


class TestOpenAICompatBackend:
    """Tests for create_openai_backend()."""

    def test_create_returns_callable(self):
        from valence.core.backends.openai_compat import create_openai_backend

        backend = create_openai_backend(
            base_url="http://localhost:11434/v1",
            api_key="test-key",
            model="llama3",
        )
        assert callable(backend)

    def test_metadata_attributes(self):
        from valence.core.backends.openai_compat import create_openai_backend

        backend = create_openai_backend(
            base_url="https://api.cerebras.ai/v1",
            api_key="key",
            model="llama3",
        )
        assert backend._model == "llama3"
        assert backend._base_url == "https://api.cerebras.ai/v1"

    @pytest.mark.asyncio
    async def test_successful_call_returns_content(self):
        from valence.core.backends.openai_compat import create_openai_backend

        backend = create_openai_backend(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model="qwen3:30b",
        )

        fake_response = MagicMock()
        fake_response.choices = [MagicMock()]
        fake_response.choices[0].message.content = '{"answer": 42}'

        with patch("openai.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=fake_response)

            # Re-import to pick up patched AsyncOpenAI
            import importlib

            from valence.core.backends import openai_compat as _oc

            importlib.reload(_oc)

            # Create fresh backend after patching
            backend2 = _oc.create_openai_backend(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
                model="qwen3:30b",
            )
            result = await backend2("test prompt")

        assert result == '{"answer": 42}'

    @pytest.mark.asyncio
    async def test_empty_content_raises_runtime_error(self):
        import importlib

        from valence.core.backends import openai_compat as _oc

        fake_response = MagicMock()
        fake_response.choices = [MagicMock()]
        fake_response.choices[0].message.content = None

        with patch("openai.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=fake_response)

            importlib.reload(_oc)
            backend = _oc.create_openai_backend(base_url="http://x/v1", api_key="k", model="m")
            with pytest.raises(RuntimeError, match="empty content"):
                await backend("test")


# ---------------------------------------------------------------------------
# Cerebras convenience wrapper tests
# ---------------------------------------------------------------------------


class TestCerebrasBackend:
    def test_create_returns_callable(self):
        from valence.core.backends.cerebras import create_cerebras_backend

        backend = create_cerebras_backend(api_key="test-key")
        assert callable(backend)

    def test_uses_cerebras_base_url(self):
        from valence.core.backends.cerebras import create_cerebras_backend

        backend = create_cerebras_backend(api_key="test-key", model="llama3.1-8b")
        assert "cerebras" in backend._base_url
        assert backend._model == "llama3.1-8b"

    def test_provider_tag(self):
        from valence.core.backends.cerebras import create_cerebras_backend

        backend = create_cerebras_backend(api_key="k")
        assert backend._provider == "cerebras"


# ---------------------------------------------------------------------------
# Ollama convenience wrapper tests
# ---------------------------------------------------------------------------


class TestOllamaBackend:
    def test_create_returns_callable(self):
        from valence.core.backends.ollama import create_ollama_backend

        backend = create_ollama_backend()
        assert callable(backend)

    def test_uses_ollama_base_url(self):
        from valence.core.backends.ollama import create_ollama_backend

        backend = create_ollama_backend(host="http://localhost:11434", model="qwen3:30b")
        assert "11434" in backend._base_url
        assert backend._model == "qwen3:30b"

    def test_custom_host(self):
        from valence.core.backends.ollama import create_ollama_backend

        backend = create_ollama_backend(host="http://100.127.143.21:11434")
        assert "100.127.143.21" in backend._base_url

    def test_provider_tag(self):
        from valence.core.backends.ollama import create_ollama_backend

        backend = create_ollama_backend()
        assert backend._provider == "ollama"

    def test_trailing_slash_stripped_from_host(self):
        from valence.core.backends.ollama import create_ollama_backend

        backend = create_ollama_backend(host="http://localhost:11434/")
        # Should not have double slash: .../v1 not ...//v1
        assert "//" not in backend._base_url.replace("http://", "").replace("https://", "")


# ---------------------------------------------------------------------------
# Integration with InferenceProvider
# ---------------------------------------------------------------------------


class TestBackendWithInferenceProvider:
    """Verify backends integrate correctly with InferenceProvider."""

    @pytest.mark.asyncio
    async def test_gemini_backend_integrates_with_provider(self):
        from valence.core.backends.gemini_cli import create_gemini_backend
        from valence.core.inference import TASK_CLASSIFY, InferenceProvider

        provider = InferenceProvider()
        backend = create_gemini_backend()

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        classify_response = json.dumps(
            {
                "relationship": "confirms",
                "confidence": 0.9,
                "reasoning": "The source confirms the article.",
            }
        )
        mock_proc.communicate = AsyncMock(return_value=(classify_response.encode(), b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            provider.configure(backend)
            result = await provider.infer(TASK_CLASSIFY, "classify this")

        assert not result.degraded
        assert result.parsed is not None
        assert result.parsed["relationship"] == "confirms"
        assert result.parsed["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_backend_error_sets_degraded(self):
        from valence.core.backends.gemini_cli import create_gemini_backend
        from valence.core.inference import TASK_CLASSIFY, InferenceProvider

        provider = InferenceProvider()
        backend = create_gemini_backend()

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"CLI error"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            provider.configure(backend)
            result = await provider.infer(TASK_CLASSIFY, "classify this")

        assert result.degraded
        assert result.error is not None
        assert "Backend error" in result.error

    @pytest.mark.asyncio
    async def test_provider_available_after_configure(self):
        from valence.core.backends.gemini_cli import create_gemini_backend
        from valence.core.inference import InferenceProvider

        provider = InferenceProvider()
        assert not provider.available

        backend = create_gemini_backend()
        provider.configure(backend)
        assert provider.available

    @pytest.mark.asyncio
    async def test_provider_reset_on_configure_none(self):
        from valence.core.backends.gemini_cli import create_gemini_backend
        from valence.core.inference import InferenceProvider

        provider = InferenceProvider()
        backend = create_gemini_backend()
        provider.configure(backend)
        assert provider.available

        provider.configure(None)
        assert not provider.available
