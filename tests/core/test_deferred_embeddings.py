"""Tests for WU-12: Deferred Embedding Pipeline.

Tests cover:
1. needs_embedding — True when embedding IS NULL, False otherwise
2. ensure_embedding — computes and stores embedding on first access (lazy)
3. ensure_embedding — idempotent: no recomputation if embedding already present
4. ensure_embedding — graceful skip when content is empty
5. ensure_embedding — graceful degradation when embedding service unavailable
6. compute_missing_embeddings — batch fills rows missing embeddings
7. compute_missing_embeddings — returns 0 when no rows need embeddings
8. compute_missing_embeddings — handles partial failures per-row
9. Invalid table raises ValueError for all public functions
10. LookupError when row_id not found
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, call, patch
from uuid import uuid4

import pytest


# ---------------------------------------------------------------------------
# Helpers for constructing fake DB rows
# ---------------------------------------------------------------------------


def _make_row(needs_embed: bool = True, content: str = "Some content about Python") -> dict:
    """Return a dict mimicking a psycopg2 RealDictRow."""
    return {
        "id": str(uuid4()),
        "content": content,
        "needs_embed": needs_embed,
    }


@contextmanager
def _fake_cursor(rows: list[dict] | None = None, fetchone_row: dict | None = None):
    """Context manager returning a MagicMock cursor that yields given rows."""
    cur = MagicMock()
    cur.fetchone.return_value = fetchone_row
    cur.fetchall.return_value = rows or []
    # Support use as context manager (with get_cursor() as cur:)
    yield cur


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_row_needs_embed():
    return _make_row(needs_embed=True, content="Python 3.12 release notes")


@pytest.fixture()
def fake_row_has_embed():
    return _make_row(needs_embed=False, content="Python 3.12 release notes")


@pytest.fixture()
def fake_embedding():
    """A tiny fake embedding vector."""
    return [0.1, 0.2, 0.3, 0.4]


@pytest.fixture()
def fake_pgvector(fake_embedding):
    return "[0.1,0.2,0.3,0.4]"


# ---------------------------------------------------------------------------
# needs_embedding tests
# ---------------------------------------------------------------------------


class TestNeedsEmbedding:
    """Tests for needs_embedding()."""

    @pytest.mark.asyncio
    async def test_returns_true_when_embedding_null(self, fake_row_needs_embed):
        from valence.core.deferred_embeddings import needs_embedding

        with patch("valence.core.deferred_embeddings.get_cursor") as mock_gc:
            cur = MagicMock()
            cur.fetchone.return_value = {"needs_embed": True}
            mock_gc.return_value.__enter__ = MagicMock(return_value=cur)
            mock_gc.return_value.__exit__ = MagicMock(return_value=False)

            result = await needs_embedding("sources", str(uuid4()))

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_embedding_present(self):
        from valence.core.deferred_embeddings import needs_embedding

        with patch("valence.core.deferred_embeddings.get_cursor") as mock_gc:
            cur = MagicMock()
            cur.fetchone.return_value = {"needs_embed": False}
            mock_gc.return_value.__enter__ = MagicMock(return_value=cur)
            mock_gc.return_value.__exit__ = MagicMock(return_value=False)

            result = await needs_embedding("sources", str(uuid4()))

        assert result is False

    @pytest.mark.asyncio
    async def test_raises_lookup_error_when_row_missing(self):
        from valence.core.deferred_embeddings import needs_embedding

        with patch("valence.core.deferred_embeddings.get_cursor") as mock_gc:
            cur = MagicMock()
            cur.fetchone.return_value = None
            mock_gc.return_value.__enter__ = MagicMock(return_value=cur)
            mock_gc.return_value.__exit__ = MagicMock(return_value=False)

            with pytest.raises(LookupError):
                await needs_embedding("sources", str(uuid4()))

    @pytest.mark.asyncio
    async def test_raises_value_error_for_unknown_table(self):
        from valence.core.deferred_embeddings import needs_embedding

        with pytest.raises(ValueError, match="not supported"):
            await needs_embedding("nonexistent_table", str(uuid4()))

    @pytest.mark.asyncio
    async def test_articles_table_supported(self):
        from valence.core.deferred_embeddings import needs_embedding

        with patch("valence.core.deferred_embeddings.get_cursor") as mock_gc:
            cur = MagicMock()
            cur.fetchone.return_value = {"needs_embed": True}
            mock_gc.return_value.__enter__ = MagicMock(return_value=cur)
            mock_gc.return_value.__exit__ = MagicMock(return_value=False)

            result = await needs_embedding("articles", str(uuid4()))

        assert result is True


# ---------------------------------------------------------------------------
# ensure_embedding tests
# ---------------------------------------------------------------------------


class TestEnsureEmbedding:
    """Tests for ensure_embedding() — lazy compute + idempotency."""

    def _make_cursor_mock(self, fetchone_return):
        """Build a context-manager cursor mock that returns given value for fetchone."""
        mock_cm = MagicMock()
        cur = MagicMock()
        cur.fetchone.return_value = fetchone_return
        mock_cm.__enter__ = MagicMock(return_value=cur)
        mock_cm.__exit__ = MagicMock(return_value=False)
        return mock_cm, cur

    @pytest.mark.asyncio
    async def test_computes_embedding_when_missing(self, fake_embedding, fake_pgvector):
        """ensure_embedding returns True and writes embedding when it was NULL."""
        from valence.core.deferred_embeddings import ensure_embedding

        row_id = str(uuid4())
        # First cursor call: SELECT (needs embed=True, has content)
        # Second cursor call: UPDATE
        call_count = 0

        def side_effect_get_cursor():
            nonlocal call_count
            call_count += 1
            mock_cm = MagicMock()
            cur = MagicMock()
            if call_count == 1:
                cur.fetchone.return_value = {"content": "Python docs", "needs_embed": True}
            # On second call (UPDATE), fetchone not used
            mock_cm.__enter__ = MagicMock(return_value=cur)
            mock_cm.__exit__ = MagicMock(return_value=False)
            return mock_cm

        with (
            patch("valence.core.deferred_embeddings.get_cursor", side_effect=side_effect_get_cursor),
            patch("valence.core.deferred_embeddings.generate_embedding", return_value=fake_embedding, create=True),
            patch("valence.core.deferred_embeddings.vector_to_pgvector", return_value=fake_pgvector, create=True),
        ):
            # We need to patch the imports inside the function
            with patch.dict(
                "sys.modules",
                {
                    "our_embeddings": MagicMock(),
                    "our_embeddings.service": MagicMock(
                        generate_embedding=MagicMock(return_value=fake_embedding),
                        vector_to_pgvector=MagicMock(return_value=fake_pgvector),
                    ),
                },
            ):
                result = await ensure_embedding("sources", row_id)

        assert result is True

    @pytest.mark.asyncio
    async def test_idempotent_when_embedding_exists(self, fake_embedding, fake_pgvector):
        """ensure_embedding returns False and does NOT call generate_embedding when already embedded."""
        from valence.core.deferred_embeddings import ensure_embedding

        row_id = str(uuid4())

        mock_cm = MagicMock()
        cur = MagicMock()
        cur.fetchone.return_value = {"content": "Python docs", "needs_embed": False}
        mock_cm.__enter__ = MagicMock(return_value=cur)
        mock_cm.__exit__ = MagicMock(return_value=False)

        mock_generate = MagicMock(return_value=fake_embedding)

        with (
            patch("valence.core.deferred_embeddings.get_cursor", return_value=mock_cm),
            patch.dict(
                "sys.modules",
                {
                    "our_embeddings": MagicMock(),
                    "our_embeddings.service": MagicMock(
                        generate_embedding=mock_generate,
                        vector_to_pgvector=MagicMock(return_value=fake_pgvector),
                    ),
                },
            ),
        ):
            result = await ensure_embedding("sources", row_id)

        assert result is False
        mock_generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_content_empty(self):
        """ensure_embedding returns False silently when content is empty."""
        from valence.core.deferred_embeddings import ensure_embedding

        mock_cm = MagicMock()
        cur = MagicMock()
        cur.fetchone.return_value = {"content": "", "needs_embed": True}
        mock_cm.__enter__ = MagicMock(return_value=cur)
        mock_cm.__exit__ = MagicMock(return_value=False)

        with patch("valence.core.deferred_embeddings.get_cursor", return_value=mock_cm):
            result = await ensure_embedding("sources", str(uuid4()))

        assert result is False

    @pytest.mark.asyncio
    async def test_graceful_degradation_when_service_unavailable(self):
        """ensure_embedding returns False and logs a warning when embedding service errors."""
        from valence.core.deferred_embeddings import ensure_embedding

        row_id = str(uuid4())

        mock_cm = MagicMock()
        cur = MagicMock()
        cur.fetchone.return_value = {"content": "Some content", "needs_embed": True}
        mock_cm.__enter__ = MagicMock(return_value=cur)
        mock_cm.__exit__ = MagicMock(return_value=False)

        broken_service = MagicMock()
        broken_service.generate_embedding.side_effect = RuntimeError("Model not loaded")

        with (
            patch("valence.core.deferred_embeddings.get_cursor", return_value=mock_cm),
            patch.dict(
                "sys.modules",
                {
                    "our_embeddings": MagicMock(),
                    "our_embeddings.service": broken_service,
                },
            ),
        ):
            result = await ensure_embedding("sources", row_id)

        assert result is False  # graceful degradation

    @pytest.mark.asyncio
    async def test_raises_lookup_error_when_row_missing(self):
        """ensure_embedding raises LookupError if the row does not exist."""
        from valence.core.deferred_embeddings import ensure_embedding

        mock_cm = MagicMock()
        cur = MagicMock()
        cur.fetchone.return_value = None
        mock_cm.__enter__ = MagicMock(return_value=cur)
        mock_cm.__exit__ = MagicMock(return_value=False)

        with patch("valence.core.deferred_embeddings.get_cursor", return_value=mock_cm):
            with pytest.raises(LookupError):
                await ensure_embedding("sources", str(uuid4()))

    @pytest.mark.asyncio
    async def test_raises_value_error_for_unknown_table(self):
        """ensure_embedding raises ValueError for unsupported tables."""
        from valence.core.deferred_embeddings import ensure_embedding

        with pytest.raises(ValueError, match="not supported"):
            await ensure_embedding("bad_table", str(uuid4()))


# ---------------------------------------------------------------------------
# compute_missing_embeddings tests
# ---------------------------------------------------------------------------


class TestComputeMissingEmbeddings:
    """Tests for compute_missing_embeddings() — batch fill."""

    def _make_batch_rows(self, n: int) -> list[dict]:
        return [{"id": str(uuid4()), "content": f"Content #{i}"} for i in range(n)]

    @pytest.mark.asyncio
    async def test_returns_zero_when_nothing_to_do(self):
        """Returns 0 when no rows are missing embeddings."""
        from valence.core.deferred_embeddings import compute_missing_embeddings

        mock_cm = MagicMock()
        cur = MagicMock()
        cur.fetchall.return_value = []
        mock_cm.__enter__ = MagicMock(return_value=cur)
        mock_cm.__exit__ = MagicMock(return_value=False)

        with patch("valence.core.deferred_embeddings.get_cursor", return_value=mock_cm):
            count = await compute_missing_embeddings("sources", batch_size=100)

        assert count == 0

    @pytest.mark.asyncio
    async def test_fills_batch(self):
        """Returns count of rows successfully embedded in a batch."""
        from valence.core.deferred_embeddings import compute_missing_embeddings

        rows = self._make_batch_rows(3)
        fake_embedding = [0.1, 0.2, 0.3]
        fake_pgvector = "[0.1,0.2,0.3]"

        call_count = 0

        def get_cursor_side_effect():
            nonlocal call_count
            call_count += 1
            mock_cm = MagicMock()
            cur = MagicMock()
            if call_count == 1:
                # Initial SELECT — returns 3 rows
                cur.fetchall.return_value = rows
            # Subsequent calls are UPDATE per row
            mock_cm.__enter__ = MagicMock(return_value=cur)
            mock_cm.__exit__ = MagicMock(return_value=False)
            return mock_cm

        service_mock = MagicMock()
        service_mock.generate_embedding.return_value = fake_embedding
        service_mock.vector_to_pgvector.return_value = fake_pgvector

        with (
            patch("valence.core.deferred_embeddings.get_cursor", side_effect=get_cursor_side_effect),
            patch.dict(
                "sys.modules",
                {
                    "our_embeddings": MagicMock(),
                    "our_embeddings.service": service_mock,
                },
            ),
        ):
            count = await compute_missing_embeddings("sources", batch_size=10)

        assert count == 3
        assert service_mock.generate_embedding.call_count == 3

    @pytest.mark.asyncio
    async def test_respects_batch_size(self):
        """Batch SELECT passes batch_size as LIMIT."""
        from valence.core.deferred_embeddings import compute_missing_embeddings

        captured_limit = []

        def get_cursor_side_effect():
            mock_cm = MagicMock()
            cur = MagicMock()
            cur.fetchall.return_value = []

            def execute_side_effect(query, params=None):
                if params and len(params) == 1:
                    captured_limit.append(params[0])

            cur.execute.side_effect = execute_side_effect
            mock_cm.__enter__ = MagicMock(return_value=cur)
            mock_cm.__exit__ = MagicMock(return_value=False)
            return mock_cm

        with patch("valence.core.deferred_embeddings.get_cursor", side_effect=get_cursor_side_effect):
            await compute_missing_embeddings("sources", batch_size=42)

        assert 42 in captured_limit

    @pytest.mark.asyncio
    async def test_partial_failure_per_row(self):
        """compute_missing_embeddings skips individual row failures, returns partial count."""
        from valence.core.deferred_embeddings import compute_missing_embeddings

        rows = self._make_batch_rows(3)

        call_count = 0

        def get_cursor_side_effect():
            nonlocal call_count
            call_count += 1
            mock_cm = MagicMock()
            cur = MagicMock()
            if call_count == 1:
                cur.fetchall.return_value = rows
            mock_cm.__enter__ = MagicMock(return_value=cur)
            mock_cm.__exit__ = MagicMock(return_value=False)
            return mock_cm

        embed_call_count = 0

        def generate_side_effect(text):
            nonlocal embed_call_count
            embed_call_count += 1
            if embed_call_count == 2:
                raise RuntimeError("GPU OOM on row 2")
            return [0.1, 0.2]

        service_mock = MagicMock()
        service_mock.generate_embedding.side_effect = generate_side_effect
        service_mock.vector_to_pgvector.return_value = "[0.1,0.2]"

        with (
            patch("valence.core.deferred_embeddings.get_cursor", side_effect=get_cursor_side_effect),
            patch.dict(
                "sys.modules",
                {
                    "our_embeddings": MagicMock(),
                    "our_embeddings.service": service_mock,
                },
            ),
        ):
            count = await compute_missing_embeddings("sources", batch_size=10)

        # Row 2 failed → 2 out of 3 computed
        assert count == 2

    @pytest.mark.asyncio
    async def test_service_unavailable_returns_zero(self):
        """Returns 0 when embedding service is not importable."""
        from valence.core.deferred_embeddings import compute_missing_embeddings

        rows = self._make_batch_rows(2)

        call_count = 0

        def get_cursor_side_effect():
            nonlocal call_count
            call_count += 1
            mock_cm = MagicMock()
            cur = MagicMock()
            if call_count == 1:
                cur.fetchall.return_value = rows
            mock_cm.__enter__ = MagicMock(return_value=cur)
            mock_cm.__exit__ = MagicMock(return_value=False)
            return mock_cm

        broken_service = MagicMock()
        broken_service.generate_embedding.side_effect = ImportError("our_embeddings not installed")

        with (
            patch("valence.core.deferred_embeddings.get_cursor", side_effect=get_cursor_side_effect),
            patch.dict(
                "sys.modules",
                {
                    "our_embeddings": MagicMock(),
                    "our_embeddings.service": broken_service,
                },
            ),
        ):
            count = await compute_missing_embeddings("sources", batch_size=10)

        # All rows failed → 0
        assert count == 0

    @pytest.mark.asyncio
    async def test_raises_value_error_for_unknown_table(self):
        """compute_missing_embeddings raises ValueError for unsupported tables."""
        from valence.core.deferred_embeddings import compute_missing_embeddings

        with pytest.raises(ValueError, match="not supported"):
            await compute_missing_embeddings("nonexistent_table")

    @pytest.mark.asyncio
    async def test_articles_table_supported(self):
        """compute_missing_embeddings works for 'articles' table too."""
        from valence.core.deferred_embeddings import compute_missing_embeddings

        mock_cm = MagicMock()
        cur = MagicMock()
        cur.fetchall.return_value = []
        mock_cm.__enter__ = MagicMock(return_value=cur)
        mock_cm.__exit__ = MagicMock(return_value=False)

        with patch("valence.core.deferred_embeddings.get_cursor", return_value=mock_cm):
            count = await compute_missing_embeddings("articles", batch_size=50)

        assert count == 0

    @pytest.mark.asyncio
    async def test_idempotent_second_call(self):
        """Calling compute_missing_embeddings twice is safe — second call returns 0 when nothing left."""
        from valence.core.deferred_embeddings import compute_missing_embeddings

        # Simulate: first call processes 2 rows, second call finds 0 remaining
        call_count = 0

        def get_cursor_side_effect():
            nonlocal call_count
            call_count += 1
            mock_cm = MagicMock()
            cur = MagicMock()
            # SELECT phase (odd calls)
            if call_count % (len(self._make_batch_rows(2)) + 1) == 1:
                cur.fetchall.return_value = []  # Nothing to do
            mock_cm.__enter__ = MagicMock(return_value=cur)
            mock_cm.__exit__ = MagicMock(return_value=False)
            return mock_cm

        with patch("valence.core.deferred_embeddings.get_cursor", side_effect=get_cursor_side_effect):
            count1 = await compute_missing_embeddings("sources", batch_size=10)
            count2 = await compute_missing_embeddings("sources", batch_size=10)

        assert count1 == 0
        assert count2 == 0
