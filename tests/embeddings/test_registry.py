"""Tests for valence.embeddings.registry module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest


# ============================================================================
# EmbeddingType Tests
# ============================================================================

class TestEmbeddingType:
    """Tests for EmbeddingType dataclass."""

    def test_create(self):
        """Should create embedding type."""
        from valence.embeddings.registry import EmbeddingType

        et = EmbeddingType(
            id="openai_text3_small",
            provider="openai",
            model="text-embedding-3-small",
            dimensions=1536,
            is_default=True,
            status="active",
        )

        assert et.id == "openai_text3_small"
        assert et.provider == "openai"
        assert et.model == "text-embedding-3-small"
        assert et.dimensions == 1536
        assert et.is_default is True
        assert et.status == "active"

    def test_default_values(self):
        """Should have correct defaults."""
        from valence.embeddings.registry import EmbeddingType

        et = EmbeddingType(
            id="test",
            provider="openai",
            model="test-model",
            dimensions=1536,
        )

        assert et.is_default is False
        assert et.status == "active"

    def test_to_dict(self):
        """Should serialize to dict."""
        from valence.embeddings.registry import EmbeddingType

        et = EmbeddingType(
            id="test",
            provider="openai",
            model="test-model",
            dimensions=1536,
            is_default=True,
        )

        d = et.to_dict()

        assert d["id"] == "test"
        assert d["provider"] == "openai"
        assert d["model"] == "test-model"
        assert d["dimensions"] == 1536
        assert d["is_default"] is True
        assert d["status"] == "active"

    def test_from_row(self):
        """Should create from database row."""
        from valence.embeddings.registry import EmbeddingType

        row = {
            "id": "openai_text3_small",
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dimensions": 1536,
            "is_default": True,
            "status": "active",
        }

        et = EmbeddingType.from_row(row)

        assert et.id == "openai_text3_small"
        assert et.provider == "openai"
        assert et.is_default is True


# ============================================================================
# get_embedding_type Tests
# ============================================================================

class TestGetEmbeddingType:
    """Tests for get_embedding_type function."""

    @pytest.fixture
    def mock_cursor(self):
        """Create mock cursor."""
        cursor = MagicMock()
        return cursor

    @pytest.fixture
    def mock_get_cursor(self, mock_cursor):
        """Mock get_cursor context manager."""
        from contextlib import contextmanager

        @contextmanager
        def fake_get_cursor(dict_cursor=True):
            yield mock_cursor

        with patch("valence.embeddings.registry.get_cursor", fake_get_cursor):
            yield mock_cursor

    def test_by_id(self, mock_get_cursor):
        """Should get embedding type by ID."""
        from valence.embeddings.registry import get_embedding_type

        mock_get_cursor.fetchone.return_value = {
            "id": "openai_text3_small",
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dimensions": 1536,
            "is_default": False,
            "status": "active",
        }

        result = get_embedding_type("openai_text3_small")

        assert result is not None
        assert result.id == "openai_text3_small"
        # Should query by ID
        call_args = mock_get_cursor.execute.call_args[0]
        assert "id = %s" in call_args[0]

    def test_default(self, mock_get_cursor):
        """Should get default embedding type when no ID provided."""
        from valence.embeddings.registry import get_embedding_type

        mock_get_cursor.fetchone.return_value = {
            "id": "default_type",
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dimensions": 1536,
            "is_default": True,
            "status": "active",
        }

        result = get_embedding_type()

        assert result is not None
        assert result.is_default is True
        # Should query for default
        call_args = mock_get_cursor.execute.call_args[0]
        assert "is_default = TRUE" in call_args[0]

    def test_not_found(self, mock_get_cursor):
        """Should return None when not found."""
        from valence.embeddings.registry import get_embedding_type

        mock_get_cursor.fetchone.return_value = None

        result = get_embedding_type("nonexistent")

        assert result is None


# ============================================================================
# list_embedding_types Tests
# ============================================================================

class TestListEmbeddingTypes:
    """Tests for list_embedding_types function."""

    @pytest.fixture
    def mock_cursor(self):
        """Create mock cursor."""
        cursor = MagicMock()
        return cursor

    @pytest.fixture
    def mock_get_cursor(self, mock_cursor):
        """Mock get_cursor context manager."""
        from contextlib import contextmanager

        @contextmanager
        def fake_get_cursor(dict_cursor=True):
            yield mock_cursor

        with patch("valence.embeddings.registry.get_cursor", fake_get_cursor):
            yield mock_cursor

    def test_list_all(self, mock_get_cursor):
        """Should list all embedding types."""
        from valence.embeddings.registry import list_embedding_types

        mock_get_cursor.fetchall.return_value = [
            {
                "id": "type1",
                "provider": "openai",
                "model": "model1",
                "dimensions": 1536,
                "is_default": True,
                "status": "active",
            },
            {
                "id": "type2",
                "provider": "openai",
                "model": "model2",
                "dimensions": 3072,
                "is_default": False,
                "status": "active",
            },
        ]

        result = list_embedding_types()

        assert len(result) == 2
        assert result[0].id == "type1"
        assert result[1].id == "type2"

    def test_filter_by_status(self, mock_get_cursor):
        """Should filter by status."""
        from valence.embeddings.registry import list_embedding_types

        mock_get_cursor.fetchall.return_value = []

        result = list_embedding_types(status="active")

        # Should include status in query
        call_args = mock_get_cursor.execute.call_args[0]
        assert "status = %s" in call_args[0]


# ============================================================================
# register_embedding_type Tests
# ============================================================================

class TestRegisterEmbeddingType:
    """Tests for register_embedding_type function."""

    @pytest.fixture
    def mock_cursor(self):
        """Create mock cursor."""
        cursor = MagicMock()
        return cursor

    @pytest.fixture
    def mock_get_cursor(self, mock_cursor):
        """Mock get_cursor context manager."""
        from contextlib import contextmanager

        @contextmanager
        def fake_get_cursor(dict_cursor=True):
            yield mock_cursor

        with patch("valence.embeddings.registry.get_cursor", fake_get_cursor):
            yield mock_cursor

    def test_create_new(self, mock_get_cursor):
        """Should create new embedding type."""
        from valence.embeddings.registry import register_embedding_type

        mock_get_cursor.fetchone.return_value = {
            "id": "new_type",
            "provider": "openai",
            "model": "new-model",
            "dimensions": 1536,
            "is_default": False,
            "status": "active",
        }

        result = register_embedding_type(
            "new_type",
            "openai",
            "new-model",
            1536,
        )

        assert result.id == "new_type"

    def test_update_existing(self, mock_get_cursor):
        """Should update existing type via upsert."""
        from valence.embeddings.registry import register_embedding_type

        mock_get_cursor.fetchone.return_value = {
            "id": "existing_type",
            "provider": "openai",
            "model": "updated-model",
            "dimensions": 3072,
            "is_default": False,
            "status": "active",
        }

        result = register_embedding_type(
            "existing_type",
            "openai",
            "updated-model",
            3072,
        )

        assert result.model == "updated-model"
        # Should use ON CONFLICT
        call_args = mock_get_cursor.execute.call_args[0]
        assert "ON CONFLICT" in call_args[0]

    def test_set_as_default(self, mock_get_cursor):
        """Should unset other defaults when setting as default."""
        from valence.embeddings.registry import register_embedding_type

        mock_get_cursor.fetchone.return_value = {
            "id": "new_default",
            "provider": "openai",
            "model": "model",
            "dimensions": 1536,
            "is_default": True,
            "status": "active",
        }

        result = register_embedding_type(
            "new_default",
            "openai",
            "model",
            1536,
            is_default=True,
        )

        assert result.is_default is True
        # Should have updated existing defaults
        calls = mock_get_cursor.execute.call_args_list
        assert any("is_default = FALSE" in str(c) for c in calls)


# ============================================================================
# ensure_default_type Tests
# ============================================================================

class TestEnsureDefaultType:
    """Tests for ensure_default_type function."""

    @pytest.fixture
    def mock_cursor(self):
        """Create mock cursor."""
        cursor = MagicMock()
        return cursor

    @pytest.fixture
    def mock_get_cursor(self, mock_cursor):
        """Mock get_cursor context manager."""
        from contextlib import contextmanager

        @contextmanager
        def fake_get_cursor(dict_cursor=True):
            yield mock_cursor

        with patch("valence.embeddings.registry.get_cursor", fake_get_cursor):
            yield mock_cursor

    def test_returns_existing(self, mock_get_cursor):
        """Should return existing default."""
        from valence.embeddings.registry import ensure_default_type

        # get_embedding_type returns existing
        mock_get_cursor.fetchone.return_value = {
            "id": "existing_default",
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dimensions": 1536,
            "is_default": True,
            "status": "active",
        }

        with patch("valence.embeddings.registry.get_embedding_type") as mock_get:
            from valence.embeddings.registry import EmbeddingType
            mock_get.return_value = EmbeddingType(
                id="existing_default",
                provider="openai",
                model="text-embedding-3-small",
                dimensions=1536,
                is_default=True,
            )

            result = ensure_default_type()

            assert result.id == "existing_default"

    def test_creates_default(self, mock_get_cursor):
        """Should create default if none exists."""
        from valence.embeddings.registry import ensure_default_type

        # First call (get_embedding_type) returns None
        # Second call (register) returns the new type
        mock_get_cursor.fetchone.side_effect = [
            None,  # get_embedding_type returns None
            {  # register returns new type
                "id": "openai_text3_small",
                "provider": "openai",
                "model": "text-embedding-3-small",
                "dimensions": 1536,
                "is_default": True,
                "status": "active",
            },
        ]

        with patch("valence.embeddings.registry.get_embedding_type") as mock_get:
            mock_get.return_value = None

            with patch("valence.embeddings.registry.register_embedding_type") as mock_register:
                from valence.embeddings.registry import EmbeddingType
                mock_register.return_value = EmbeddingType(
                    id="openai_text3_small",
                    provider="openai",
                    model="text-embedding-3-small",
                    dimensions=1536,
                    is_default=True,
                )

                result = ensure_default_type()

                mock_register.assert_called_once_with(
                    type_id="openai_text3_small",
                    provider="openai",
                    model="text-embedding-3-small",
                    dimensions=1536,
                    is_default=True,
                )
                assert result.is_default is True


# ============================================================================
# KNOWN_EMBEDDINGS Tests
# ============================================================================

class TestKnownEmbeddings:
    """Tests for KNOWN_EMBEDDINGS constant."""

    def test_contains_text3_small(self):
        """Should contain text-embedding-3-small config."""
        from valence.embeddings.registry import KNOWN_EMBEDDINGS

        assert "openai_text3_small" in KNOWN_EMBEDDINGS
        config = KNOWN_EMBEDDINGS["openai_text3_small"]
        assert config["model"] == "text-embedding-3-small"
        assert config["dimensions"] == 1536

    def test_contains_text3_large(self):
        """Should contain text-embedding-3-large config."""
        from valence.embeddings.registry import KNOWN_EMBEDDINGS

        assert "openai_text3_large" in KNOWN_EMBEDDINGS
        config = KNOWN_EMBEDDINGS["openai_text3_large"]
        assert config["model"] == "text-embedding-3-large"
        assert config["dimensions"] == 3072

    def test_contains_ada(self):
        """Should contain ada-002 config."""
        from valence.embeddings.registry import KNOWN_EMBEDDINGS

        assert "openai_ada_002" in KNOWN_EMBEDDINGS
        config = KNOWN_EMBEDDINGS["openai_ada_002"]
        assert config["model"] == "text-embedding-ada-002"
        assert config["dimensions"] == 1536
