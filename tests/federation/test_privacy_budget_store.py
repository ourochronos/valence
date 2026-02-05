"""Tests for privacy budget persistence stores.

Tests cover:
- InMemoryBudgetStore for in-memory testing
- FileBudgetStore for file-based persistence
- DatabaseBudgetStore for PostgreSQL persistence
- TopicBudget and RequesterBudget serialization
- Period reset handling
- Full round-trip save/load tests

These tests use mocked database connections to avoid requiring
a live PostgreSQL instance.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from valence.federation.privacy import (
    # Budget stores
    BudgetStore,
    DatabaseBudgetStore,
    FileBudgetStore,
    InMemoryBudgetStore,
    # Budget classes
    PrivacyBudget,
    RequesterBudget,
    TopicBudget,
    # Constants
    DEFAULT_DAILY_DELTA_BUDGET,
    DEFAULT_DAILY_EPSILON_BUDGET,
    MAX_QUERIES_PER_REQUESTER_PER_HOUR,
    MAX_QUERIES_PER_TOPIC_PER_DAY,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def federation_id() -> UUID:
    """Create a test federation ID."""
    return uuid4()


@pytest.fixture
def sample_budget_data(federation_id: UUID) -> dict[str, Any]:
    """Create sample budget data for testing."""
    now = datetime.now(UTC)
    return {
        "federation_id": str(federation_id),
        "daily_epsilon_budget": 10.0,
        "daily_delta_budget": 1e-4,
        "budget_period_hours": 24,
        "spent_epsilon": 2.5,
        "spent_delta": 1e-5,
        "queries_today": 15,
        "period_start": now.isoformat(),
        "topic_budgets": {
            "topic_hash_1": {
                "topic_hash": "topic_hash_1",
                "query_count": 2,
                "epsilon_spent": 1.0,
                "last_query": now.isoformat(),
            },
            "topic_hash_2": {
                "topic_hash": "topic_hash_2",
                "query_count": 1,
                "epsilon_spent": 0.5,
                "last_query": (now - timedelta(hours=1)).isoformat(),
            },
        },
        "requester_budgets": {
            "requester_1": {
                "requester_id": "requester_1",
                "queries_this_hour": 5,
                "hour_start": now.isoformat(),
            },
            "requester_2": {
                "requester_id": "requester_2",
                "queries_this_hour": 3,
                "hour_start": (now - timedelta(minutes=30)).isoformat(),
            },
        },
        "_version": 1,
    }


@pytest.fixture
def mock_async_pool() -> MagicMock:
    """Create a mock asyncpg connection pool."""
    pool = MagicMock()
    conn = AsyncMock()
    
    # Mock the async context manager for acquire
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    
    return pool


@pytest.fixture
def mock_sync_connection() -> MagicMock:
    """Create a mock psycopg2-style synchronous connection."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    return conn


# =============================================================================
# TOPIC BUDGET TESTS
# =============================================================================


class TestTopicBudget:
    """Tests for TopicBudget serialization."""
    
    def test_to_dict(self) -> None:
        """Test serializing TopicBudget to dictionary."""
        now = datetime.now(UTC)
        budget = TopicBudget(
            topic_hash="test_hash",
            query_count=3,
            epsilon_spent=1.5,
            last_query=now,
        )
        
        result = budget.to_dict()
        
        assert result["topic_hash"] == "test_hash"
        assert result["query_count"] == 3
        assert result["epsilon_spent"] == 1.5
        assert result["last_query"] == now.isoformat()
    
    def test_from_dict(self) -> None:
        """Test deserializing TopicBudget from dictionary."""
        now = datetime.now(UTC)
        data = {
            "topic_hash": "test_hash",
            "query_count": 5,
            "epsilon_spent": 2.0,
            "last_query": now.isoformat(),
        }
        
        budget = TopicBudget.from_dict(data)
        
        assert budget.topic_hash == "test_hash"
        assert budget.query_count == 5
        assert budget.epsilon_spent == 2.0
        assert budget.last_query.isoformat() == now.isoformat()
    
    def test_from_dict_missing_optional_fields(self) -> None:
        """Test deserializing with missing optional fields."""
        data = {"topic_hash": "minimal_hash"}
        
        budget = TopicBudget.from_dict(data)
        
        assert budget.topic_hash == "minimal_hash"
        assert budget.query_count == 0
        assert budget.epsilon_spent == 0.0
        # last_query defaults to datetime.now(UTC)
        assert budget.last_query is not None
    
    def test_can_query_under_limit(self) -> None:
        """Test can_query returns True when under limit."""
        budget = TopicBudget(topic_hash="test", query_count=1)
        assert budget.can_query() is True
    
    def test_can_query_at_limit(self) -> None:
        """Test can_query returns False at limit."""
        budget = TopicBudget(
            topic_hash="test",
            query_count=MAX_QUERIES_PER_TOPIC_PER_DAY,
        )
        assert budget.can_query() is False
    
    def test_record_query(self) -> None:
        """Test recording a query updates state."""
        budget = TopicBudget(topic_hash="test", query_count=0, epsilon_spent=0.0)
        before = budget.last_query
        
        budget.record_query(epsilon=1.0)
        
        assert budget.query_count == 1
        assert budget.epsilon_spent == 1.0
        assert budget.last_query >= before
    
    def test_round_trip_serialization(self) -> None:
        """Test full serialization round trip."""
        original = TopicBudget(
            topic_hash="round_trip_test",
            query_count=7,
            epsilon_spent=3.5,
        )
        
        data = original.to_dict()
        restored = TopicBudget.from_dict(data)
        
        assert restored.topic_hash == original.topic_hash
        assert restored.query_count == original.query_count
        assert restored.epsilon_spent == original.epsilon_spent


# =============================================================================
# REQUESTER BUDGET TESTS
# =============================================================================


class TestRequesterBudget:
    """Tests for RequesterBudget serialization."""
    
    def test_to_dict(self) -> None:
        """Test serializing RequesterBudget to dictionary."""
        now = datetime.now(UTC)
        budget = RequesterBudget(
            requester_id="test_requester",
            queries_this_hour=10,
            hour_start=now,
        )
        
        result = budget.to_dict()
        
        assert result["requester_id"] == "test_requester"
        assert result["queries_this_hour"] == 10
        assert result["hour_start"] == now.isoformat()
    
    def test_from_dict(self) -> None:
        """Test deserializing RequesterBudget from dictionary."""
        now = datetime.now(UTC)
        data = {
            "requester_id": "test_requester",
            "queries_this_hour": 15,
            "hour_start": now.isoformat(),
        }
        
        budget = RequesterBudget.from_dict(data)
        
        assert budget.requester_id == "test_requester"
        assert budget.queries_this_hour == 15
    
    def test_from_dict_missing_optional_fields(self) -> None:
        """Test deserializing with missing optional fields."""
        data = {"requester_id": "minimal_requester"}
        
        budget = RequesterBudget.from_dict(data)
        
        assert budget.requester_id == "minimal_requester"
        assert budget.queries_this_hour == 0
    
    def test_can_query_under_limit(self) -> None:
        """Test can_query returns True when under limit."""
        budget = RequesterBudget(requester_id="test", queries_this_hour=5)
        assert budget.can_query() is True
    
    def test_can_query_at_limit(self) -> None:
        """Test can_query returns False at limit."""
        budget = RequesterBudget(
            requester_id="test",
            queries_this_hour=MAX_QUERIES_PER_REQUESTER_PER_HOUR,
        )
        assert budget.can_query() is False
    
    def test_hourly_reset(self) -> None:
        """Test that budget resets after an hour."""
        past_hour = datetime.now(UTC) - timedelta(hours=2)
        budget = RequesterBudget(
            requester_id="test",
            queries_this_hour=MAX_QUERIES_PER_REQUESTER_PER_HOUR,
            hour_start=past_hour,
        )
        
        # Should reset on can_query check
        assert budget.can_query() is True
        assert budget.queries_this_hour == 0
    
    def test_round_trip_serialization(self) -> None:
        """Test full serialization round trip."""
        original = RequesterBudget(
            requester_id="round_trip_requester",
            queries_this_hour=12,
        )
        
        data = original.to_dict()
        restored = RequesterBudget.from_dict(data)
        
        assert restored.requester_id == original.requester_id
        assert restored.queries_this_hour == original.queries_this_hour


# =============================================================================
# IN-MEMORY BUDGET STORE TESTS
# =============================================================================


class TestInMemoryBudgetStore:
    """Tests for InMemoryBudgetStore."""
    
    def test_save_and_load(self) -> None:
        """Test saving and loading budget data."""
        store = InMemoryBudgetStore()
        data = {"key": "value", "number": 42}
        
        store.save("fed-1", data)
        result = store.load("fed-1")
        
        assert result == data
    
    def test_load_nonexistent(self) -> None:
        """Test loading non-existent federation returns None."""
        store = InMemoryBudgetStore()
        
        result = store.load("nonexistent")
        
        assert result is None
    
    def test_delete(self) -> None:
        """Test deleting budget data."""
        store = InMemoryBudgetStore()
        store.save("fed-1", {"data": "test"})
        
        deleted = store.delete("fed-1")
        
        assert deleted is True
        assert store.load("fed-1") is None
    
    def test_delete_nonexistent(self) -> None:
        """Test deleting non-existent federation returns False."""
        store = InMemoryBudgetStore()
        
        deleted = store.delete("nonexistent")
        
        assert deleted is False
    
    def test_list_federations(self) -> None:
        """Test listing all federation IDs."""
        store = InMemoryBudgetStore()
        store.save("fed-1", {})
        store.save("fed-2", {})
        store.save("fed-3", {})
        
        federations = store.list_federations()
        
        assert set(federations) == {"fed-1", "fed-2", "fed-3"}
    
    def test_clear(self) -> None:
        """Test clearing all data."""
        store = InMemoryBudgetStore()
        store.save("fed-1", {})
        store.save("fed-2", {})
        
        store.clear()
        
        assert store.list_federations() == []
    
    def test_overwrite_existing(self) -> None:
        """Test that save overwrites existing data."""
        store = InMemoryBudgetStore()
        store.save("fed-1", {"version": 1})
        store.save("fed-1", {"version": 2})
        
        result = store.load("fed-1")
        
        assert result == {"version": 2}


# =============================================================================
# FILE BUDGET STORE TESTS
# =============================================================================


class TestFileBudgetStore:
    """Tests for FileBudgetStore."""
    
    def test_save_and_load(self, tmp_path: Path) -> None:
        """Test saving and loading budget data."""
        store = FileBudgetStore(tmp_path)
        data = {"key": "value", "number": 42}
        
        store.save("fed-1", data)
        result = store.load("fed-1")
        
        assert result == data
    
    def test_load_nonexistent(self, tmp_path: Path) -> None:
        """Test loading non-existent federation returns None."""
        store = FileBudgetStore(tmp_path)
        
        result = store.load("nonexistent")
        
        assert result is None
    
    def test_delete(self, tmp_path: Path) -> None:
        """Test deleting budget data."""
        store = FileBudgetStore(tmp_path)
        store.save("fed-1", {"data": "test"})
        
        deleted = store.delete("fed-1")
        
        assert deleted is True
        assert store.load("fed-1") is None
    
    def test_delete_nonexistent(self, tmp_path: Path) -> None:
        """Test deleting non-existent federation returns False."""
        store = FileBudgetStore(tmp_path)
        
        deleted = store.delete("nonexistent")
        
        assert deleted is False
    
    def test_list_federations(self, tmp_path: Path) -> None:
        """Test listing all federation IDs."""
        store = FileBudgetStore(tmp_path)
        store.save("fed-1", {})
        store.save("fed-2", {})
        store.save("fed-3", {})
        
        federations = store.list_federations()
        
        assert set(federations) == {"fed-1", "fed-2", "fed-3"}
    
    def test_list_federations_ignores_temp_files(self, tmp_path: Path) -> None:
        """Test that .tmp files are not listed."""
        store = FileBudgetStore(tmp_path)
        store.save("fed-1", {})
        
        # Manually create a temp file
        (tmp_path / "fed-2.json.tmp").write_text("{}")
        
        federations = store.list_federations()
        
        assert federations == ["fed-1"]
    
    def test_sanitizes_federation_id(self, tmp_path: Path) -> None:
        """Test that federation IDs are sanitized for filesystem."""
        store = FileBudgetStore(tmp_path)
        
        # IDs with path separators should be sanitized
        store.save("fed/with/slashes", {"test": True})
        result = store.load("fed/with/slashes")
        
        assert result == {"test": True}
    
    def test_creates_directory(self, tmp_path: Path) -> None:
        """Test that store creates base directory if needed."""
        subdir = tmp_path / "nested" / "budgets"
        store = FileBudgetStore(subdir)
        
        store.save("fed-1", {})
        
        assert subdir.exists()
        assert store.load("fed-1") == {}
    
    def test_handles_invalid_json_gracefully(self, tmp_path: Path) -> None:
        """Test that invalid JSON returns None."""
        store = FileBudgetStore(tmp_path)
        
        # Write invalid JSON
        (tmp_path / "broken.json").write_text("not valid json {{{")
        
        result = store.load("broken")
        
        assert result is None


# =============================================================================
# DATABASE BUDGET STORE TESTS - ASYNC MODE
# =============================================================================


class TestDatabaseBudgetStoreAsync:
    """Tests for DatabaseBudgetStore in async mode."""
    
    @pytest.fixture
    def store(self, mock_async_pool: MagicMock) -> DatabaseBudgetStore:
        """Create a DatabaseBudgetStore with mocked pool."""
        return DatabaseBudgetStore(mock_async_pool)
    
    @pytest.mark.asyncio
    async def test_save_async_new_record(
        self,
        store: DatabaseBudgetStore,
        mock_async_pool: MagicMock,
        sample_budget_data: dict[str, Any],
    ) -> None:
        """Test saving a new budget record."""
        conn = await mock_async_pool.acquire().__aenter__()
        conn.execute = AsyncMock()
        
        await store.save_async("fed-123", sample_budget_data)
        
        conn.execute.assert_called_once()
        call_args = conn.execute.call_args
        
        # Check the SQL contains INSERT with ON CONFLICT
        sql = call_args[0][0]
        assert "INSERT INTO privacy_budgets" in sql
        assert "ON CONFLICT (federation_id) DO UPDATE" in sql
    
    @pytest.mark.asyncio
    async def test_save_async_includes_all_fields(
        self,
        store: DatabaseBudgetStore,
        mock_async_pool: MagicMock,
        sample_budget_data: dict[str, Any],
    ) -> None:
        """Test that save includes all required fields."""
        conn = await mock_async_pool.acquire().__aenter__()
        conn.execute = AsyncMock()
        
        await store.save_async("fed-123", sample_budget_data)
        
        call_args = conn.execute.call_args[0]
        
        # Federation ID should be first param
        assert call_args[1] == "fed-123"
        # Daily epsilon budget
        assert call_args[2] == 10.0
        # Daily delta budget
        assert call_args[3] == 1e-4
    
    @pytest.mark.asyncio
    async def test_load_async_existing_record(
        self,
        store: DatabaseBudgetStore,
        mock_async_pool: MagicMock,
    ) -> None:
        """Test loading an existing budget record."""
        conn = await mock_async_pool.acquire().__aenter__()
        
        # Mock the fetchrow result
        mock_row = {
            "federation_id": "fed-123",
            "daily_epsilon_budget": 10.0,
            "daily_delta_budget": 1e-4,
            "budget_period_hours": 24,
            "spent_epsilon": 2.5,
            "spent_delta": 1e-5,
            "queries_today": 15,
            "period_start": datetime.now(UTC),
            "topic_budgets": "{}",
            "requester_budgets": "{}",
            "schema_version": 1,
        }
        conn.fetchrow = AsyncMock(return_value=mock_row)
        
        result = await store.load_async("fed-123")
        
        assert result is not None
        assert result["federation_id"] == "fed-123"
        assert result["daily_epsilon_budget"] == 10.0
        assert result["spent_epsilon"] == 2.5
    
    @pytest.mark.asyncio
    async def test_load_async_nonexistent(
        self,
        store: DatabaseBudgetStore,
        mock_async_pool: MagicMock,
    ) -> None:
        """Test loading a non-existent record returns None."""
        conn = await mock_async_pool.acquire().__aenter__()
        conn.fetchrow = AsyncMock(return_value=None)
        
        result = await store.load_async("nonexistent")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_load_async_parses_jsonb_fields(
        self,
        store: DatabaseBudgetStore,
        mock_async_pool: MagicMock,
    ) -> None:
        """Test that JSONB fields are properly parsed."""
        conn = await mock_async_pool.acquire().__aenter__()
        
        topic_budgets = {
            "hash1": {"topic_hash": "hash1", "query_count": 2}
        }
        requester_budgets = {
            "req1": {"requester_id": "req1", "queries_this_hour": 5}
        }
        
        mock_row = {
            "federation_id": "fed-123",
            "daily_epsilon_budget": 10.0,
            "daily_delta_budget": 1e-4,
            "budget_period_hours": 24,
            "spent_epsilon": 0.0,
            "spent_delta": 0.0,
            "queries_today": 0,
            "period_start": datetime.now(UTC),
            "topic_budgets": json.dumps(topic_budgets),
            "requester_budgets": json.dumps(requester_budgets),
            "schema_version": 1,
        }
        conn.fetchrow = AsyncMock(return_value=mock_row)
        
        result = await store.load_async("fed-123")
        
        assert result["topic_budgets"] == topic_budgets
        assert result["requester_budgets"] == requester_budgets
    
    @pytest.mark.asyncio
    async def test_delete_async_existing(
        self,
        store: DatabaseBudgetStore,
        mock_async_pool: MagicMock,
    ) -> None:
        """Test deleting an existing record."""
        conn = await mock_async_pool.acquire().__aenter__()
        conn.execute = AsyncMock(return_value="DELETE 1")
        
        result = await store.delete_async("fed-123")
        
        assert result is True
        conn.execute.assert_called_once()
        assert "DELETE FROM privacy_budgets" in conn.execute.call_args[0][0]
    
    @pytest.mark.asyncio
    async def test_delete_async_nonexistent(
        self,
        store: DatabaseBudgetStore,
        mock_async_pool: MagicMock,
    ) -> None:
        """Test deleting a non-existent record."""
        conn = await mock_async_pool.acquire().__aenter__()
        conn.execute = AsyncMock(return_value="DELETE 0")
        
        result = await store.delete_async("nonexistent")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_list_federations_async(
        self,
        store: DatabaseBudgetStore,
        mock_async_pool: MagicMock,
    ) -> None:
        """Test listing all federation IDs."""
        conn = await mock_async_pool.acquire().__aenter__()
        conn.fetch = AsyncMock(return_value=[
            {"federation_id": "fed-1"},
            {"federation_id": "fed-2"},
            {"federation_id": "fed-3"},
        ])
        
        result = await store.list_federations_async()
        
        assert result == ["fed-1", "fed-2", "fed-3"]


# =============================================================================
# DATABASE BUDGET STORE TESTS - SYNC MODE
# =============================================================================


class TestDatabaseBudgetStoreSync:
    """Tests for DatabaseBudgetStore in sync mode (psycopg2)."""
    
    @pytest.fixture
    def store(self, mock_sync_connection: MagicMock) -> DatabaseBudgetStore:
        """Create a DatabaseBudgetStore in sync mode."""
        return DatabaseBudgetStore.from_sync_connection(mock_sync_connection)
    
    def test_sync_mode_flag(self, store: DatabaseBudgetStore) -> None:
        """Test that sync mode is correctly set."""
        assert store._sync_mode is True
    
    def test_save_sync(
        self,
        store: DatabaseBudgetStore,
        mock_sync_connection: MagicMock,
        sample_budget_data: dict[str, Any],
    ) -> None:
        """Test saving in sync mode."""
        cursor = mock_sync_connection.cursor.return_value
        
        store.save("fed-123", sample_budget_data)
        
        cursor.execute.assert_called_once()
        mock_sync_connection.commit.assert_called_once()
        cursor.close.assert_called_once()
    
    def test_save_sync_sql_params(
        self,
        store: DatabaseBudgetStore,
        mock_sync_connection: MagicMock,
        sample_budget_data: dict[str, Any],
    ) -> None:
        """Test that sync save uses correct SQL placeholder style."""
        cursor = mock_sync_connection.cursor.return_value
        
        store.save("fed-123", sample_budget_data)
        
        sql = cursor.execute.call_args[0][0]
        # psycopg2 uses %s placeholders
        assert "%s" in sql
        # asyncpg uses $1 placeholders - should not be present
        assert "$1" not in sql
    
    def test_load_sync_existing(
        self,
        store: DatabaseBudgetStore,
        mock_sync_connection: MagicMock,
    ) -> None:
        """Test loading in sync mode."""
        cursor = mock_sync_connection.cursor.return_value
        cursor.fetchone.return_value = (
            "fed-123",  # federation_id
            10.0,       # daily_epsilon_budget
            1e-4,       # daily_delta_budget
            24,         # budget_period_hours
            2.5,        # spent_epsilon
            1e-5,       # spent_delta
            15,         # queries_today
            datetime.now(UTC),  # period_start
            "{}",       # topic_budgets
            "{}",       # requester_budgets
            1,          # schema_version
        )
        
        result = store.load("fed-123")
        
        assert result is not None
        assert result["federation_id"] == "fed-123"
        assert result["daily_epsilon_budget"] == 10.0
    
    def test_load_sync_nonexistent(
        self,
        store: DatabaseBudgetStore,
        mock_sync_connection: MagicMock,
    ) -> None:
        """Test loading non-existent record in sync mode."""
        cursor = mock_sync_connection.cursor.return_value
        cursor.fetchone.return_value = None
        
        result = store.load("nonexistent")
        
        assert result is None
    
    def test_delete_sync_existing(
        self,
        store: DatabaseBudgetStore,
        mock_sync_connection: MagicMock,
    ) -> None:
        """Test deleting in sync mode."""
        cursor = mock_sync_connection.cursor.return_value
        cursor.rowcount = 1
        
        result = store.delete("fed-123")
        
        assert result is True
        mock_sync_connection.commit.assert_called_once()
    
    def test_delete_sync_nonexistent(
        self,
        store: DatabaseBudgetStore,
        mock_sync_connection: MagicMock,
    ) -> None:
        """Test deleting non-existent record in sync mode."""
        cursor = mock_sync_connection.cursor.return_value
        cursor.rowcount = 0
        
        result = store.delete("nonexistent")
        
        assert result is False
    
    def test_list_federations_sync(
        self,
        store: DatabaseBudgetStore,
        mock_sync_connection: MagicMock,
    ) -> None:
        """Test listing federation IDs in sync mode."""
        cursor = mock_sync_connection.cursor.return_value
        cursor.fetchall.return_value = [
            ("fed-1",),
            ("fed-2",),
            ("fed-3",),
        ]
        
        result = store.list_federations()
        
        assert result == ["fed-1", "fed-2", "fed-3"]


# =============================================================================
# DATABASE BUDGET STORE - ROW CONVERSION TESTS
# =============================================================================


class TestDatabaseBudgetStoreRowConversion:
    """Tests for _row_to_dict conversion."""
    
    @pytest.fixture
    def store(self, mock_async_pool: MagicMock) -> DatabaseBudgetStore:
        """Create a store for testing."""
        return DatabaseBudgetStore(mock_async_pool)
    
    def test_row_to_dict_with_dict_like_row(
        self,
        store: DatabaseBudgetStore,
    ) -> None:
        """Test conversion with dict-like row (asyncpg Record)."""
        now = datetime.now(UTC)
        row = {
            "federation_id": "fed-123",
            "daily_epsilon_budget": 10.0,
            "daily_delta_budget": 1e-4,
            "budget_period_hours": 24,
            "spent_epsilon": 2.5,
            "spent_delta": 1e-5,
            "queries_today": 15,
            "period_start": now,
            "topic_budgets": {"hash1": {"topic_hash": "hash1"}},
            "requester_budgets": {},
            "schema_version": 1,
        }
        
        result = store._row_to_dict(row)
        
        assert result["federation_id"] == "fed-123"
        assert result["daily_epsilon_budget"] == 10.0
        assert result["spent_epsilon"] == 2.5
        assert result["_version"] == 1
    
    def test_row_to_dict_parses_json_string(
        self,
        store: DatabaseBudgetStore,
    ) -> None:
        """Test that JSON strings are parsed."""
        row = {
            "federation_id": "fed-123",
            "daily_epsilon_budget": 10.0,
            "daily_delta_budget": 1e-4,
            "budget_period_hours": 24,
            "spent_epsilon": 0.0,
            "spent_delta": 0.0,
            "queries_today": 0,
            "period_start": datetime.now(UTC),
            "topic_budgets": '{"hash1": {"topic_hash": "hash1"}}',
            "requester_budgets": '{"req1": {"requester_id": "req1"}}',
            "schema_version": 1,
        }
        
        result = store._row_to_dict(row)
        
        assert isinstance(result["topic_budgets"], dict)
        assert result["topic_budgets"]["hash1"]["topic_hash"] == "hash1"
        assert isinstance(result["requester_budgets"], dict)
        assert result["requester_budgets"]["req1"]["requester_id"] == "req1"
    
    def test_row_to_dict_handles_datetime(
        self,
        store: DatabaseBudgetStore,
    ) -> None:
        """Test that datetime is converted to ISO string."""
        now = datetime.now(UTC)
        row = {
            "federation_id": "fed-123",
            "daily_epsilon_budget": 10.0,
            "daily_delta_budget": 1e-4,
            "budget_period_hours": 24,
            "spent_epsilon": 0.0,
            "spent_delta": 0.0,
            "queries_today": 0,
            "period_start": now,
            "topic_budgets": {},
            "requester_budgets": {},
            "schema_version": 1,
        }
        
        result = store._row_to_dict(row)
        
        assert result["period_start"] == now.isoformat()
    
    def test_row_to_dict_uses_defaults_for_missing(
        self,
        store: DatabaseBudgetStore,
    ) -> None:
        """Test that defaults are used for missing values."""
        row = {
            "federation_id": "fed-123",
            "period_start": datetime.now(UTC),
            "topic_budgets": {},
            "requester_budgets": {},
        }
        
        result = store._row_to_dict(row)
        
        assert result["daily_epsilon_budget"] == DEFAULT_DAILY_EPSILON_BUDGET
        assert result["daily_delta_budget"] == DEFAULT_DAILY_DELTA_BUDGET
        assert result["budget_period_hours"] == 24
        assert result["spent_epsilon"] == 0.0
        assert result["queries_today"] == 0


# =============================================================================
# PRIVACY BUDGET PERSISTENCE INTEGRATION TESTS
# =============================================================================


class TestPrivacyBudgetPersistence:
    """Integration tests for PrivacyBudget with storage backends."""
    
    def test_load_or_create_new(self, federation_id: UUID) -> None:
        """Test creating a new budget with load_or_create."""
        store = InMemoryBudgetStore()
        
        budget = PrivacyBudget.load_or_create(
            federation_id,
            store,
            daily_epsilon_budget=15.0,
        )
        
        assert budget.federation_id == federation_id
        assert budget.daily_epsilon_budget == 15.0
        assert budget.has_store
        # Should be persisted
        assert store.load(str(federation_id)) is not None
    
    def test_load_or_create_existing(self, federation_id: UUID) -> None:
        """Test loading an existing budget with load_or_create."""
        store = InMemoryBudgetStore()
        
        # Create initial budget
        budget1 = PrivacyBudget.load_or_create(federation_id, store)
        budget1.consume(1.0, 1e-6, "topic_hash", "requester_1")
        
        # Load again
        budget2 = PrivacyBudget.load_or_create(federation_id, store)
        
        assert budget2.spent_epsilon == budget1.spent_epsilon
        assert budget2.queries_today == 1
    
    def test_auto_save_on_consume(self, federation_id: UUID) -> None:
        """Test that consume() auto-saves when store is attached."""
        store = InMemoryBudgetStore()
        budget = PrivacyBudget.load_or_create(federation_id, store)
        
        budget.consume(2.0, 1e-6, "topic_hash")
        
        # Reload and verify state was persisted
        reloaded = PrivacyBudget.load_or_create(federation_id, store)
        assert reloaded.spent_epsilon == 2.0
    
    def test_save_without_store_raises(self, federation_id: UUID) -> None:
        """Test that save() raises if no store is attached."""
        budget = PrivacyBudget(federation_id=federation_id)
        
        with pytest.raises(RuntimeError, match="No BudgetStore attached"):
            budget.save()
    
    def test_attach_store(self, federation_id: UUID) -> None:
        """Test attaching a store after creation."""
        budget = PrivacyBudget(federation_id=federation_id)
        store = InMemoryBudgetStore()
        
        assert not budget.has_store
        
        budget.attach_store(store)
        
        assert budget.has_store
        budget.save()  # Should not raise
    
    def test_serialize_full_state(self, federation_id: UUID) -> None:
        """Test that serialize() captures full state."""
        budget = PrivacyBudget(federation_id=federation_id)
        
        # Add some state
        budget.consume(1.0, 1e-6, "topic_1", "requester_1")
        budget.consume(0.5, 1e-7, "topic_2", "requester_2")
        
        serialized = budget.serialize()
        
        assert serialized["federation_id"] == str(federation_id)
        assert serialized["spent_epsilon"] == 1.5
        assert serialized["queries_today"] == 2
        assert "topic_1" in serialized["topic_budgets"]
        assert "topic_2" in serialized["topic_budgets"]
        assert "requester_1" in serialized["requester_budgets"]
        assert "_version" in serialized
    
    def test_from_dict_restores_full_state(self, federation_id: UUID) -> None:
        """Test that from_dict() restores full state."""
        # Create budget with state
        original = PrivacyBudget(federation_id=federation_id)
        original.consume(2.5, 1e-5, "topic_hash", "requester_1")
        
        # Serialize and restore
        serialized = original.serialize()
        restored = PrivacyBudget.from_dict(serialized)
        
        assert restored.federation_id == original.federation_id
        assert restored.spent_epsilon == original.spent_epsilon
        assert restored.queries_today == original.queries_today
        assert "topic_hash" in restored.topic_budgets
        assert "requester_1" in restored.requester_budgets
    
    def test_period_reset_clears_topic_budgets(self, federation_id: UUID) -> None:
        """Test that period reset clears topic budgets."""
        store = InMemoryBudgetStore()
        budget = PrivacyBudget.load_or_create(federation_id, store)
        
        # Add state
        budget.consume(1.0, 1e-6, "topic_hash")
        assert len(budget.topic_budgets) == 1
        
        # Force period reset by setting old period_start
        budget.period_start = datetime.now(UTC) - timedelta(hours=25)
        budget._maybe_reset_period()
        
        assert budget.spent_epsilon == 0.0
        assert budget.queries_today == 0
        assert len(budget.topic_budgets) == 0
    
    def test_period_reset_preserves_requester_budgets(
        self,
        federation_id: UUID,
    ) -> None:
        """Test that period reset doesn't clear requester budgets."""
        budget = PrivacyBudget(federation_id=federation_id)
        budget.consume(1.0, 1e-6, "topic", "requester_1")
        
        # Force period reset
        budget.period_start = datetime.now(UTC) - timedelta(hours=25)
        budget._maybe_reset_period()
        
        # Requester budgets are preserved (they reset hourly, not daily)
        assert "requester_1" in budget.requester_budgets
    
    def test_file_store_round_trip(
        self,
        tmp_path: Path,
        federation_id: UUID,
    ) -> None:
        """Test full round-trip with FileBudgetStore."""
        store = FileBudgetStore(tmp_path)
        
        # Create budget with state
        budget = PrivacyBudget.load_or_create(federation_id, store)
        budget.consume(3.0, 1e-5, "topic_1", "requester_1")
        budget.consume(2.0, 1e-6, "topic_2")
        
        # Create new store instance (simulating restart)
        store2 = FileBudgetStore(tmp_path)
        restored = PrivacyBudget.load_or_create(federation_id, store2)
        
        assert restored.spent_epsilon == 5.0
        assert restored.queries_today == 2
        assert len(restored.topic_budgets) == 2
        assert len(restored.requester_budgets) == 1


# =============================================================================
# BUDGET STORE PROTOCOL COMPLIANCE
# =============================================================================


class TestBudgetStoreProtocol:
    """Test that all stores implement BudgetStore protocol."""
    
    def test_inmemory_is_budget_store(self) -> None:
        """Test InMemoryBudgetStore implements BudgetStore."""
        store = InMemoryBudgetStore()
        assert isinstance(store, BudgetStore)
    
    def test_file_is_budget_store(self, tmp_path: Path) -> None:
        """Test FileBudgetStore implements BudgetStore."""
        store = FileBudgetStore(tmp_path)
        assert isinstance(store, BudgetStore)
    
    def test_database_is_budget_store(self, mock_async_pool: MagicMock) -> None:
        """Test DatabaseBudgetStore implements BudgetStore."""
        store = DatabaseBudgetStore(mock_async_pool)
        assert isinstance(store, BudgetStore)
