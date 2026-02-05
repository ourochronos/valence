"""Differential privacy implementation for Valence federation.

Provides privacy-preserving aggregation with configurable epsilon,
privacy budget tracking, temporal smoothing, and rate limiting.

Addresses THREAT-MODEL.md §1.3.3 - k-Anonymity Threshold Attack.

Reference: spec/components/federation-layer/DIFFERENTIAL-PRIVACY.md
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

import numpy as np

# =============================================================================
# CONSTANTS
# =============================================================================

# Epsilon bounds
MIN_EPSILON = 0.01  # Below this provides negligible utility
MAX_EPSILON = 3.0  # Above this provides negligible privacy

# Default values
DEFAULT_EPSILON = 1.0
DEFAULT_DELTA = 1e-6
DEFAULT_MIN_CONTRIBUTORS = 5
SENSITIVE_MIN_CONTRIBUTORS = 10

# Histogram suppression
HISTOGRAM_SUPPRESSION_THRESHOLD = 20

# Temporal smoothing
DEFAULT_MEMBERSHIP_SMOOTHING_HOURS = 24

# Rate limits
MAX_QUERIES_PER_TOPIC_PER_DAY = 3
MAX_QUERIES_PER_FEDERATION_PER_DAY = 100
MAX_QUERIES_PER_REQUESTER_PER_HOUR = 20

# Daily budget
DEFAULT_DAILY_EPSILON_BUDGET = 10.0
DEFAULT_DAILY_DELTA_BUDGET = 1e-4

# Budget consumption for failed k-anonymity queries (Issue #177 - prevent probing)
# Even failed queries consume budget to prevent adversaries from learning
# about population size through repeated queries
FAILED_QUERY_EPSILON_COST = 0.1  # Small but non-zero cost

# =============================================================================
# SENSITIVE DOMAIN CLASSIFICATION (Issue #177 - Structured classification)
# =============================================================================

# Sensitive domain categories with their exact matches and normalized forms
# Using structured classification instead of substring matching for precision
SENSITIVE_DOMAIN_CATEGORIES: dict[str, frozenset[str]] = {
    "health": frozenset([
        "health", "medical", "mental_health", "diagnosis", "treatment",
        "healthcare", "clinical", "therapy", "counseling", "psychiatric",
    ]),
    "finance": frozenset([
        "finance", "banking", "investments", "salary", "debt",
        "financial", "credit", "taxes", "income", "wealth",
    ]),
    "legal": frozenset([
        "legal", "law", "criminal", "lawsuit", "arrest",
        "court", "litigation", "prosecution", "conviction",
    ]),
    "political": frozenset([
        "politics", "political", "voting", "election",
        "government", "partisan", "campaign", "ballot",
    ]),
    "religious": frozenset([
        "religion", "religious", "faith", "spiritual",
        "worship", "church", "mosque", "temple", "synagogue",
    ]),
    "identity": frozenset([
        "sexuality", "sexual", "gender", "lgbtq",
        "orientation", "identity", "transgender", "nonbinary",
    ]),
    "employment": frozenset([
        "employment", "hr", "hiring", "termination",
        "workplace", "employee", "employer", "human_resources",
    ]),
    "substance": frozenset([
        "addiction", "substance", "abuse",
        "recovery", "rehab", "dependency", "sobriety",
    ]),
    "immigration": frozenset([
        "immigration", "visa", "asylum",
        "refugee", "citizenship", "deportation", "naturalization",
    ]),
}

# Flattened set for quick membership testing (exact match only)
SENSITIVE_DOMAINS: frozenset[str] = frozenset(
    domain
    for category_domains in SENSITIVE_DOMAIN_CATEGORIES.values()
    for domain in category_domains
)


# =============================================================================
# ENUMS
# =============================================================================


class PrivacyLevel(str, Enum):
    """Pre-defined privacy levels with recommended parameters."""

    MAXIMUM = "maximum"  # ε=0.1, δ=10⁻⁸ - Medical, financial, legal
    HIGH = "high"  # ε=0.5, δ=10⁻⁷ - Personal opinions, sensitive
    STANDARD = "standard"  # ε=1.0, δ=10⁻⁶ - General knowledge
    RELAXED = "relaxed"  # ε=2.0, δ=10⁻⁵ - Low-sensitivity


class NoiseMechanism(str, Enum):
    """Noise mechanism for differential privacy."""

    LAPLACE = "laplace"  # Pure DP (δ=0)
    GAUSSIAN = "gaussian"  # Approximate DP (δ>0), better for composition


class BudgetCheckResult(str, Enum):
    """Result of privacy budget check."""

    OK = "ok"
    DAILY_EPSILON_EXHAUSTED = "daily_epsilon_exhausted"
    DAILY_DELTA_EXHAUSTED = "daily_delta_exhausted"
    TOPIC_RATE_LIMITED = "topic_rate_limited"
    FEDERATION_RATE_LIMITED = "federation_rate_limited"
    REQUESTER_RATE_LIMITED = "requester_rate_limited"


# =============================================================================
# PRIVACY PARAMETERS
# =============================================================================


@dataclass
class PrivacyConfig:
    """Privacy configuration for a federation.

    Specifies differential privacy parameters and budget limits.
    """

    # Core DP parameters
    epsilon: float = DEFAULT_EPSILON
    delta: float = DEFAULT_DELTA

    # k-anonymity
    min_contributors: int = DEFAULT_MIN_CONTRIBUTORS
    sensitive_domain: bool = False

    # Budget
    daily_epsilon_budget: float = DEFAULT_DAILY_EPSILON_BUDGET
    daily_delta_budget: float = DEFAULT_DAILY_DELTA_BUDGET

    # Temporal smoothing
    membership_smoothing_hours: int = DEFAULT_MEMBERSHIP_SMOOTHING_HOURS

    # Histogram
    histogram_suppression_threshold: int = HISTOGRAM_SUPPRESSION_THRESHOLD

    # Noise mechanism
    noise_mechanism: NoiseMechanism = NoiseMechanism.LAPLACE

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not MIN_EPSILON <= self.epsilon <= MAX_EPSILON:
            raise ValueError(
                f"epsilon must be in [{MIN_EPSILON}, {MAX_EPSILON}], got {self.epsilon}"
            )
        if self.delta >= 1e-4:
            raise ValueError(f"delta must be < 10⁻⁴, got {self.delta}")
        if self.min_contributors < 5:
            raise ValueError(f"min_contributors must be >= 5, got {self.min_contributors}")

    @classmethod
    def from_level(cls, level: PrivacyLevel) -> PrivacyConfig:
        """Create config from a predefined privacy level."""
        configs = {
            PrivacyLevel.MAXIMUM: cls(epsilon=0.1, delta=1e-8, min_contributors=10),
            PrivacyLevel.HIGH: cls(epsilon=0.5, delta=1e-7, min_contributors=8),
            PrivacyLevel.STANDARD: cls(epsilon=1.0, delta=1e-6, min_contributors=5),
            PrivacyLevel.RELAXED: cls(epsilon=2.0, delta=1e-5, min_contributors=5),
        }
        return configs[level]

    @property
    def effective_min_contributors(self) -> int:
        """Get effective minimum contributors, considering sensitivity."""
        if self.sensitive_domain:
            return max(self.min_contributors, SENSITIVE_MIN_CONTRIBUTORS)
        return self.min_contributors

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "min_contributors": self.min_contributors,
            "effective_min_contributors": self.effective_min_contributors,
            "sensitive_domain": self.sensitive_domain,
            "daily_epsilon_budget": self.daily_epsilon_budget,
            "membership_smoothing_hours": self.membership_smoothing_hours,
            "histogram_suppression_threshold": self.histogram_suppression_threshold,
            "noise_mechanism": self.noise_mechanism.value,
        }


# =============================================================================
# TOPIC BUDGET TRACKING
# =============================================================================


@dataclass
class TopicBudget:
    """Budget tracking for a specific topic."""

    topic_hash: str
    query_count: int = 0
    epsilon_spent: float = 0.0
    last_query: datetime = field(default_factory=datetime.utcnow)

    def can_query(self, max_queries: int = MAX_QUERIES_PER_TOPIC_PER_DAY) -> bool:
        """Check if topic can be queried."""
        return self.query_count < max_queries

    def record_query(self, epsilon: float) -> None:
        """Record a query against this topic."""
        self.query_count += 1
        self.epsilon_spent += epsilon
        self.last_query = datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for persistence."""
        return {
            "topic_hash": self.topic_hash,
            "query_count": self.query_count,
            "epsilon_spent": self.epsilon_spent,
            "last_query": self.last_query.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TopicBudget:
        """Deserialize from dictionary."""
        return cls(
            topic_hash=data["topic_hash"],
            query_count=data.get("query_count", 0),
            epsilon_spent=data.get("epsilon_spent", 0.0),
            last_query=datetime.fromisoformat(data["last_query"])
            if "last_query" in data
            else datetime.utcnow(),
        )


@dataclass
class RequesterBudget:
    """Budget tracking for a specific requester."""

    requester_id: str
    queries_this_hour: int = 0
    hour_start: datetime = field(default_factory=datetime.utcnow)

    def can_query(self, max_per_hour: int = MAX_QUERIES_PER_REQUESTER_PER_HOUR) -> bool:
        """Check if requester can query."""
        self._maybe_reset_hour()
        return self.queries_this_hour < max_per_hour

    def record_query(self) -> None:
        """Record a query from this requester."""
        self._maybe_reset_hour()
        self.queries_this_hour += 1

    def _maybe_reset_hour(self) -> None:
        """Reset hourly counter if needed."""
        now = datetime.utcnow()
        if (now - self.hour_start).total_seconds() > 3600:
            self.queries_this_hour = 0
            self.hour_start = now

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for persistence."""
        return {
            "requester_id": self.requester_id,
            "queries_this_hour": self.queries_this_hour,
            "hour_start": self.hour_start.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RequesterBudget:
        """Deserialize from dictionary."""
        return cls(
            requester_id=data["requester_id"],
            queries_this_hour=data.get("queries_this_hour", 0),
            hour_start=datetime.fromisoformat(data["hour_start"])
            if "hour_start" in data
            else datetime.utcnow(),
        )


# =============================================================================
# BUDGET STORAGE (Issue #144 - Persist privacy budget across restarts)
# =============================================================================


@runtime_checkable
class BudgetStore(Protocol):
    """Protocol for privacy budget persistence.

    Implementations must provide save/load/delete/list operations
    for privacy budgets keyed by federation_id.
    """

    def save(self, federation_id: str, data: dict[str, Any]) -> None:
        """Save budget data for a federation.

        Args:
            federation_id: Federation identifier (UUID as string)
            data: Serialized budget data
        """
        ...

    def load(self, federation_id: str) -> dict[str, Any] | None:
        """Load budget data for a federation.

        Args:
            federation_id: Federation identifier

        Returns:
            Serialized budget data or None if not found
        """
        ...

    def delete(self, federation_id: str) -> bool:
        """Delete budget data for a federation.

        Args:
            federation_id: Federation identifier

        Returns:
            True if deleted, False if not found
        """
        ...

    def list_federations(self) -> list[str]:
        """List all federation IDs with stored budgets.

        Returns:
            List of federation IDs
        """
        ...


class InMemoryBudgetStore:
    """In-memory budget store for testing.

    Not persistent - data lost on process restart.
    """

    def __init__(self) -> None:
        self._storage: dict[str, dict[str, Any]] = {}

    def save(self, federation_id: str, data: dict[str, Any]) -> None:
        """Save budget data in memory."""
        self._storage[federation_id] = data

    def load(self, federation_id: str) -> dict[str, Any] | None:
        """Load budget data from memory."""
        return self._storage.get(federation_id)

    def delete(self, federation_id: str) -> bool:
        """Delete budget data from memory."""
        if federation_id in self._storage:
            del self._storage[federation_id]
            return True
        return False

    def list_federations(self) -> list[str]:
        """List all federation IDs."""
        return list(self._storage.keys())

    def clear(self) -> None:
        """Clear all stored data (testing helper)."""
        self._storage.clear()


class FileBudgetStore:
    """File-based budget store for persistent storage.

    Stores each federation's budget as a separate JSON file.
    Thread-safe through atomic file operations.
    """

    def __init__(self, base_path: str | Path) -> None:
        """Initialize file-based budget store.

        Args:
            base_path: Directory for budget files
        """
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)

    def _file_path(self, federation_id: str) -> Path:
        """Get file path for a federation's budget."""
        # Sanitize federation_id for filesystem safety
        safe_id = federation_id.replace("/", "_").replace("\\", "_")
        return self._base_path / f"{safe_id}.json"

    def save(self, federation_id: str, data: dict[str, Any]) -> None:
        """Save budget data to file atomically."""
        file_path = self._file_path(federation_id)
        temp_path = file_path.with_suffix(".json.tmp")

        # Write to temp file first, then rename (atomic on POSIX)
        try:
            temp_path.write_text(json.dumps(data, indent=2, default=str))
            temp_path.rename(file_path)
        except Exception:
            # Clean up temp file on failure
            if temp_path.exists():
                temp_path.unlink()
            raise

    def load(self, federation_id: str) -> dict[str, Any] | None:
        """Load budget data from file."""
        file_path = self._file_path(federation_id)
        if not file_path.exists():
            return None

        try:
            return json.loads(file_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def delete(self, federation_id: str) -> bool:
        """Delete budget file."""
        file_path = self._file_path(federation_id)
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def list_federations(self) -> list[str]:
        """List all federation IDs with stored budgets."""
        return [f.stem for f in self._base_path.glob("*.json") if not f.name.endswith(".tmp")]


class DatabaseBudgetStore:
    """PostgreSQL-based budget store for persistent storage.

    Stores privacy budgets in the privacy_budgets table.
    Requires migration 014-privacy-budget.sql to be applied.

    This is the recommended store for production deployments as it:
    - Survives process restarts
    - Works in multi-process environments
    - Integrates with existing Valence database infrastructure

    Example:
        import asyncpg

        pool = await asyncpg.create_pool(database_url)
        store = DatabaseBudgetStore(pool)
        budget = PrivacyBudget.load_or_create(federation_id, store=store)
    """

    def __init__(self, pool: Any) -> None:
        """Initialize database budget store.

        Args:
            pool: asyncpg connection pool or compatible async pool
        """
        self._pool = pool
        self._sync_mode = False

    @classmethod
    def from_sync_connection(cls, conn: Any) -> "DatabaseBudgetStore":
        """Create store from a synchronous connection (psycopg2 or similar).

        Args:
            conn: Synchronous database connection

        Returns:
            DatabaseBudgetStore configured for sync operations
        """
        store = cls(conn)
        store._sync_mode = True
        return store

    def save(self, federation_id: str, data: dict[str, Any]) -> None:
        """Save budget data to database.

        Note: This method is sync for compatibility with BudgetStore protocol.
        For async contexts, use save_async().
        """
        import asyncio

        if self._sync_mode:
            self._save_sync(federation_id, data)
        else:
            # Run async save in event loop
            try:
                loop = asyncio.get_running_loop()
                # If in async context, schedule as task
                loop.create_task(self.save_async(federation_id, data))
            except RuntimeError:
                # No running loop, run synchronously
                asyncio.run(self.save_async(federation_id, data))

    async def save_async(self, federation_id: str, data: dict[str, Any]) -> None:
        """Save budget data to database (async version)."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO privacy_budgets (
                    federation_id,
                    daily_epsilon_budget,
                    daily_delta_budget,
                    budget_period_hours,
                    spent_epsilon,
                    spent_delta,
                    queries_today,
                    period_start,
                    topic_budgets,
                    requester_budgets,
                    schema_version
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (federation_id) DO UPDATE SET
                    daily_epsilon_budget = EXCLUDED.daily_epsilon_budget,
                    daily_delta_budget = EXCLUDED.daily_delta_budget,
                    budget_period_hours = EXCLUDED.budget_period_hours,
                    spent_epsilon = EXCLUDED.spent_epsilon,
                    spent_delta = EXCLUDED.spent_delta,
                    queries_today = EXCLUDED.queries_today,
                    period_start = EXCLUDED.period_start,
                    topic_budgets = EXCLUDED.topic_budgets,
                    requester_budgets = EXCLUDED.requester_budgets,
                    schema_version = EXCLUDED.schema_version
                """,
                federation_id,
                data.get("daily_epsilon_budget", DEFAULT_DAILY_EPSILON_BUDGET),
                data.get("daily_delta_budget", DEFAULT_DAILY_DELTA_BUDGET),
                data.get("budget_period_hours", 24),
                data.get("spent_epsilon", 0.0),
                data.get("spent_delta", 0.0),
                data.get("queries_today", 0),
                datetime.fromisoformat(data["period_start"])
                if "period_start" in data
                else datetime.utcnow(),
                json.dumps(data.get("topic_budgets", {})),
                json.dumps(data.get("requester_budgets", {})),
                data.get("_version", 1),
            )

    def _save_sync(self, federation_id: str, data: dict[str, Any]) -> None:
        """Save budget data synchronously (for psycopg2)."""
        cursor = self._pool.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO privacy_budgets (
                    federation_id,
                    daily_epsilon_budget,
                    daily_delta_budget,
                    budget_period_hours,
                    spent_epsilon,
                    spent_delta,
                    queries_today,
                    period_start,
                    topic_budgets,
                    requester_budgets,
                    schema_version
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (federation_id) DO UPDATE SET
                    daily_epsilon_budget = EXCLUDED.daily_epsilon_budget,
                    daily_delta_budget = EXCLUDED.daily_delta_budget,
                    budget_period_hours = EXCLUDED.budget_period_hours,
                    spent_epsilon = EXCLUDED.spent_epsilon,
                    spent_delta = EXCLUDED.spent_delta,
                    queries_today = EXCLUDED.queries_today,
                    period_start = EXCLUDED.period_start,
                    topic_budgets = EXCLUDED.topic_budgets,
                    requester_budgets = EXCLUDED.requester_budgets,
                    schema_version = EXCLUDED.schema_version
                """,
                (
                    federation_id,
                    data.get("daily_epsilon_budget", DEFAULT_DAILY_EPSILON_BUDGET),
                    data.get("daily_delta_budget", DEFAULT_DAILY_DELTA_BUDGET),
                    data.get("budget_period_hours", 24),
                    data.get("spent_epsilon", 0.0),
                    data.get("spent_delta", 0.0),
                    data.get("queries_today", 0),
                    datetime.fromisoformat(data["period_start"])
                    if "period_start" in data
                    else datetime.utcnow(),
                    json.dumps(data.get("topic_budgets", {})),
                    json.dumps(data.get("requester_budgets", {})),
                    data.get("_version", 1),
                ),
            )
            self._pool.commit()
        finally:
            cursor.close()

    def load(self, federation_id: str) -> dict[str, Any] | None:
        """Load budget data from database.

        Note: This method is sync for compatibility with BudgetStore protocol.
        For async contexts, use load_async().
        """
        import asyncio

        if self._sync_mode:
            return self._load_sync(federation_id)
        else:
            try:
                loop = asyncio.get_running_loop()
                # Can't await here, return None and rely on async usage
                return None
            except RuntimeError:
                return asyncio.run(self.load_async(federation_id))

    async def load_async(self, federation_id: str) -> dict[str, Any] | None:
        """Load budget data from database (async version)."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    federation_id,
                    daily_epsilon_budget,
                    daily_delta_budget,
                    budget_period_hours,
                    spent_epsilon,
                    spent_delta,
                    queries_today,
                    period_start,
                    topic_budgets,
                    requester_budgets,
                    schema_version
                FROM privacy_budgets
                WHERE federation_id = $1
                """,
                federation_id,
            )

            if row is None:
                return None

            return self._row_to_dict(row)

    def _load_sync(self, federation_id: str) -> dict[str, Any] | None:
        """Load budget data synchronously (for psycopg2)."""
        cursor = self._pool.cursor()
        try:
            cursor.execute(
                """
                SELECT
                    federation_id,
                    daily_epsilon_budget,
                    daily_delta_budget,
                    budget_period_hours,
                    spent_epsilon,
                    spent_delta,
                    queries_today,
                    period_start,
                    topic_budgets,
                    requester_budgets,
                    schema_version
                FROM privacy_budgets
                WHERE federation_id = %s
                """,
                (federation_id,),
            )
            row = cursor.fetchone()

            if row is None:
                return None

            # Convert tuple to dict-like for _row_to_dict
            columns = [
                "federation_id",
                "daily_epsilon_budget",
                "daily_delta_budget",
                "budget_period_hours",
                "spent_epsilon",
                "spent_delta",
                "queries_today",
                "period_start",
                "topic_budgets",
                "requester_budgets",
                "schema_version",
            ]
            row_dict = dict(zip(columns, row))
            return self._row_to_dict(row_dict)
        finally:
            cursor.close()

    def _row_to_dict(self, row: Any) -> dict[str, Any]:
        """Convert database row to serialized dict format."""
        # Handle both asyncpg Record and dict-like objects
        if hasattr(row, "get"):
            get = row.get
        else:
            get = lambda k, d=None: getattr(row, k, d)

        # Parse JSONB fields
        topic_budgets = get("topic_budgets", {})
        if isinstance(topic_budgets, str):
            topic_budgets = json.loads(topic_budgets)

        requester_budgets = get("requester_budgets", {})
        if isinstance(requester_budgets, str):
            requester_budgets = json.loads(requester_budgets)

        period_start = get("period_start")
        if isinstance(period_start, datetime):
            period_start_str = period_start.isoformat()
        else:
            period_start_str = str(period_start)

        return {
            "federation_id": get("federation_id"),
            "daily_epsilon_budget": float(get("daily_epsilon_budget", DEFAULT_DAILY_EPSILON_BUDGET)),
            "daily_delta_budget": float(get("daily_delta_budget", DEFAULT_DAILY_DELTA_BUDGET)),
            "budget_period_hours": int(get("budget_period_hours", 24)),
            "spent_epsilon": float(get("spent_epsilon", 0.0)),
            "spent_delta": float(get("spent_delta", 0.0)),
            "queries_today": int(get("queries_today", 0)),
            "period_start": period_start_str,
            "topic_budgets": topic_budgets,
            "requester_budgets": requester_budgets,
            "_version": int(get("schema_version", 1)),
        }

    def delete(self, federation_id: str) -> bool:
        """Delete budget data from database.

        Note: This method is sync for compatibility with BudgetStore protocol.
        For async contexts, use delete_async().
        """
        import asyncio

        if self._sync_mode:
            return self._delete_sync(federation_id)
        else:
            try:
                loop = asyncio.get_running_loop()
                return False  # Can't determine in non-async context
            except RuntimeError:
                return asyncio.run(self.delete_async(federation_id))

    async def delete_async(self, federation_id: str) -> bool:
        """Delete budget data from database (async version)."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM privacy_budgets WHERE federation_id = $1",
                federation_id,
            )
            return result == "DELETE 1"

    def _delete_sync(self, federation_id: str) -> bool:
        """Delete budget data synchronously (for psycopg2)."""
        cursor = self._pool.cursor()
        try:
            cursor.execute(
                "DELETE FROM privacy_budgets WHERE federation_id = %s",
                (federation_id,),
            )
            deleted = cursor.rowcount > 0
            self._pool.commit()
            return deleted
        finally:
            cursor.close()

    def list_federations(self) -> list[str]:
        """List all federation IDs with stored budgets.

        Note: This method is sync for compatibility with BudgetStore protocol.
        For async contexts, use list_federations_async().
        """
        import asyncio

        if self._sync_mode:
            return self._list_federations_sync()
        else:
            try:
                loop = asyncio.get_running_loop()
                return []  # Can't determine in non-async context
            except RuntimeError:
                return asyncio.run(self.list_federations_async())

    async def list_federations_async(self) -> list[str]:
        """List all federation IDs (async version)."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT federation_id FROM privacy_budgets ORDER BY federation_id"
            )
            return [row["federation_id"] for row in rows]

    def _list_federations_sync(self) -> list[str]:
        """List all federation IDs synchronously (for psycopg2)."""
        cursor = self._pool.cursor()
        try:
            cursor.execute(
                "SELECT federation_id FROM privacy_budgets ORDER BY federation_id"
            )
            return [row[0] for row in cursor.fetchall()]
        finally:
            cursor.close()


# =============================================================================
# PRIVACY BUDGET
# =============================================================================


@dataclass
class PrivacyBudget:
    """Track cumulative privacy loss for a federation.

    Implements budget tracking with:
    - Daily epsilon/delta limits (reset every 24h)
    - Per-topic query limits (prevent enumeration)
    - Per-requester rate limits (slow down adversaries)
    - Optional persistence via BudgetStore (Issue #144)

    Example with persistence:
        store = FileBudgetStore("/var/lib/valence/budgets")
        budget = PrivacyBudget.load_or_create(federation_id, store=store)
        # Budget now auto-saves on consume()
    """

    federation_id: UUID

    # Daily budget
    daily_epsilon_budget: float = DEFAULT_DAILY_EPSILON_BUDGET
    daily_delta_budget: float = DEFAULT_DAILY_DELTA_BUDGET

    # Current spend
    spent_epsilon: float = 0.0
    spent_delta: float = 0.0
    queries_today: int = 0

    # Per-topic tracking
    topic_budgets: dict[str, TopicBudget] = field(default_factory=dict)

    # Per-requester tracking
    requester_budgets: dict[str, RequesterBudget] = field(default_factory=dict)

    # Reset tracking
    period_start: datetime = field(default_factory=datetime.utcnow)
    budget_period_hours: int = 24

    # Storage (not serialized, set after creation)
    _store: BudgetStore | None = field(default=None, repr=False, compare=False)

    def check_budget(
        self,
        epsilon: float,
        delta: float,
        topic_hash: str,
        requester_id: str | None = None,
    ) -> tuple[bool, BudgetCheckResult]:
        """Check if budget allows this query.

        Args:
            epsilon: Privacy budget to consume
            delta: Delta to consume
            topic_hash: Hash of the query topic
            requester_id: ID of the requester (for rate limiting)

        Returns:
            Tuple of (can_query, reason)
        """
        self._maybe_reset_period()

        # Global epsilon budget
        if self.spent_epsilon + epsilon > self.daily_epsilon_budget:
            return False, BudgetCheckResult.DAILY_EPSILON_EXHAUSTED

        # Global delta budget
        if self.spent_delta + delta > self.daily_delta_budget:
            return False, BudgetCheckResult.DAILY_DELTA_EXHAUSTED

        # Federation-wide query limit
        if self.queries_today >= MAX_QUERIES_PER_FEDERATION_PER_DAY:
            return False, BudgetCheckResult.FEDERATION_RATE_LIMITED

        # Per-topic rate limit
        topic = self.topic_budgets.get(topic_hash)
        if topic and not topic.can_query():
            return False, BudgetCheckResult.TOPIC_RATE_LIMITED

        # Per-requester rate limit
        if requester_id:
            requester = self.requester_budgets.get(requester_id)
            if requester and not requester.can_query():
                return False, BudgetCheckResult.REQUESTER_RATE_LIMITED

        return True, BudgetCheckResult.OK

    def consume(
        self,
        epsilon: float,
        delta: float,
        topic_hash: str,
        requester_id: str | None = None,
    ) -> None:
        """Record budget consumption.

        Args:
            epsilon: Epsilon consumed
            delta: Delta consumed
            topic_hash: Hash of the query topic
            requester_id: ID of the requester

        Note:
            If a BudgetStore is attached, automatically persists after consumption.
        """
        self._maybe_reset_period()

        # Global spend
        self.spent_epsilon += epsilon
        self.spent_delta += delta
        self.queries_today += 1

        # Topic spend
        if topic_hash not in self.topic_budgets:
            self.topic_budgets[topic_hash] = TopicBudget(topic_hash=topic_hash)
        self.topic_budgets[topic_hash].record_query(epsilon)

        # Requester spend
        if requester_id:
            if requester_id not in self.requester_budgets:
                self.requester_budgets[requester_id] = RequesterBudget(requester_id=requester_id)
            self.requester_budgets[requester_id].record_query()

        # Auto-persist if store is attached (Issue #144)
        if self._store is not None:
            self.save()

    def remaining_epsilon(self) -> float:
        """Get remaining epsilon budget."""
        self._maybe_reset_period()
        return max(0.0, self.daily_epsilon_budget - self.spent_epsilon)

    def remaining_queries(self) -> int:
        """Get remaining query count."""
        self._maybe_reset_period()
        return max(0, MAX_QUERIES_PER_FEDERATION_PER_DAY - self.queries_today)

    def topic_queries_remaining(self, topic_hash: str) -> int:
        """Get remaining queries for a specific topic."""
        topic = self.topic_budgets.get(topic_hash)
        if not topic:
            return MAX_QUERIES_PER_TOPIC_PER_DAY
        return max(0, MAX_QUERIES_PER_TOPIC_PER_DAY - topic.query_count)

    def _maybe_reset_period(self) -> None:
        """Reset budget if period has elapsed."""
        now = datetime.utcnow()
        if (now - self.period_start).total_seconds() > self.budget_period_hours * 3600:
            self.spent_epsilon = 0.0
            self.spent_delta = 0.0
            self.queries_today = 0
            self.topic_budgets.clear()
            # Don't clear requester budgets (hourly reset handled separately)
            self.period_start = now

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (summary, for display/API)."""
        return {
            "federation_id": str(self.federation_id),
            "daily_epsilon_budget": self.daily_epsilon_budget,
            "daily_delta_budget": self.daily_delta_budget,
            "spent_epsilon": self.spent_epsilon,
            "spent_delta": self.spent_delta,
            "queries_today": self.queries_today,
            "remaining_epsilon": self.remaining_epsilon(),
            "remaining_queries": self.remaining_queries(),
            "period_start": self.period_start.isoformat(),
            "topics_queried": len(self.topic_budgets),
        }

    def serialize(self) -> dict[str, Any]:
        """Serialize full state for persistence.

        Unlike to_dict(), this includes all data needed to reconstruct
        the budget state (topic_budgets, requester_budgets).

        Returns:
            Complete serialized state
        """
        return {
            "federation_id": str(self.federation_id),
            "daily_epsilon_budget": self.daily_epsilon_budget,
            "daily_delta_budget": self.daily_delta_budget,
            "spent_epsilon": self.spent_epsilon,
            "spent_delta": self.spent_delta,
            "queries_today": self.queries_today,
            "period_start": self.period_start.isoformat(),
            "budget_period_hours": self.budget_period_hours,
            "topic_budgets": {k: v.to_dict() for k, v in self.topic_budgets.items()},
            "requester_budgets": {k: v.to_dict() for k, v in self.requester_budgets.items()},
            "_version": 1,  # Schema version for future migrations
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], store: BudgetStore | None = None) -> PrivacyBudget:
        """Deserialize from dictionary.

        Args:
            data: Serialized budget data
            store: Optional storage backend to attach

        Returns:
            Reconstructed PrivacyBudget
        """
        # Parse topic budgets
        topic_budgets = {}
        for k, v in data.get("topic_budgets", {}).items():
            topic_budgets[k] = TopicBudget.from_dict(v)

        # Parse requester budgets
        requester_budgets = {}
        for k, v in data.get("requester_budgets", {}).items():
            requester_budgets[k] = RequesterBudget.from_dict(v)

        # Parse period_start
        period_start_str = data.get("period_start")
        if period_start_str:
            period_start = datetime.fromisoformat(period_start_str)
        else:
            period_start = datetime.utcnow()

        budget = cls(
            federation_id=UUID(data["federation_id"]),
            daily_epsilon_budget=data.get("daily_epsilon_budget", DEFAULT_DAILY_EPSILON_BUDGET),
            daily_delta_budget=data.get("daily_delta_budget", DEFAULT_DAILY_DELTA_BUDGET),
            spent_epsilon=data.get("spent_epsilon", 0.0),
            spent_delta=data.get("spent_delta", 0.0),
            queries_today=data.get("queries_today", 0),
            topic_budgets=topic_budgets,
            requester_budgets=requester_budgets,
            period_start=period_start,
            budget_period_hours=data.get("budget_period_hours", 24),
        )

        # Attach store if provided
        if store is not None:
            budget._store = store

        return budget

    @classmethod
    def load_or_create(
        cls,
        federation_id: UUID,
        store: BudgetStore,
        daily_epsilon_budget: float = DEFAULT_DAILY_EPSILON_BUDGET,
        daily_delta_budget: float = DEFAULT_DAILY_DELTA_BUDGET,
    ) -> PrivacyBudget:
        """Load existing budget from storage or create new one.

        This is the recommended way to create budgets with persistence.
        The returned budget will auto-save on consume().

        Args:
            federation_id: Federation identifier
            store: Storage backend for persistence
            daily_epsilon_budget: Default epsilon budget for new budgets
            daily_delta_budget: Default delta budget for new budgets

        Returns:
            Loaded or newly created PrivacyBudget with store attached

        Example:
            store = FileBudgetStore("/var/lib/valence/budgets")
            budget = PrivacyBudget.load_or_create(federation_id, store)
            # Budget persists automatically on each consume()
        """
        fed_id_str = str(federation_id)
        data = store.load(fed_id_str)

        if data is not None:
            budget = cls.from_dict(data, store=store)
            # Ensure federation_id matches (in case of file rename/corruption)
            if budget.federation_id != federation_id:
                budget.federation_id = federation_id
            return budget

        # Create new budget with store attached
        budget = cls(
            federation_id=federation_id,
            daily_epsilon_budget=daily_epsilon_budget,
            daily_delta_budget=daily_delta_budget,
        )
        budget._store = store
        budget.save()  # Persist initial state
        return budget

    def attach_store(self, store: BudgetStore) -> None:
        """Attach a storage backend for persistence.

        After attaching, consume() will auto-save budget state.

        Args:
            store: Storage backend to attach
        """
        self._store = store

    def save(self) -> None:
        """Manually persist budget state to attached store.

        Raises:
            RuntimeError: If no store is attached
        """
        if self._store is None:
            raise RuntimeError("No BudgetStore attached. Use attach_store() first.")
        self._store.save(str(self.federation_id), self.serialize())

    @property
    def has_store(self) -> bool:
        """Check if a storage backend is attached."""
        return self._store is not None


# =============================================================================
# NOISE MECHANISMS
# =============================================================================


def add_laplace_noise(
    true_value: float,
    sensitivity: float,
    epsilon: float,
) -> float:
    """Add Laplace noise for (ε, 0)-differential privacy.

    Args:
        true_value: The true value to protect
        sensitivity: Maximum change from adding/removing one record
        epsilon: Privacy parameter

    Returns:
        Noisy value
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    return true_value + noise


def add_gaussian_noise(
    true_value: float,
    sensitivity: float,
    epsilon: float,
    delta: float,
) -> float:
    """Add Gaussian noise for (ε, δ)-differential privacy.

    Args:
        true_value: The true value to protect
        sensitivity: Maximum change from adding/removing one record
        epsilon: Privacy parameter
        delta: Failure probability

    Returns:
        Noisy value
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if delta <= 0:
        raise ValueError(f"delta must be positive, got {delta}")

    sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    noise = np.random.normal(0, sigma)
    return true_value + noise


def add_noise(
    true_value: float,
    sensitivity: float,
    config: PrivacyConfig,
) -> float:
    """Add noise according to configured mechanism.

    Args:
        true_value: The true value to protect
        sensitivity: Maximum change from adding/removing one record
        config: Privacy configuration

    Returns:
        Noisy value
    """
    if config.noise_mechanism == NoiseMechanism.GAUSSIAN:
        return add_gaussian_noise(true_value, sensitivity, config.epsilon, config.delta)
    return add_laplace_noise(true_value, sensitivity, config.epsilon)


# =============================================================================
# TEMPORAL SMOOTHING
# =============================================================================


@dataclass
class MembershipEvent:
    """Record of a membership change."""

    member_id: str
    event_type: str  # "joined" or "departed"
    timestamp: datetime


class TemporalSmoother:
    """Smooth membership changes over time to prevent inference.

    When a member joins or departs, their contribution is phased
    in/out over the smoothing period to prevent timing attacks.
    """

    def __init__(self, smoothing_hours: int = DEFAULT_MEMBERSHIP_SMOOTHING_HOURS):
        self.smoothing_hours = smoothing_hours
        self.events: list[MembershipEvent] = []

    def record_join(self, member_id: str, timestamp: datetime | None = None) -> None:
        """Record a member joining."""
        self.events.append(
            MembershipEvent(
                member_id=member_id,
                event_type="joined",
                timestamp=timestamp or datetime.utcnow(),
            )
        )

    def record_departure(self, member_id: str, timestamp: datetime | None = None) -> None:
        """Record a member departing."""
        self.events.append(
            MembershipEvent(
                member_id=member_id,
                event_type="departed",
                timestamp=timestamp or datetime.utcnow(),
            )
        )

    def get_contribution_weight(
        self,
        member_id: str,
        joined_at: datetime | None,
        departed_at: datetime | None,
        query_time: datetime | None = None,
    ) -> float:
        """Get contribution weight for a member.

        Args:
            member_id: Member identifier
            joined_at: When member joined (None for founding members)
            departed_at: When member departed (None if still active)
            query_time: Time of query (default: now)

        Returns:
            Weight in [0, 1] for member's contribution
        """
        query_time = query_time or datetime.utcnow()

        # Active member: check if recently joined (ramp up)
        if departed_at is None:
            if joined_at is None:
                return 1.0  # Founding member

            hours_since_join = (query_time - joined_at).total_seconds() / 3600

            if hours_since_join < self.smoothing_hours:
                # Linearly ramp up over smoothing period
                return hours_since_join / self.smoothing_hours

            return 1.0

        # Departed member: check if recently departed (ramp down)
        hours_since_departure = (query_time - departed_at).total_seconds() / 3600

        if hours_since_departure >= self.smoothing_hours:
            return 0.0  # Fully phased out

        # Linearly ramp down, with probabilistic inclusion
        weight = 1.0 - (hours_since_departure / self.smoothing_hours)

        # Probabilistic inclusion for additional privacy
        if random.random() > weight:
            return 0.0

        return weight

    def should_include_member(
        self,
        member_id: str,
        joined_at: datetime | None,
        departed_at: datetime | None,
        query_time: datetime | None = None,
    ) -> bool:
        """Determine if member should be included in aggregate.

        Args:
            member_id: Member identifier
            joined_at: When member joined
            departed_at: When member departed (None if active)
            query_time: Time of query

        Returns:
            True if member should be included
        """
        weight = self.get_contribution_weight(member_id, joined_at, departed_at, query_time)
        return weight > 0.0


# =============================================================================
# HISTOGRAM HANDLING
# =============================================================================


def should_include_histogram(
    contributor_count: int,
    threshold: int = HISTOGRAM_SUPPRESSION_THRESHOLD,
) -> bool:
    """Check if histogram should be included in response.

    Args:
        contributor_count: Number of contributors
        threshold: Minimum contributors for histogram

    Returns:
        True if histogram can be safely included
    """
    return contributor_count >= threshold


def build_noisy_histogram(
    values: list[float],
    epsilon: float,
    bins: int = 5,
) -> dict[str, int]:
    """Build histogram with differential privacy noise.

    Args:
        values: Values to histogram (assumed in [0, 1])
        epsilon: Privacy budget for histogram (split across bins)
        bins: Number of histogram bins

    Returns:
        Noisy histogram as {bin_label: count}
    """
    # Create bin edges
    bin_edges = [i / bins for i in range(bins + 1)]
    counts = [0] * bins

    # Count values in each bin
    for v in values:
        v = max(0.0, min(1.0, v))  # Clamp to [0, 1]
        bin_idx = min(int(v * bins), bins - 1)
        counts[bin_idx] += 1

    # Add Laplace noise to each bin
    # Split epsilon budget across bins
    per_bin_epsilon = epsilon / bins
    noisy_counts = [max(0, round(c + np.random.laplace(0, 1.0 / per_bin_epsilon))) for c in counts]

    # Build result
    return {f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}": noisy_counts[i] for i in range(bins)}


# =============================================================================
# SENSITIVITY DETECTION
# =============================================================================


def is_sensitive_domain(domains: list[str]) -> bool:
    """Check if any domain is considered sensitive using structured classification.

    Uses exact token matching against a categorized sensitive domain registry
    rather than substring matching, which prevents false positives like
    "therapist" matching "the" or "healer" matching "heal".

    Args:
        domains: List of domain names (can be paths like "health/mental_health")

    Returns:
        True if any domain token matches a sensitive category
    """
    for domain in domains:
        # Normalize: lowercase and split on common separators
        domain_lower = domain.lower()
        # Split on path separators, underscores, hyphens to get tokens
        tokens = set()
        for sep in ["/", "_", "-", ".", ":"]:
            domain_lower = domain_lower.replace(sep, " ")
        tokens.update(domain_lower.split())
        # Also check the full domain as-is (normalized)
        tokens.add(domain.lower().replace("/", "_").replace("-", "_"))

        # Check for exact token match against sensitive domains
        if tokens & SENSITIVE_DOMAINS:
            return True
    return False


def get_sensitive_category(domain: str) -> str | None:
    """Get the sensitive category for a domain if it matches.

    Useful for determining what type of sensitivity applies.

    Args:
        domain: Domain name to check

    Returns:
        Category name if sensitive, None otherwise
    """
    domain_lower = domain.lower()
    tokens = set()
    for sep in ["/", "_", "-", ".", ":"]:
        domain_lower = domain_lower.replace(sep, " ")
    tokens.update(domain_lower.split())

    for category, category_domains in SENSITIVE_DOMAIN_CATEGORIES.items():
        if tokens & category_domains:
            return category
    return None


def compute_topic_hash(
    domain_filter: list[str],
    semantic_query: str | None = None,
) -> str:
    """Compute a stable hash for a query topic.

    Args:
        domain_filter: Domain path
        semantic_query: Optional semantic query

    Returns:
        SHA-256 hash of the topic
    """
    content = "|".join(sorted(domain_filter))
    if semantic_query:
        content += f"||{semantic_query}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# =============================================================================
# AGGREGATE PRIVACY
# =============================================================================


@dataclass
class PrivateAggregateResult:
    """Result of a privacy-preserving aggregation.

    Contains noisy statistics and privacy guarantees.
    """

    # Noisy results
    collective_confidence: float
    contributor_count: int  # Noisy count
    agreement_score: float | None = None

    # Optional histogram (only if above threshold)
    confidence_distribution: dict[str, int] | None = None

    # Privacy guarantees
    epsilon_used: float = 0.0
    delta: float = 0.0
    noise_mechanism: str = "laplace"
    k_anonymity_satisfied: bool = True
    histogram_suppressed: bool = False

    # Temporal smoothing applied
    temporal_smoothing_applied: bool = False
    smoothing_window_hours: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {
            "collective_confidence": self.collective_confidence,
            "contributor_count": self.contributor_count,
            "privacy_guarantees": {
                "epsilon": self.epsilon_used,
                "delta": self.delta,
                "mechanism": self.noise_mechanism,
                "k_anonymity_satisfied": self.k_anonymity_satisfied,
            },
        }

        if self.agreement_score is not None:
            result["agreement_score"] = self.agreement_score

        if self.confidence_distribution is not None:
            result["confidence_distribution"] = self.confidence_distribution
        else:
            result["histogram_suppressed"] = True
            result["histogram_suppression_reason"] = "insufficient_contributors"

        if self.temporal_smoothing_applied:
            result["temporal_smoothing"] = {
                "applied": True,
                "window_hours": self.smoothing_window_hours,
            }

        return result


def compute_private_aggregate(
    confidences: list[float],
    config: PrivacyConfig,
    include_histogram: bool = True,
) -> PrivateAggregateResult | None:
    """Compute privacy-preserving aggregate statistics.

    Args:
        confidences: List of confidence values from contributors
        config: Privacy configuration
        include_histogram: Whether to include histogram if possible

    Returns:
        PrivateAggregateResult or None if k-anonymity not satisfied
    """
    true_count = len(confidences)

    # Check k-anonymity
    if true_count < config.effective_min_contributors:
        return None

    # Compute true statistics
    true_mean = sum(confidences) / true_count if confidences else 0.0

    # Compute agreement (stddev-based)
    if len(confidences) > 1:
        variance = sum((c - true_mean) ** 2 for c in confidences) / len(confidences)
        true_agreement = 1.0 - min(1.0, math.sqrt(variance) * 2)  # Scale to [0, 1]
    else:
        true_agreement = 1.0

    # Add noise
    noisy_mean = add_noise(true_mean, 1.0 / true_count, config)
    noisy_count = max(0, round(add_noise(true_count, 1.0, config)))
    noisy_agreement = add_noise(true_agreement, 1.0 / true_count, config)

    # Clamp to valid ranges
    noisy_mean = max(0.0, min(1.0, noisy_mean))
    noisy_agreement = max(0.0, min(1.0, noisy_agreement))

    # Histogram
    histogram = None
    histogram_suppressed = True
    if include_histogram and should_include_histogram(
        true_count, config.histogram_suppression_threshold
    ):
        histogram = build_noisy_histogram(confidences, config.epsilon / 5)
        histogram_suppressed = False

    return PrivateAggregateResult(
        collective_confidence=noisy_mean,
        contributor_count=noisy_count,
        agreement_score=noisy_agreement,
        confidence_distribution=histogram,
        epsilon_used=config.epsilon,
        delta=config.delta,
        noise_mechanism=config.noise_mechanism.value,
        k_anonymity_satisfied=True,
        histogram_suppressed=histogram_suppressed,
    )


@dataclass
class QueryResult:
    """Result of a privacy-preserving query with budget tracking.

    Wraps PrivateAggregateResult with additional query metadata.
    """

    success: bool
    result: PrivateAggregateResult | None
    budget_consumed: bool
    epsilon_consumed: float
    failure_reason: str | None = None


def execute_private_query(
    confidences: list[float],
    config: PrivacyConfig,
    budget: PrivacyBudget,
    topic_hash: str,
    requester_id: str | None = None,
    include_histogram: bool = True,
) -> QueryResult:
    """Execute a privacy-preserving query with budget tracking.

    Unlike compute_private_aggregate, this function:
    1. Checks budget before execution
    2. Consumes budget even for failed k-anonymity queries (prevents probing)
    3. Returns detailed result with budget consumption info

    Issue #177: Consuming budget for failed queries prevents adversaries from
    learning about population size through repeated queries. Without this,
    an attacker could probe "does group X have at least k members?" for free.

    Args:
        confidences: List of confidence values from contributors
        config: Privacy configuration
        budget: Privacy budget to check and consume
        topic_hash: Hash of the query topic for rate limiting
        requester_id: ID of the requester for rate limiting
        include_histogram: Whether to include histogram if possible

    Returns:
        QueryResult with success status and optional aggregate result
    """
    # Check budget first
    can_query, check_result = budget.check_budget(
        config.epsilon, config.delta, topic_hash, requester_id
    )

    if not can_query:
        return QueryResult(
            success=False,
            result=None,
            budget_consumed=False,
            epsilon_consumed=0.0,
            failure_reason=f"Budget check failed: {check_result.value}",
        )

    # Check k-anonymity
    true_count = len(confidences)
    if true_count < config.effective_min_contributors:
        # IMPORTANT: Consume budget even for failed k-anonymity queries
        # This prevents probing attacks where adversaries learn population
        # size by observing which queries succeed vs fail
        budget.consume(
            FAILED_QUERY_EPSILON_COST,
            0.0,  # No delta for failed queries
            topic_hash,
            requester_id,
        )
        return QueryResult(
            success=False,
            result=None,
            budget_consumed=True,
            epsilon_consumed=FAILED_QUERY_EPSILON_COST,
            failure_reason=f"Insufficient contributors: {true_count} < {config.effective_min_contributors}",
        )

    # Execute the aggregate computation
    result = compute_private_aggregate(confidences, config, include_histogram)

    # Consume full budget for successful queries
    budget.consume(config.epsilon, config.delta, topic_hash, requester_id)

    return QueryResult(
        success=True,
        result=result,
        budget_consumed=True,
        epsilon_consumed=config.epsilon,
    )
