"""Migration framework for Valence.

Provides sequential, versioned database migrations with:
- Auto-discovery from a migrations directory
- State tracking in a `_migrations` table
- Up/down support with checksums for drift detection
- Dry-run mode
- Bootstrap for fresh installs

Each migration file must define:
    version: str      — e.g. "001"
    description: str  — human-readable name
    def up(conn) -> None:   — apply migration (receives psycopg2 connection)
    def down(conn) -> None: — rollback migration

Usage:
    runner = MigrationRunner(migrations_dir="/path/to/migrations")
    runner.up()          # apply all pending
    runner.down(to="001")  # rollback to version 001
    runner.status()      # list applied/pending
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

# The table used to track applied migrations
MIGRATIONS_TABLE = "_migrations"


class MigrationModule(Protocol):
    """Protocol defining what a migration module must expose."""

    version: str
    description: str

    def up(self, conn: Any) -> None: ...
    def down(self, conn: Any) -> None: ...


@dataclass
class MigrationInfo:
    """Metadata about a discovered migration."""

    version: str
    description: str
    checksum: str
    file_path: Path
    module: ModuleType

    def __lt__(self, other: MigrationInfo) -> bool:
        return self.version < other.version


@dataclass
class AppliedMigration:
    """Record of an applied migration from the DB."""

    version: str
    description: str
    checksum: str
    applied_at: datetime


@dataclass
class MigrationStatus:
    """Status of a single migration: applied, pending, or checksum mismatch."""

    version: str
    description: str
    state: str  # "applied", "pending", "checksum_mismatch"
    applied_at: datetime | None = None
    file_checksum: str | None = None
    db_checksum: str | None = None


class MigrationRunner:
    """Discovers, tracks, and applies database migrations.

    Args:
        migrations_dir: Path to directory containing NNN_description.py files.
        connection_factory: Callable that returns a psycopg2 connection.
            If None, uses valence.core.db.get_connection / put_connection.
    """

    def __init__(
        self,
        migrations_dir: str | Path | None = None,
        connection_factory: Any | None = None,
    ):
        if migrations_dir is None:
            # Default: <repo_root>/migrations
            migrations_dir = Path(__file__).resolve().parent.parent.parent.parent / "migrations"
        self.migrations_dir = Path(migrations_dir)
        self._connection_factory = connection_factory
        self._migrations: list[MigrationInfo] | None = None

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _get_connection(self):
        """Get a database connection."""
        if self._connection_factory:
            return self._connection_factory()
        from valence.lib.our_db import get_connection

        return get_connection()

    def _put_connection(self, conn):
        """Return a database connection."""
        if self._connection_factory:
            # For custom factories, just close
            try:
                conn.close()
            except Exception:
                pass
            return
        from valence.lib.our_db import put_connection

        put_connection(conn)

    # ------------------------------------------------------------------
    # Migration discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_checksum(file_path: Path) -> str:
        """Compute SHA-256 checksum of a migration file."""
        content = file_path.read_bytes()
        return hashlib.sha256(content).hexdigest()[:16]

    @staticmethod
    def _load_module(file_path: Path) -> ModuleType:
        """Dynamically load a Python migration module."""
        module_name = f"valence_migration_{file_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load migration: {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def discover(self) -> list[MigrationInfo]:
        """Discover all migration files in migrations_dir, sorted by version."""
        if self._migrations is not None:
            return self._migrations

        migrations: list[MigrationInfo] = []
        if not self.migrations_dir.is_dir():
            logger.warning(f"Migrations directory not found: {self.migrations_dir}")
            self._migrations = []
            return []

        for path in sorted(self.migrations_dir.glob("*.py")):
            if path.name.startswith("__"):
                continue
            # Expect NNN_description.py
            parts = path.stem.split("_", 1)
            if len(parts) < 2 or not parts[0].isdigit():
                logger.debug(f"Skipping non-migration file: {path.name}")
                continue

            try:
                module = self._load_module(path)
            except Exception as e:
                logger.error(f"Failed to load migration {path.name}: {e}")
                raise

            # Validate the module has required attributes
            for attr in ("version", "description", "up", "down"):
                if not hasattr(module, attr):
                    raise ValueError(f"Migration {path.name} missing required attribute: {attr}")

            checksum = self._compute_checksum(path)
            migrations.append(
                MigrationInfo(
                    version=module.version,
                    description=module.description,
                    checksum=checksum,
                    file_path=path,
                    module=module,
                )
            )

        migrations.sort()
        self._migrations = migrations
        return migrations

    def invalidate_cache(self) -> None:
        """Clear the cached migration list."""
        self._migrations = None

    # ------------------------------------------------------------------
    # State tracking
    # ------------------------------------------------------------------

    def _ensure_table(self, conn) -> None:
        """Create the _migrations tracking table if it doesn't exist."""
        cur = conn.cursor()
        try:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {MIGRATIONS_TABLE} (
                    version TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            conn.commit()
        finally:
            cur.close()

    def _get_applied(self, conn) -> list[AppliedMigration]:
        """Get list of applied migrations from the DB."""
        self._ensure_table(conn)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        try:
            cur.execute(f"SELECT version, description, checksum, applied_at FROM {MIGRATIONS_TABLE} ORDER BY version")
            rows = cur.fetchall()
            return [
                AppliedMigration(
                    version=row["version"],
                    description=row["description"],
                    checksum=row["checksum"],
                    applied_at=row["applied_at"],
                )
                for row in rows
            ]
        finally:
            cur.close()

    def _record_applied(self, conn, migration: MigrationInfo) -> None:
        """Record a migration as applied."""
        cur = conn.cursor()
        try:
            cur.execute(
                f"INSERT INTO {MIGRATIONS_TABLE} (version, description, checksum) VALUES (%s, %s, %s)",
                (migration.version, migration.description, migration.checksum),
            )
        finally:
            cur.close()

    def _remove_applied(self, conn, version: str) -> None:
        """Remove a migration record (for rollback)."""
        cur = conn.cursor()
        try:
            cur.execute(f"DELETE FROM {MIGRATIONS_TABLE} WHERE version = %s", (version,))
        finally:
            cur.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def status(self) -> list[MigrationStatus]:
        """Return status of all migrations (applied/pending/checksum_mismatch)."""
        migrations = self.discover()
        conn = self._get_connection()
        try:
            applied = {m.version: m for m in self._get_applied(conn)}
            result: list[MigrationStatus] = []

            for m in migrations:
                if m.version in applied:
                    db_record = applied[m.version]
                    if db_record.checksum != m.checksum:
                        state = "checksum_mismatch"
                    else:
                        state = "applied"
                    result.append(
                        MigrationStatus(
                            version=m.version,
                            description=m.description,
                            state=state,
                            applied_at=db_record.applied_at,
                            file_checksum=m.checksum,
                            db_checksum=db_record.checksum,
                        )
                    )
                else:
                    result.append(
                        MigrationStatus(
                            version=m.version,
                            description=m.description,
                            state="pending",
                            file_checksum=m.checksum,
                        )
                    )

            return result
        finally:
            self._put_connection(conn)

    def pending(self) -> list[MigrationInfo]:
        """Return list of pending (unapplied) migrations."""
        migrations = self.discover()
        conn = self._get_connection()
        try:
            applied_versions = {m.version for m in self._get_applied(conn)}
            return [m for m in migrations if m.version not in applied_versions]
        finally:
            self._put_connection(conn)

    def up(self, *, target: str | None = None, dry_run: bool = False) -> list[str]:
        """Apply pending migrations, optionally up to a target version.

        Args:
            target: Stop after applying this version (inclusive). None = apply all.
            dry_run: If True, only report what would be applied without executing.

        Returns:
            List of applied version strings.
        """
        migrations = self.discover()
        conn = self._get_connection()
        applied_versions: list[str] = []

        try:
            applied = {m.version for m in self._get_applied(conn)}
            to_apply = [m for m in migrations if m.version not in applied]

            if target:
                to_apply = [m for m in to_apply if m.version <= target]

            if not to_apply:
                logger.info("No pending migrations to apply.")
                return []

            for migration in to_apply:
                if dry_run:
                    logger.info(f"[DRY RUN] Would apply: {migration.version} — {migration.description}")
                    applied_versions.append(migration.version)
                    continue

                logger.info(f"Applying migration {migration.version}: {migration.description}")
                try:
                    migration.module.up(conn)
                    self._record_applied(conn, migration)
                    conn.commit()
                    applied_versions.append(migration.version)
                    logger.info(f"  ✓ Applied {migration.version}")
                except Exception as e:
                    conn.rollback()
                    logger.error(f"  ✗ Failed {migration.version}: {e}")
                    raise

            return applied_versions
        finally:
            self._put_connection(conn)

    def down(self, *, target: str | None = None, dry_run: bool = False) -> list[str]:
        """Rollback migrations, optionally down to a target version.

        Args:
            target: Stop after rolling back to this version (exclusive — this version
                    stays applied). None = rollback only the latest migration.
            dry_run: If True, only report what would be rolled back.

        Returns:
            List of rolled-back version strings.
        """
        migrations = self.discover()
        conn = self._get_connection()
        rolled_back: list[str] = []

        try:
            applied = {m.version for m in self._get_applied(conn)}
            migration_map = {m.version: m for m in migrations}

            # Sort applied in reverse order for rollback
            to_rollback = sorted(applied, reverse=True)

            if target:
                to_rollback = [v for v in to_rollback if v > target]
            else:
                # Default: rollback only the latest
                to_rollback = to_rollback[:1]

            if not to_rollback:
                logger.info("No migrations to rollback.")
                return []

            for version in to_rollback:
                migration = migration_map.get(version)
                if migration is None:
                    logger.warning(f"Migration file for version {version} not found, skipping rollback")
                    continue

                if dry_run:
                    logger.info(f"[DRY RUN] Would rollback: {version} — {migration.description}")
                    rolled_back.append(version)
                    continue

                logger.info(f"Rolling back migration {version}: {migration.description}")
                try:
                    migration.module.down(conn)
                    self._remove_applied(conn, version)
                    conn.commit()
                    rolled_back.append(version)
                    logger.info(f"  ✓ Rolled back {version}")
                except Exception as e:
                    conn.rollback()
                    logger.error(f"  ✗ Failed rollback {version}: {e}")
                    raise

            return rolled_back
        finally:
            self._put_connection(conn)

    def bootstrap(self, *, dry_run: bool = False) -> list[str]:
        """Bootstrap a fresh database by applying all migrations.

        This is semantically the same as `up()` but provides a clear intent
        for fresh installs. It also validates that no migrations are already applied.

        Returns:
            List of applied version strings.
        """
        conn = self._get_connection()
        try:
            applied = self._get_applied(conn)
            if applied:
                raise RuntimeError(f"Cannot bootstrap: {len(applied)} migration(s) already applied. Use 'migrate up' for incremental upgrades.")
        finally:
            self._put_connection(conn)

        return self.up(dry_run=dry_run)

    @staticmethod
    def create_migration(
        migrations_dir: str | Path,
        name: str,
    ) -> Path:
        """Scaffold a new migration file.

        Args:
            migrations_dir: Directory to create the migration in.
            name: Description for the migration (e.g. "add_users_table").

        Returns:
            Path to the created migration file.
        """
        migrations_dir = Path(migrations_dir)
        migrations_dir.mkdir(parents=True, exist_ok=True)

        # Determine next version number
        existing = sorted(migrations_dir.glob("*.py"))
        existing = [f for f in existing if not f.name.startswith("__")]
        next_num = 1
        for f in existing:
            parts = f.stem.split("_", 1)
            if parts[0].isdigit():
                next_num = max(next_num, int(parts[0]) + 1)

        # Sanitize name
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        version = f"{next_num:03d}"
        filename = f"{version}_{safe_name}.py"
        file_path = migrations_dir / filename

        template = f'''"""Migration {version}: {name}."""

version = "{version}"
description = "{name}"


def up(conn) -> None:
    """Apply migration."""
    cur = conn.cursor()
    try:
        # TODO: Add migration SQL here
        cur.execute("""
            -- Add your migration SQL here
            SELECT 1
        """)
    finally:
        cur.close()


def down(conn) -> None:
    """Rollback migration."""
    cur = conn.cursor()
    try:
        # TODO: Add rollback SQL here
        cur.execute("""
            -- Add your rollback SQL here
            SELECT 1
        """)
    finally:
        cur.close()
'''

        file_path.write_text(template)
        return file_path
