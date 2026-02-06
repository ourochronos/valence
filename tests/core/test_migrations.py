"""Tests for valence.core.migrations module.

Uses an in-memory SQLite-style approach with a temp directory of migration
files and a real (temporary) PostgreSQL-like test fixture — but since we
can't assume PostgreSQL in unit tests, we mock the DB layer.
"""

from __future__ import annotations

import textwrap
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from valence.core.migrations import (
    MIGRATIONS_TABLE,
    AppliedMigration,
    MigrationInfo,
    MigrationRunner,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def migrations_dir(tmp_path: Path) -> Path:
    """Create a temporary migrations directory with sample migrations."""
    mdir = tmp_path / "migrations"
    mdir.mkdir()
    return mdir


def _write_migration(mdir: Path, version: str, name: str, *, up_sql: str = "SELECT 1", down_sql: str = "SELECT 1") -> Path:
    """Helper to write a migration file."""
    filename = f"{version}_{name}.py"
    content = textwrap.dedent(f'''\
        """Migration {version}: {name}."""

        version = "{version}"
        description = "{name}"


        def up(conn) -> None:
            cur = conn.cursor()
            try:
                cur.execute("""{up_sql}""")
            finally:
                cur.close()


        def down(conn) -> None:
            cur = conn.cursor()
            try:
                cur.execute("""{down_sql}""")
            finally:
                cur.close()
    ''')
    path = mdir / filename
    path.write_text(content)
    return path


@pytest.fixture
def sample_migrations(migrations_dir: Path) -> Path:
    """Create three sample migrations."""
    _write_migration(migrations_dir, "001", "create_users")
    _write_migration(migrations_dir, "002", "add_email_column")
    _write_migration(migrations_dir, "003", "create_posts")
    return migrations_dir


def _mock_connection():
    """Create a mock connection with cursor support."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    # For RealDictCursor calls
    cursor.fetchall.return_value = []
    cursor.fetchone.return_value = None
    return conn, cursor


# ============================================================================
# Discovery Tests
# ============================================================================


class TestMigrationDiscovery:
    """Tests for migration file discovery."""

    def test_discover_empty_dir(self, migrations_dir: Path):
        runner = MigrationRunner(migrations_dir=migrations_dir)
        migrations = runner.discover()
        assert migrations == []

    def test_discover_finds_migrations(self, sample_migrations: Path):
        runner = MigrationRunner(migrations_dir=sample_migrations)
        migrations = runner.discover()
        assert len(migrations) == 3
        assert [m.version for m in migrations] == ["001", "002", "003"]

    def test_discover_sorted_by_version(self, migrations_dir: Path):
        # Write out of order
        _write_migration(migrations_dir, "003", "third")
        _write_migration(migrations_dir, "001", "first")
        _write_migration(migrations_dir, "002", "second")

        runner = MigrationRunner(migrations_dir=migrations_dir)
        migrations = runner.discover()
        assert [m.version for m in migrations] == ["001", "002", "003"]

    def test_discover_skips_dunder_files(self, migrations_dir: Path):
        (migrations_dir / "__init__.py").write_text("")
        _write_migration(migrations_dir, "001", "real")

        runner = MigrationRunner(migrations_dir=migrations_dir)
        migrations = runner.discover()
        assert len(migrations) == 1

    def test_discover_skips_non_numbered_files(self, migrations_dir: Path):
        (migrations_dir / "README.py").write_text("# not a migration")
        _write_migration(migrations_dir, "001", "real")

        runner = MigrationRunner(migrations_dir=migrations_dir)
        migrations = runner.discover()
        assert len(migrations) == 1

    def test_discover_caches_results(self, sample_migrations: Path):
        runner = MigrationRunner(migrations_dir=sample_migrations)
        m1 = runner.discover()
        m2 = runner.discover()
        assert m1 is m2

    def test_invalidate_cache(self, sample_migrations: Path):
        runner = MigrationRunner(migrations_dir=sample_migrations)
        m1 = runner.discover()
        runner.invalidate_cache()
        m2 = runner.discover()
        assert m1 is not m2

    def test_discover_nonexistent_dir(self, tmp_path: Path):
        runner = MigrationRunner(migrations_dir=tmp_path / "nonexistent")
        migrations = runner.discover()
        assert migrations == []

    def test_discover_validates_required_attrs(self, migrations_dir: Path):
        # Write a migration missing 'version'
        bad = migrations_dir / "001_bad.py"
        bad.write_text('description = "bad"\ndef up(conn): pass\ndef down(conn): pass\n')

        runner = MigrationRunner(migrations_dir=migrations_dir)
        with pytest.raises(ValueError, match="missing required attribute: version"):
            runner.discover()

    def test_discover_validates_up_method(self, migrations_dir: Path):
        bad = migrations_dir / "001_bad.py"
        bad.write_text('version = "001"\ndescription = "bad"\ndef down(conn): pass\n')

        runner = MigrationRunner(migrations_dir=migrations_dir)
        with pytest.raises(ValueError, match="missing required attribute: up"):
            runner.discover()

    def test_checksum_changes_with_content(self, migrations_dir: Path):
        path = _write_migration(migrations_dir, "001", "test")
        runner = MigrationRunner(migrations_dir=migrations_dir)
        checksum1 = runner.discover()[0].checksum

        # Modify the file
        runner.invalidate_cache()
        path.write_text(path.read_text() + "\n# modified\n")
        checksum2 = runner.discover()[0].checksum

        assert checksum1 != checksum2


# ============================================================================
# MigrationInfo Tests
# ============================================================================


class TestMigrationInfo:
    """Tests for MigrationInfo ordering."""

    def test_ordering(self):
        m1 = MigrationInfo(version="001", description="a", checksum="x", file_path=Path("a"), module=MagicMock())
        m2 = MigrationInfo(version="002", description="b", checksum="y", file_path=Path("b"), module=MagicMock())
        assert m1 < m2
        assert not m2 < m1

    def test_sort(self):
        m3 = MigrationInfo(version="003", description="c", checksum="z", file_path=Path("c"), module=MagicMock())
        m1 = MigrationInfo(version="001", description="a", checksum="x", file_path=Path("a"), module=MagicMock())
        m2 = MigrationInfo(version="002", description="b", checksum="y", file_path=Path("b"), module=MagicMock())
        assert sorted([m3, m1, m2]) == [m1, m2, m3]


# ============================================================================
# State Tracking Tests (mocked DB)
# ============================================================================


class TestStateTracking:
    """Tests for _ensure_table, _get_applied, _record_applied, _remove_applied."""

    def test_ensure_table_creates_table(self, sample_migrations: Path):
        conn, cursor = _mock_connection()
        runner = MigrationRunner(migrations_dir=sample_migrations)
        runner._ensure_table(conn)

        # Should have executed CREATE TABLE
        create_call = cursor.execute.call_args_list[0]
        assert MIGRATIONS_TABLE in create_call[0][0]
        conn.commit.assert_called_once()

    def test_record_applied(self, sample_migrations: Path):
        conn, cursor = _mock_connection()
        runner = MigrationRunner(migrations_dir=sample_migrations)
        migration = runner.discover()[0]

        runner._record_applied(conn, migration)

        insert_call = cursor.execute.call_args
        assert "INSERT INTO" in insert_call[0][0]
        assert insert_call[0][1][0] == "001"

    def test_remove_applied(self, sample_migrations: Path):
        conn, cursor = _mock_connection()
        runner = MigrationRunner(migrations_dir=sample_migrations)

        runner._remove_applied(conn, "001")

        delete_call = cursor.execute.call_args
        assert "DELETE FROM" in delete_call[0][0]
        assert delete_call[0][1] == ("001",)


# ============================================================================
# Status Tests
# ============================================================================


class TestStatus:
    """Tests for the status() method."""

    def test_all_pending(self, sample_migrations: Path):
        conn, cursor = _mock_connection()
        cursor.fetchall.return_value = []  # no applied

        runner = MigrationRunner(
            migrations_dir=sample_migrations,
            connection_factory=lambda: conn,
        )

        statuses = runner.status()
        assert len(statuses) == 3
        assert all(s.state == "pending" for s in statuses)

    def test_all_applied(self, sample_migrations: Path):
        conn, cursor = _mock_connection()
        runner = MigrationRunner(
            migrations_dir=sample_migrations,
            connection_factory=lambda: conn,
        )
        migrations = runner.discover()

        # Mock _get_applied to return matching records
        applied = [
            AppliedMigration(
                version=m.version,
                description=m.description,
                checksum=m.checksum,
                applied_at=datetime.now(UTC),
            )
            for m in migrations
        ]
        with patch.object(runner, "_get_applied", return_value=applied):
            statuses = runner.status()

        assert len(statuses) == 3
        assert all(s.state == "applied" for s in statuses)

    def test_checksum_mismatch_detected(self, sample_migrations: Path):
        conn, cursor = _mock_connection()
        runner = MigrationRunner(
            migrations_dir=sample_migrations,
            connection_factory=lambda: conn,
        )
        migrations = runner.discover()

        applied = [
            AppliedMigration(
                version=migrations[0].version,
                description=migrations[0].description,
                checksum="WRONG_CHECKSUM",
                applied_at=datetime.now(UTC),
            )
        ]
        with patch.object(runner, "_get_applied", return_value=applied):
            statuses = runner.status()

        assert statuses[0].state == "checksum_mismatch"
        assert statuses[1].state == "pending"
        assert statuses[2].state == "pending"

    def test_mixed_status(self, sample_migrations: Path):
        conn, cursor = _mock_connection()
        runner = MigrationRunner(
            migrations_dir=sample_migrations,
            connection_factory=lambda: conn,
        )
        migrations = runner.discover()

        applied = [
            AppliedMigration(
                version=migrations[0].version,
                description=migrations[0].description,
                checksum=migrations[0].checksum,
                applied_at=datetime.now(UTC),
            )
        ]
        with patch.object(runner, "_get_applied", return_value=applied):
            statuses = runner.status()

        assert statuses[0].state == "applied"
        assert statuses[1].state == "pending"
        assert statuses[2].state == "pending"


# ============================================================================
# Up (Apply) Tests
# ============================================================================


class TestMigrateUp:
    """Tests for the up() method."""

    def test_apply_all_pending(self, sample_migrations: Path):
        conn, cursor = _mock_connection()
        runner = MigrationRunner(
            migrations_dir=sample_migrations,
            connection_factory=lambda: conn,
        )

        with patch.object(runner, "_get_applied", return_value=[]):
            applied = runner.up()

        assert applied == ["001", "002", "003"]
        # Should commit for each migration
        assert conn.commit.call_count >= 3

    def test_apply_with_target(self, sample_migrations: Path):
        conn, cursor = _mock_connection()
        runner = MigrationRunner(
            migrations_dir=sample_migrations,
            connection_factory=lambda: conn,
        )

        with patch.object(runner, "_get_applied", return_value=[]):
            applied = runner.up(target="002")

        assert applied == ["001", "002"]

    def test_apply_skips_already_applied(self, sample_migrations: Path):
        conn, cursor = _mock_connection()
        runner = MigrationRunner(
            migrations_dir=sample_migrations,
            connection_factory=lambda: conn,
        )
        migrations = runner.discover()

        already = [
            AppliedMigration(
                version="001",
                description="create_users",
                checksum=migrations[0].checksum,
                applied_at=datetime.now(UTC),
            )
        ]
        with patch.object(runner, "_get_applied", return_value=already):
            applied = runner.up()

        assert applied == ["002", "003"]

    def test_apply_none_pending(self, sample_migrations: Path):
        conn, cursor = _mock_connection()
        runner = MigrationRunner(
            migrations_dir=sample_migrations,
            connection_factory=lambda: conn,
        )
        migrations = runner.discover()

        all_applied = [
            AppliedMigration(
                version=m.version,
                description=m.description,
                checksum=m.checksum,
                applied_at=datetime.now(UTC),
            )
            for m in migrations
        ]
        with patch.object(runner, "_get_applied", return_value=all_applied):
            applied = runner.up()

        assert applied == []

    def test_dry_run_does_not_execute(self, sample_migrations: Path):
        conn, cursor = _mock_connection()
        runner = MigrationRunner(
            migrations_dir=sample_migrations,
            connection_factory=lambda: conn,
        )

        with patch.object(runner, "_get_applied", return_value=[]):
            applied = runner.up(dry_run=True)

        assert applied == ["001", "002", "003"]
        # _record_applied should not have been called (no INSERT)
        # The migration up() should not have been called
        # In dry run we skip the module.up() call, so only _ensure_table + _get_applied happen
        # conn.commit should NOT be called for each migration (only for _ensure_table)

    def test_failure_rolls_back(self, sample_migrations: Path):
        conn, cursor = _mock_connection()
        runner = MigrationRunner(
            migrations_dir=sample_migrations,
            connection_factory=lambda: conn,
        )

        # Make second migration's up() raise
        migrations = runner.discover()
        migrations[1].module.up = MagicMock(side_effect=RuntimeError("boom"))

        with patch.object(runner, "_get_applied", return_value=[]):
            with pytest.raises(RuntimeError, match="boom"):
                runner.up()

        conn.rollback.assert_called()


# ============================================================================
# Down (Rollback) Tests
# ============================================================================


class TestMigrateDown:
    """Tests for the down() method."""

    def test_rollback_latest(self, sample_migrations: Path):
        conn, cursor = _mock_connection()
        runner = MigrationRunner(
            migrations_dir=sample_migrations,
            connection_factory=lambda: conn,
        )

        applied = [
            AppliedMigration(version="001", description="a", checksum="x", applied_at=datetime.now(UTC)),
            AppliedMigration(version="002", description="b", checksum="y", applied_at=datetime.now(UTC)),
            AppliedMigration(version="003", description="c", checksum="z", applied_at=datetime.now(UTC)),
        ]
        with patch.object(runner, "_get_applied", return_value=applied):
            rolled_back = runner.down()

        assert rolled_back == ["003"]

    def test_rollback_to_target(self, sample_migrations: Path):
        conn, cursor = _mock_connection()
        runner = MigrationRunner(
            migrations_dir=sample_migrations,
            connection_factory=lambda: conn,
        )

        applied = [
            AppliedMigration(version="001", description="a", checksum="x", applied_at=datetime.now(UTC)),
            AppliedMigration(version="002", description="b", checksum="y", applied_at=datetime.now(UTC)),
            AppliedMigration(version="003", description="c", checksum="z", applied_at=datetime.now(UTC)),
        ]
        with patch.object(runner, "_get_applied", return_value=applied):
            rolled_back = runner.down(target="001")

        assert rolled_back == ["003", "002"]

    def test_rollback_nothing_applied(self, sample_migrations: Path):
        conn, cursor = _mock_connection()
        runner = MigrationRunner(
            migrations_dir=sample_migrations,
            connection_factory=lambda: conn,
        )

        with patch.object(runner, "_get_applied", return_value=[]):
            rolled_back = runner.down()

        assert rolled_back == []

    def test_dry_run_rollback(self, sample_migrations: Path):
        conn, cursor = _mock_connection()
        runner = MigrationRunner(
            migrations_dir=sample_migrations,
            connection_factory=lambda: conn,
        )

        applied = [
            AppliedMigration(version="001", description="a", checksum="x", applied_at=datetime.now(UTC)),
            AppliedMigration(version="002", description="b", checksum="y", applied_at=datetime.now(UTC)),
        ]
        with patch.object(runner, "_get_applied", return_value=applied):
            rolled_back = runner.down(target="000", dry_run=True)

        assert rolled_back == ["002", "001"]

    def test_rollback_failure(self, sample_migrations: Path):
        conn, cursor = _mock_connection()
        runner = MigrationRunner(
            migrations_dir=sample_migrations,
            connection_factory=lambda: conn,
        )

        # Make down() raise on 003
        migrations = runner.discover()
        migrations[2].module.down = MagicMock(side_effect=RuntimeError("rollback boom"))

        applied = [
            AppliedMigration(version="001", description="a", checksum="x", applied_at=datetime.now(UTC)),
            AppliedMigration(version="002", description="b", checksum="y", applied_at=datetime.now(UTC)),
            AppliedMigration(version="003", description="c", checksum="z", applied_at=datetime.now(UTC)),
        ]
        with patch.object(runner, "_get_applied", return_value=applied):
            with pytest.raises(RuntimeError, match="rollback boom"):
                runner.down()

        conn.rollback.assert_called()


# ============================================================================
# Bootstrap Tests
# ============================================================================


class TestBootstrap:
    """Tests for the bootstrap() method."""

    def test_bootstrap_fresh_db(self, sample_migrations: Path):
        conn, cursor = _mock_connection()
        runner = MigrationRunner(
            migrations_dir=sample_migrations,
            connection_factory=lambda: conn,
        )

        with patch.object(runner, "_get_applied", return_value=[]):
            applied = runner.bootstrap()

        assert applied == ["001", "002", "003"]

    def test_bootstrap_rejects_existing_migrations(self, sample_migrations: Path):
        conn, cursor = _mock_connection()
        runner = MigrationRunner(
            migrations_dir=sample_migrations,
            connection_factory=lambda: conn,
        )

        already = [
            AppliedMigration(version="001", description="a", checksum="x", applied_at=datetime.now(UTC)),
        ]
        with patch.object(runner, "_get_applied", return_value=already):
            with pytest.raises(RuntimeError, match="Cannot bootstrap"):
                runner.bootstrap()

    def test_bootstrap_dry_run(self, sample_migrations: Path):
        conn, cursor = _mock_connection()
        runner = MigrationRunner(
            migrations_dir=sample_migrations,
            connection_factory=lambda: conn,
        )

        with patch.object(runner, "_get_applied", return_value=[]):
            applied = runner.bootstrap(dry_run=True)

        assert applied == ["001", "002", "003"]


# ============================================================================
# Create Migration Tests
# ============================================================================


class TestCreateMigration:
    """Tests for the create_migration() static method."""

    def test_create_first_migration(self, migrations_dir: Path):
        path = MigrationRunner.create_migration(migrations_dir, "add_users")
        assert path.exists()
        assert path.name == "001_add_users.py"

        content = path.read_text()
        assert 'version = "001"' in content
        assert 'description = "add_users"' in content
        assert "def up(conn)" in content
        assert "def down(conn)" in content

    def test_create_increments_version(self, sample_migrations: Path):
        path = MigrationRunner.create_migration(sample_migrations, "add_comments")
        assert path.name == "004_add_comments.py"

        content = path.read_text()
        assert 'version = "004"' in content

    def test_create_sanitizes_name(self, migrations_dir: Path):
        path = MigrationRunner.create_migration(migrations_dir, "Add User Table")
        assert path.name == "001_add_user_table.py"

    def test_create_handles_hyphens(self, migrations_dir: Path):
        path = MigrationRunner.create_migration(migrations_dir, "add-user-table")
        assert path.name == "001_add_user_table.py"

    def test_created_migration_is_loadable(self, migrations_dir: Path):
        MigrationRunner.create_migration(migrations_dir, "test")
        runner = MigrationRunner(migrations_dir=migrations_dir)
        migrations = runner.discover()
        assert len(migrations) == 1
        assert migrations[0].version == "001"
        assert migrations[0].description == "test"

    def test_create_in_nonexistent_dir(self, tmp_path: Path):
        new_dir = tmp_path / "new" / "migrations"
        path = MigrationRunner.create_migration(new_dir, "first")
        assert path.exists()
        assert new_dir.is_dir()


# ============================================================================
# Pending Tests
# ============================================================================


class TestPending:
    """Tests for the pending() method."""

    def test_all_pending(self, sample_migrations: Path):
        conn, cursor = _mock_connection()
        runner = MigrationRunner(
            migrations_dir=sample_migrations,
            connection_factory=lambda: conn,
        )

        with patch.object(runner, "_get_applied", return_value=[]):
            pending = runner.pending()

        assert [m.version for m in pending] == ["001", "002", "003"]

    def test_some_pending(self, sample_migrations: Path):
        conn, cursor = _mock_connection()
        runner = MigrationRunner(
            migrations_dir=sample_migrations,
            connection_factory=lambda: conn,
        )

        already = [
            AppliedMigration(version="001", description="a", checksum="x", applied_at=datetime.now(UTC)),
        ]
        with patch.object(runner, "_get_applied", return_value=already):
            pending = runner.pending()

        assert [m.version for m in pending] == ["002", "003"]

    def test_none_pending(self, sample_migrations: Path):
        conn, cursor = _mock_connection()
        runner = MigrationRunner(
            migrations_dir=sample_migrations,
            connection_factory=lambda: conn,
        )
        migrations = runner.discover()

        all_applied = [
            AppliedMigration(
                version=m.version,
                description=m.description,
                checksum=m.checksum,
                applied_at=datetime.now(UTC),
            )
            for m in migrations
        ]
        with patch.object(runner, "_get_applied", return_value=all_applied):
            pending = runner.pending()

        assert pending == []


# ============================================================================
# 001_initial_schema Migration Tests
# ============================================================================


class TestInitialSchemaMigration:
    """Tests that the 001_initial_schema migration module is well-formed."""

    def test_can_import(self):
        """The initial schema migration can be loaded."""
        repo_root = Path(__file__).resolve().parent.parent.parent
        migrations_dir = repo_root / "migrations"
        runner = MigrationRunner(migrations_dir=migrations_dir)
        migrations = runner.discover()
        assert len(migrations) >= 1
        assert migrations[0].version == "001"
        assert migrations[0].description == "initial_schema"

    def test_has_up_and_down(self):
        repo_root = Path(__file__).resolve().parent.parent.parent
        migrations_dir = repo_root / "migrations"
        runner = MigrationRunner(migrations_dir=migrations_dir)
        migrations = runner.discover()
        m = migrations[0]
        assert callable(m.module.up)
        assert callable(m.module.down)

    def test_checksum_is_stable(self):
        repo_root = Path(__file__).resolve().parent.parent.parent
        migrations_dir = repo_root / "migrations"
        runner1 = MigrationRunner(migrations_dir=migrations_dir)
        m1 = runner1.discover()

        runner2 = MigrationRunner(migrations_dir=migrations_dir)
        m2 = runner2.discover()

        assert m1[0].checksum == m2[0].checksum


# ============================================================================
# Connection Factory Tests
# ============================================================================


class TestConnectionFactory:
    """Tests for connection factory handling."""

    def test_custom_factory_used(self, sample_migrations: Path):
        conn, cursor = _mock_connection()
        factory_called = []

        def factory():
            factory_called.append(True)
            return conn

        runner = MigrationRunner(
            migrations_dir=sample_migrations,
            connection_factory=factory,
        )

        with patch.object(runner, "_get_applied", return_value=[]):
            runner.pending()

        assert len(factory_called) == 1

    def test_default_factory_uses_core_db(self, sample_migrations: Path):
        runner = MigrationRunner(migrations_dir=sample_migrations)
        # Verify it tries to use core.db when no factory is provided
        with patch("valence.core.migrations.MigrationRunner._get_connection") as mock_get:
            mock_conn, _ = _mock_connection()
            mock_get.return_value = mock_conn
            with patch.object(runner, "_get_applied", return_value=[]):
                runner.status()
            mock_get.assert_called()
