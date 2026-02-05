# Database Migrations

## Structure

```
src/valence/substrate/
├── schema.sql              # Complete schema for fresh installs
├── procedures.sql          # Stored procedures
└── migrations/
    └── NNN_description.sql # Incremental migrations
```

## Fresh Install vs Upgrade

**Fresh install:** Apply `schema.sql` + `procedures.sql` only.

**Upgrade:** Apply migrations in order since last version.

## Migration Best Practices

### During Development

1. Create numbered migrations: `migrations/NNN_description.sql`
2. Make migrations **idempotent** where possible (`IF NOT EXISTS`, `IF EXISTS`)
3. Never modify existing migrations — create new ones
4. Test both fresh install and upgrade paths

### At Release Time

**Squash migrations into schema.sql:**

1. Apply all migrations to a fresh DB
2. Dump the schema: `pg_dump --schema-only`
3. Replace `schema.sql` with the dump
4. Archive old migrations to `migrations/archive/vX.Y.Z/`
5. Start fresh with new migrations for next cycle

This ensures:
- `schema.sql` is always the complete, current schema
- Fresh installs don't run 50+ migrations
- Upgrade path is clear (apply migrations since your version)

### Migration Naming

```
NNN_short_description.sql

Examples:
001_add_federation.sql
002_add_sharing.sql
003_add_unique_constraints.sql
```

After release, reset to `001_*` for the next development cycle.

## Schema Changes and Versioning

| Change Type | Version Bump | Example |
|-------------|--------------|---------|
| New table/column (additive) | MINOR | Add `audit_logs` table |
| New index | PATCH | Add index for performance |
| Drop/rename column | MAJOR | Rename `user_id` to `actor_id` |
| Drop table | MAJOR | Remove deprecated table |
| Change column type | MAJOR | `VARCHAR` → `UUID` |

**Rule:** If existing data or queries would break, it's a MAJOR bump.

## Running Migrations

```bash
# Fresh install
valence db init

# Check migration status
valence db status

# Apply pending migrations
valence db migrate

# Rollback (if supported)
valence db rollback
```

## Writing Safe Migrations

### DO:
```sql
-- Idempotent creation
CREATE TABLE IF NOT EXISTS new_table (...);
CREATE INDEX IF NOT EXISTS idx_name ON table(column);
ALTER TABLE t ADD COLUMN IF NOT EXISTS new_col TYPE;

-- Wrap in transaction
BEGIN;
-- changes
COMMIT;
```

### DON'T:
```sql
-- Non-idempotent (fails if exists)
CREATE TABLE new_table (...);

-- Data loss risk without backup
DROP TABLE important_data;
ALTER TABLE t DROP COLUMN used_column;
```

## Pre-Release Checklist

- [ ] All migrations tested on fresh DB
- [ ] All migrations tested on previous version DB
- [ ] `schema.sql` updated with squashed migrations
- [ ] Old migrations archived
- [ ] Version bumped appropriately (MAJOR/MINOR/PATCH)
- [ ] CHANGELOG updated with schema changes
