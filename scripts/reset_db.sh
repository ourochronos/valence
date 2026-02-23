#!/usr/bin/env bash
set -euo pipefail

DB_HOST="${VALENCE_DB_HOST:-127.0.0.1}"
DB_PORT="${VALENCE_DB_PORT:-5433}"
DB_NAME="${VALENCE_DB_NAME:-valence}"
DB_USER="${VALENCE_DB_USER:-valence}"
DB_PASSWORD="${VALENCE_DB_PASSWORD:-valence}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCHEMA_FILE="$SCRIPT_DIR/../schema.sql"

if [[ ! -f "$SCHEMA_FILE" ]]; then
    echo "ERROR: schema.sql not found at $SCHEMA_FILE"
    exit 1
fi

echo "========================================="
echo "Valence Database Reset"
echo "========================================="
echo "Host:     $DB_HOST:$DB_PORT"
echo "Database: $DB_NAME"
echo "User:     $DB_USER"
echo ""
echo "⚠️  WARNING: This will DROP and recreate the '$DB_NAME' database."
echo "⚠️  ALL DATA WILL BE LOST."
echo ""
read -p "Continue? [y/N] " confirm
[[ "$confirm" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }

echo ""
echo "Dropping database..."
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "DROP DATABASE IF EXISTS $DB_NAME;"

echo "Creating database..."
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "CREATE DATABASE $DB_NAME;"

echo "Applying schema..."
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$SCHEMA_FILE"

echo ""
echo "✅ Database reset complete."
