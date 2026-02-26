#!/usr/bin/env bash
# On-demand test PostgreSQL container for Valence.
# Uses port 5435 to avoid conflicts with:
#   - Default postgres (5432)
#   - Real valence DB (5433)
#   - Federation peer (5434)
#
# All psql commands run via docker exec (no local psql needed).
#
# Usage:
#   ./scripts/test-db.sh up       Start container and apply schema
#   ./scripts/test-db.sh down     Stop and remove container
#   ./scripts/test-db.sh status   Show container status
#   ./scripts/test-db.sh env      Print shell exports for test config

set -euo pipefail

CONTAINER_NAME="valence-test-pg"
IMAGE="pgvector/pgvector:pg16"
PORT=5435
DB_NAME="valence_test"
DB_USER="valence"
DB_PASS="testpass"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Run psql inside the container
dpsql() {
    docker exec -e PGPASSWORD="$DB_PASS" "$CONTAINER_NAME" \
        psql -U "$DB_USER" -d "${1:-$DB_NAME}" -q "${@:2}"
}

wait_for_pg() {
    local max_attempts=30
    local attempt=0
    while ! docker exec -e PGPASSWORD="$DB_PASS" "$CONTAINER_NAME" \
            pg_isready -U "$DB_USER" -d "$DB_NAME" >/dev/null 2>&1; do
        attempt=$((attempt + 1))
        if [ "$attempt" -ge "$max_attempts" ]; then
            echo "ERROR: PostgreSQL did not become ready in time" >&2
            return 1
        fi
        sleep 1
    done
}

apply_schema() {
    echo "Applying schema..."

    # Create pgvector extension
    dpsql "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null || true

    # Core schema
    docker cp "$PROJECT_DIR/src/valence/substrate/schema.sql" "$CONTAINER_NAME:/tmp/schema.sql"
    dpsql "$DB_NAME" -f /tmp/schema.sql

    # Procedures
    docker cp "$PROJECT_DIR/src/valence/substrate/procedures.sql" "$CONTAINER_NAME:/tmp/procedures.sql"
    dpsql "$DB_NAME" -f /tmp/procedures.sql

    # Migrations (SQL files only, in order)
    for migration in "$PROJECT_DIR"/migrations/*.sql; do
        [ -f "$migration" ] || continue
        local basename
        basename="$(basename "$migration")"
        echo "  $basename"
        docker cp "$migration" "$CONTAINER_NAME:/tmp/$basename"
        dpsql "$DB_NAME" -f "/tmp/$basename" 2>/dev/null || true
    done

    echo "Schema applied."
}

cmd_up() {
    # Check if already running
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container '$CONTAINER_NAME' is already running on port $PORT."
        return 0
    fi

    # Remove stopped container if exists
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

    echo "Starting test PostgreSQL on port $PORT..."
    docker run -d \
        --name "$CONTAINER_NAME" \
        -e POSTGRES_USER="$DB_USER" \
        -e POSTGRES_PASSWORD="$DB_PASS" \
        -e POSTGRES_DB="$DB_NAME" \
        -p "${PORT}:5432" \
        "$IMAGE" >/dev/null

    echo "Waiting for PostgreSQL to be ready..."
    wait_for_pg

    apply_schema
    echo "Test database ready on localhost:$PORT"
}

cmd_down() {
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Stopping and removing '$CONTAINER_NAME'..."
        docker rm -f "$CONTAINER_NAME" >/dev/null
        echo "Done."
    else
        echo "Container '$CONTAINER_NAME' not found."
    fi
}

cmd_status() {
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "RUNNING on port $PORT"
        docker ps --filter "name=$CONTAINER_NAME" --format "table {{.ID}}\t{{.Status}}\t{{.Ports}}"
    elif docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "STOPPED"
        docker ps -a --filter "name=$CONTAINER_NAME" --format "table {{.ID}}\t{{.Status}}"
    else
        echo "NOT FOUND"
    fi
}

cmd_env() {
    cat <<EOF
export VALENCE_DB_HOST=localhost
export VALENCE_DB_PORT=$PORT
export VALENCE_DB_NAME=$DB_NAME
export VALENCE_DB_USER=$DB_USER
export VALENCE_DB_PASSWORD=$DB_PASS
EOF
}

case "${1:-help}" in
    up)     cmd_up ;;
    down)   cmd_down ;;
    status) cmd_status ;;
    env)    cmd_env ;;
    *)
        echo "Usage: $0 {up|down|status|env}"
        exit 1
        ;;
esac
