"""Migration 024: Add degraded column to articles and contentions.

Supports WU-13 / DR-9: explicit degraded mode for articles and contentions
that were produced via fallback (no inference backend available).

Degraded articles are queued for reprocessing via mutation_queue when
inference becomes available.  The ``degraded`` flag makes quality gaps
visible rather than silently accepting fallback output as production quality.
"""

version = "024"
description = "add_degraded_columns"

_STATEMENTS = [
    (
        "add degraded to articles",
        "ALTER TABLE articles ADD COLUMN IF NOT EXISTS degraded BOOLEAN DEFAULT FALSE",
    ),
    (
        "add degraded to contentions",
        "ALTER TABLE contentions ADD COLUMN IF NOT EXISTS degraded BOOLEAN DEFAULT FALSE",
    ),
]


def up(conn) -> None:
    with conn.cursor() as cur:
        for _desc, sql in _STATEMENTS:
            cur.execute(sql)


def down(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE articles DROP COLUMN IF EXISTS degraded")
        cur.execute("ALTER TABLE contentions DROP COLUMN IF EXISTS degraded")
