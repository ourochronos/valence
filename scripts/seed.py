#!/usr/bin/env python3
"""
Seed the Valence Knowledge Base with founding documents.
Run from repo root: python scripts/seed.py
"""

import sqlite3
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

DB_PATH = Path(__file__).parent.parent / "valence.kb.sqlite"
SCHEMA_PATH = Path(__file__).parent.parent / "migrations" / "schema.sql"
DOCS_PATH = Path(__file__).parent.parent / "docs"

def generate_id(prefix=""):
    """Generate a short, readable ID."""
    short_uuid = uuid4().hex[:8]
    return f"{prefix}_{short_uuid}" if prefix else short_uuid

def file_hash(filepath):
    """Get SHA256 hash of file contents."""
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]

def init_db(conn):
    """Initialize database with schema."""
    with open(SCHEMA_PATH, 'r') as f:
        conn.executescript(f.read())
    conn.commit()

def add_entry(conn, entry_id, entry_type, content, summary=None, source=None, 
              source_type=None, parent_id=None, confidence=1.0, created_at=None,
              tags=None, visibility='project', filepath=None):
    """Add an entry with optional modules."""
    now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    created = created_at or now
    
    conn.execute("""
        INSERT INTO entries (id, type, content, summary, created_at, ingested_at, 
                            modified_at, confidence, source, source_type, parent_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (entry_id, entry_type, content, summary, created, now, now, 
          confidence, source, source_type, parent_id))
    
    if tags:
        for tag in tags:
            conn.execute("INSERT INTO tags (entry_id, tag) VALUES (?, ?)", 
                        (entry_id, tag))
    
    conn.execute("""
        INSERT INTO scope (entry_id, visibility, owned_by)
        VALUES (?, ?, ?)
    """, (entry_id, visibility, 'chris'))
    
    if filepath:
        conn.execute("""
            INSERT INTO artifacts (entry_id, filepath, filehash, last_synced_at)
            VALUES (?, ?, ?, ?)
        """, (entry_id, filepath, file_hash(DOCS_PATH.parent / filepath), now))
    
    return entry_id

def add_relationship(conn, from_id, to_id, rel_type, reasoning=None, strength=1.0):
    """Add a relationship between entries."""
    rel_id = generate_id("rel")
    conn.execute("""
        INSERT INTO relationships (id, from_id, to_id, type, strength, reasoning)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (rel_id, from_id, to_id, rel_type, strength, reasoning))
    return rel_id

def seed_founding_documents(conn):
    """Seed the KB with the founding documents and their structure."""
    
    # Creation timestamp (when these were actually created in conversation)
    founding_date = "2024-12-14T00:00:00Z"  # Approximate
    
    # Root entry for the project
    root_id = add_entry(
        conn,
        entry_id="valence_root",
        entry_type="project",
        content="Valence: A universal, AI-driven platform where users interact with services through personal agents while maintaining data ownership.",
        summary="Valence meta-project root",
        source="founding conversation with Claude",
        source_type="conversation",
        created_at=founding_date,
        tags=["root", "meta"],
        visibility="project"
    )
    
    # Founding documents as artifacts
    principles_id = add_entry(
        conn,
        entry_id="doc_principles",
        entry_type="artifact",
        content="The constitutional principles that constrain what Valence can become.",
        summary="Founding principles document",
        source="founding conversation with Claude",
        source_type="conversation",
        parent_id=root_id,
        created_at=founding_date,
        tags=["founding", "principles", "constitution"],
        filepath="docs/PRINCIPLES.md"
    )
    
    system_id = add_entry(
        conn,
        entry_id="doc_system",
        entry_type="artifact",
        content="The system outline describing architecture derived from principles.",
        summary="System architecture outline",
        source="founding conversation with Claude",
        source_type="conversation",
        parent_id=root_id,
        created_at=founding_date,
        tags=["founding", "architecture", "system"],
        filepath="docs/SYSTEM.md"
    )
    
    unknowns_id = add_entry(
        conn,
        entry_id="doc_unknowns",
        entry_type="artifact",
        content="Known unknowns - honest documentation of gaps and open questions.",
        summary="Known unknowns document",
        source="founding conversation with Claude",
        source_type="conversation",
        parent_id=root_id,
        created_at=founding_date,
        tags=["founding", "unknowns", "gaps"],
        filepath="docs/UNKNOWNS.md"
    )
    
    # Individual principles as entries
    principles = [
        ("user_sovereignty", "User Sovereignty", 
         "Users own their data. Your agent represents you. Data never leaves user control without explicit, informed consent."),
        ("collective_emergence", "Collective Emergence",
         "Emergence over prescription. Individual value drives adoption. Aggregation exists only to increase value for users."),
        ("structural_integrity", "Structural Integrity",
         "Structurally incapable of betrayal. Security through structure, not policy. N integrations, not N!"),
        ("openness_resilience", "Openness as Resilience",
         "Transparency by default. Designed to survive being stolen. Invites criticism. AI-centric from the ground up."),
        ("mission_permanence", "Mission Permanence",
         "Structure prevents capture. Principles are constitutional."),
        ("principles_foundation", "Principles as Foundation",
         "Principles-based at every level. When in doubt, derive from principles."),
    ]
    
    for p_id, p_name, p_content in principles:
        entry_id = add_entry(
            conn,
            entry_id=f"principle_{p_id}",
            entry_type="principle",
            content=p_content,
            summary=p_name,
            source="docs/PRINCIPLES.md",
            source_type="document",
            parent_id=principles_id,
            created_at=founding_date,
            tags=["principle", "founding"],
            confidence=1.0  # Principles are definitional
        )
        add_relationship(conn, entry_id, principles_id, "derived_from",
                        "Extracted from founding principles document")
    
    # Key unknowns as entries
    unknowns = [
        ("sustainability", "Sustainability",
         "How does this survive economically without compromising principles?"),
        ("kb_mechanics", "Knowledge Base Mechanics",
         "How does the agent actually learn about the user?"),
        ("trust_verification", "Agent-User Trust Verification",
         "How does a user verify their agent is actually aligned to them?"),
        ("adoption_path", "Adoption Path",
         "What's the wedge? What's valuable enough at small scale?"),
        ("scope_boundaries", "Scope and Boundaries",
         "What is this, and what is it not?"),
        ("timing", "Timing",
         "Does this have runway before the landscape shifts?"),
    ]
    
    for u_id, u_name, u_content in unknowns:
        entry_id = add_entry(
            conn,
            entry_id=f"unknown_{u_id}",
            entry_type="unknown",
            content=u_content,
            summary=u_name,
            source="docs/UNKNOWNS.md",
            source_type="document",
            parent_id=unknowns_id,
            created_at=founding_date,
            tags=["unknown", "gap", "founding"],
            confidence=0.5  # Unknowns are uncertain by definition
        )
        add_relationship(conn, entry_id, unknowns_id, "derived_from",
                        "Extracted from known unknowns document")
    
    # The name decision
    name_id = add_entry(
        conn,
        entry_id="decision_name",
        entry_type="decision",
        content="The project is named 'Valence'. In chemistry, valence determines how atoms bond. In psychology, it's the intrinsic quality that makes something attractive or aversive. Your values are your valence. Contracts to 'Val'.",
        summary="Project name: Valence",
        source="founding conversation with Claude",
        source_type="conversation",
        parent_id=root_id,
        created_at=founding_date,
        tags=["decision", "name", "founding"],
        confidence=1.0
    )
    
    conn.commit()
    print(f"Seeded {conn.execute('SELECT COUNT(*) FROM entries').fetchone()[0]} entries")
    print(f"Created {conn.execute('SELECT COUNT(*) FROM relationships').fetchone()[0]} relationships")

def main():
    """Main entry point."""
    # Remove existing DB for clean seed
    if DB_PATH.exists():
        response = input(f"{DB_PATH} exists. Overwrite? [y/N] ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        DB_PATH.unlink()
    
    conn = sqlite3.connect(DB_PATH)
    try:
        print(f"Initializing database at {DB_PATH}")
        init_db(conn)
        
        print("Seeding founding documents...")
        seed_founding_documents(conn)
        
        print("Done.")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
