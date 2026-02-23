"""Core utilities for our-db."""


def escape_ilike(s: str) -> str:
    """Escape ILIKE metacharacters for safe SQL pattern matching.

    PostgreSQL ILIKE treats '%' as wildcard (any chars) and '_' as wildcard
    (single char). This function escapes these characters so user input is
    matched literally.

    Args:
        s: The string to escape.

    Returns:
        The escaped string safe for use in ILIKE patterns.
    """
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
