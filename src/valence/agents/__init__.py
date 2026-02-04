"""Valence Agents - Interface agents for different platforms."""

try:
    from .matrix_bot import ValenceBot, main
    __all__ = ["ValenceBot", "main"]
except ImportError:
    # matrix-nio not installed - Matrix bot unavailable
    ValenceBot = None  # type: ignore[assignment,misc]
    main = None  # type: ignore[assignment]
    __all__ = []
