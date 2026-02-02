"""Valence - Personal knowledge substrate for AI agents.

Valence provides:
- Knowledge substrate (beliefs, entities, tensions)
- Conversation tracking (sessions, exchanges, patterns)
- Claude Code integration (plugin, hooks, skills)
- Multi-platform agents (Matrix, API)
"""

__version__ = "1.0.0"

# Core library
from . import core

# Knowledge substrate (EKB)
from . import substrate

# Conversation tracking (VKB)
from . import vkb

# Embedding infrastructure
from . import embeddings

# Agent implementations
from . import agents
