"""Valence - Personal knowledge substrate for AI agents.

Valence provides:
- Knowledge substrate (beliefs, entities, tensions)
- Conversation tracking (sessions, exchanges, patterns)
- Claude Code integration (plugin, hooks, skills)
- Multi-platform agents (Matrix, API)

Brick packages (oro-*) provide: consensus, federation, privacy,
network, embeddings, crypto, identity, storage, compliance.
"""

__version__ = "1.0.0"

from . import (
    agents as agents,
)
from . import (
    core as core,
)
from . import (
    substrate as substrate,
)
from . import (
    vkb as vkb,
)
