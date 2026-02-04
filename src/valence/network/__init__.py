"""
Valence Network - E2E encrypted relay protocol.

This module provides end-to-end encryption for messages relayed through
router nodes, ensuring routers cannot read message content.
"""

from valence.network.crypto import (
    KeyPair,
    generate_identity_keypair,
    generate_encryption_keypair,
    encrypt_message,
    decrypt_message,
)
from valence.network.messages import RelayMessage, DeliverPayload
from valence.network.router import RouterNode, Connection, QueuedMessage

__all__ = [
    "KeyPair",
    "generate_identity_keypair",
    "generate_encryption_keypair",
    "encrypt_message",
    "decrypt_message",
    "RelayMessage",
    "DeliverPayload",
    "RouterNode",
    "Connection",
    "QueuedMessage",
]
