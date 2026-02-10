"""
Serialization codecs for VFP messages over libp2p.

Two framing strategies:

* **StreamCodec** — length-prefixed binary framing for 1:1 stream protocols
  (sync, auth, trust).  Each frame is ``<4-byte big-endian length><JSON payload>``.

* **GossipSubCodec** — plain UTF-8 JSON for pub/sub topics (beliefs, peers).
  GossipSub already handles message boundaries, so no length prefix is needed.

Both codecs reuse the existing federation message ``to_dict()`` / ``parse_message()``
round-trip defined in ``valence.federation.protocol``.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any

from oro_federation.protocol import ProtocolMessage, parse_message

# Maximum allowed frame payload (8 MiB).  Protects against OOM from a
# malicious peer sending a huge length prefix.
MAX_FRAME_SIZE: int = 8 * 1024 * 1024

# 4-byte big-endian unsigned int
_LENGTH_STRUCT = struct.Struct("!I")
FRAME_HEADER_SIZE: int = _LENGTH_STRUCT.size  # 4


class CodecError(Exception):
    """Raised when encoding or decoding fails."""


# ============================================================================
# Stream codec (length-prefixed JSON)
# ============================================================================


class StreamCodec:
    """Length-prefixed JSON framing for libp2p stream protocols.

    Wire format::

        +-------------------+-----------------------------+
        | 4 bytes (BE u32)  |  JSON payload (UTF-8)       |
        | = payload length  |                             |
        +-------------------+-----------------------------+

    Usage::

        codec = StreamCodec()

        # Encode
        frame = codec.encode_message(some_protocol_message)

        # Decode (from a buffer that may contain partial data)
        msg, consumed = codec.decode_frame(buffer)
    """

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    @staticmethod
    def encode(data: dict[str, Any]) -> bytes:
        """Encode a dict as a length-prefixed JSON frame.

        Args:
            data: JSON-serializable dictionary.

        Returns:
            ``<4-byte length><JSON payload>`` bytes.

        Raises:
            CodecError: If the payload exceeds ``MAX_FRAME_SIZE``.
        """
        payload = json.dumps(data, separators=(",", ":"), default=str).encode("utf-8")
        if len(payload) > MAX_FRAME_SIZE:
            raise CodecError(f"Payload size {len(payload)} exceeds maximum {MAX_FRAME_SIZE}")
        return _LENGTH_STRUCT.pack(len(payload)) + payload

    @staticmethod
    def encode_message(message: ProtocolMessage) -> bytes:
        """Encode a ``ProtocolMessage`` as a length-prefixed frame.

        Convenience wrapper around :meth:`encode` that calls ``message.to_dict()``
        first.
        """
        return StreamCodec.encode(message.to_dict())

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    @staticmethod
    def decode_frame(buf: bytes | bytearray | memoryview) -> tuple[dict[str, Any], int]:
        """Decode one frame from *buf*.

        Args:
            buf: Buffer that starts with a length-prefixed frame (may contain
                 trailing data).

        Returns:
            ``(parsed_dict, bytes_consumed)`` — the decoded dict and how many
            bytes of *buf* were consumed.

        Raises:
            CodecError: If the frame is incomplete, oversized, or contains
                invalid JSON.
        """
        if len(buf) < FRAME_HEADER_SIZE:
            raise CodecError("Incomplete frame header")

        (payload_len,) = _LENGTH_STRUCT.unpack_from(buf, 0)

        if payload_len > MAX_FRAME_SIZE:
            raise CodecError(f"Frame payload size {payload_len} exceeds maximum {MAX_FRAME_SIZE}")

        total = FRAME_HEADER_SIZE + payload_len
        if len(buf) < total:
            raise CodecError(f"Incomplete frame body: need {total} bytes, have {len(buf)}")

        payload_bytes = bytes(buf[FRAME_HEADER_SIZE:total])
        try:
            data = json.loads(payload_bytes)
        except json.JSONDecodeError as exc:
            raise CodecError(f"Invalid JSON in frame payload: {exc}") from exc

        if not isinstance(data, dict):
            raise CodecError(f"Expected JSON object, got {type(data).__name__}")

        return data, total

    @staticmethod
    def decode_message(buf: bytes | bytearray | memoryview) -> tuple[ProtocolMessage | None, int]:
        """Decode one frame and parse into a ``ProtocolMessage``.

        Returns:
            ``(message_or_None, bytes_consumed)``.  The message is ``None``
            when the JSON is valid but does not match a known VFP message type.
        """
        data, consumed = StreamCodec.decode_frame(buf)
        return parse_message(data), consumed

    @staticmethod
    def try_decode_frame(buf: bytes | bytearray | memoryview) -> tuple[dict[str, Any] | None, int]:
        """Non-raising variant of :meth:`decode_frame`.

        Returns ``(None, 0)`` when the buffer does not yet contain a
        complete frame (useful when reading from a stream incrementally).

        Raises:
            CodecError: Only for *irrecoverable* errors (oversized frame,
                bad JSON).  Incomplete data returns ``(None, 0)`` instead.
        """
        if len(buf) < FRAME_HEADER_SIZE:
            return None, 0

        (payload_len,) = _LENGTH_STRUCT.unpack_from(buf, 0)

        if payload_len > MAX_FRAME_SIZE:
            raise CodecError(f"Frame payload size {payload_len} exceeds maximum {MAX_FRAME_SIZE}")

        total = FRAME_HEADER_SIZE + payload_len
        if len(buf) < total:
            return None, 0

        payload_bytes = bytes(buf[FRAME_HEADER_SIZE:total])
        try:
            data = json.loads(payload_bytes)
        except json.JSONDecodeError as exc:
            raise CodecError(f"Invalid JSON in frame payload: {exc}") from exc

        if not isinstance(data, dict):
            raise CodecError(f"Expected JSON object, got {type(data).__name__}")

        return data, total


# ============================================================================
# GossipSub codec (plain JSON)
# ============================================================================


class GossipSubCodec:
    """Plain JSON codec for GossipSub messages.

    GossipSub already handles message delimiting, so we just need
    JSON ↔ bytes conversion.
    """

    @staticmethod
    def encode(data: dict[str, Any]) -> bytes:
        """Encode a dict as compact JSON bytes.

        Args:
            data: JSON-serializable dictionary.

        Returns:
            UTF-8 encoded JSON bytes.

        Raises:
            CodecError: If the payload exceeds ``MAX_FRAME_SIZE``.
        """
        payload = json.dumps(data, separators=(",", ":"), default=str).encode("utf-8")
        if len(payload) > MAX_FRAME_SIZE:
            raise CodecError(f"Payload size {len(payload)} exceeds maximum {MAX_FRAME_SIZE}")
        return payload

    @staticmethod
    def encode_message(message: ProtocolMessage) -> bytes:
        """Encode a ``ProtocolMessage`` as GossipSub JSON."""
        return GossipSubCodec.encode(message.to_dict())

    @staticmethod
    def decode(data: bytes) -> dict[str, Any]:
        """Decode GossipSub message bytes into a dict.

        Args:
            data: UTF-8 JSON bytes.

        Returns:
            Parsed dictionary.

        Raises:
            CodecError: If the data is not valid JSON or exceeds size limits.
        """
        if len(data) > MAX_FRAME_SIZE:
            raise CodecError(f"Message size {len(data)} exceeds maximum {MAX_FRAME_SIZE}")
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as exc:
            raise CodecError(f"Invalid JSON in GossipSub message: {exc}") from exc

        if not isinstance(parsed, dict):
            raise CodecError(f"Expected JSON object, got {type(parsed).__name__}")

        return parsed

    @staticmethod
    def decode_message(data: bytes) -> ProtocolMessage | None:
        """Decode GossipSub bytes and parse into a ``ProtocolMessage``.

        Returns ``None`` if the JSON is valid but not a recognised VFP type.
        """
        return parse_message(GossipSubCodec.decode(data))


# ============================================================================
# Streaming buffer helper
# ============================================================================


@dataclass
class StreamBuffer:
    """Accumulates bytes from a libp2p stream and yields complete frames.

    Typical usage in an async read loop::

        buf = StreamBuffer()
        async for chunk in stream:
            buf.feed(chunk)
            for msg_dict in buf.drain():
                handle(msg_dict)
    """

    _buf: bytearray

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, data: bytes | bytearray | memoryview) -> None:
        """Append incoming bytes."""
        self._buf.extend(data)

    def drain(self) -> list[dict[str, Any]]:
        """Extract all complete frames currently in the buffer.

        Returns:
            List of decoded dicts (may be empty).

        Raises:
            CodecError: On irrecoverable decoding errors.
        """
        results: list[dict[str, Any]] = []
        while True:
            parsed, consumed = StreamCodec.try_decode_frame(self._buf)
            if parsed is None:
                break
            results.append(parsed)
            del self._buf[:consumed]
        return results

    def __len__(self) -> int:
        """Current buffer size in bytes."""
        return len(self._buf)
