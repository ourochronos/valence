"""Tests for valence.transport.message_codec — serialization codecs."""

from __future__ import annotations

import json
import struct

import pytest
from oro_federation.protocol import (
    AuthChallengeRequest,
    SyncRequest,
)

from valence.transport.message_codec import (
    MAX_FRAME_SIZE,
    CodecError,
    GossipSubCodec,
    StreamBuffer,
    StreamCodec,
)

# ============================================================================
# StreamCodec — length-prefixed framing
# ============================================================================


class TestStreamCodecEncode:
    """StreamCodec.encode / encode_message."""

    def test_encode_basic_dict(self) -> None:
        data = {"type": "SYNC_REQUEST", "page_size": 100}
        frame = StreamCodec.encode(data)

        # First 4 bytes = big-endian u32 length
        payload_len = struct.unpack("!I", frame[:4])[0]
        payload = frame[4:]
        assert len(payload) == payload_len

        # Payload is valid JSON
        decoded = json.loads(payload)
        assert decoded["type"] == "SYNC_REQUEST"
        assert decoded["page_size"] == 100

    def test_encode_message(self) -> None:
        msg = SyncRequest(page_size=50, domains=["science"])
        frame = StreamCodec.encode_message(msg)

        payload = json.loads(frame[4:])
        assert payload["type"] == "SYNC_REQUEST"
        assert payload["page_size"] == 50
        assert payload["domains"] == ["science"]

    def test_encode_rejects_oversized_payload(self) -> None:
        data = {"big": "x" * (MAX_FRAME_SIZE + 1)}
        with pytest.raises(CodecError, match="exceeds maximum"):
            StreamCodec.encode(data)


class TestStreamCodecDecode:
    """StreamCodec.decode_frame / decode_message."""

    def test_roundtrip(self) -> None:
        original = {"type": "AUTH_CHALLENGE", "client_did": "did:vkb:web:test"}
        frame = StreamCodec.encode(original)
        decoded, consumed = StreamCodec.decode_frame(frame)
        assert decoded == original
        assert consumed == len(frame)

    def test_message_roundtrip(self) -> None:
        msg = AuthChallengeRequest(client_did="did:vkb:web:alice")
        frame = StreamCodec.encode_message(msg)
        parsed, consumed = StreamCodec.decode_message(frame)
        assert parsed is not None
        assert isinstance(parsed, AuthChallengeRequest)
        assert parsed.client_did == "did:vkb:web:alice"
        assert consumed == len(frame)

    def test_decode_with_trailing_data(self) -> None:
        frame = StreamCodec.encode({"a": 1})
        buf = frame + b"extra_data"
        decoded, consumed = StreamCodec.decode_frame(buf)
        assert decoded == {"a": 1}
        assert consumed == len(frame)

    def test_decode_incomplete_header_raises(self) -> None:
        with pytest.raises(CodecError, match="Incomplete frame header"):
            StreamCodec.decode_frame(b"\x00\x00")

    def test_decode_incomplete_body_raises(self) -> None:
        # Header says 100 bytes, but only 10 provided
        header = struct.pack("!I", 100)
        with pytest.raises(CodecError, match="Incomplete frame body"):
            StreamCodec.decode_frame(header + b"x" * 10)

    def test_decode_oversized_frame_raises(self) -> None:
        header = struct.pack("!I", MAX_FRAME_SIZE + 1)
        with pytest.raises(CodecError, match="exceeds maximum"):
            StreamCodec.decode_frame(header + b"\x00" * 10)

    def test_decode_invalid_json_raises(self) -> None:
        bad_payload = b"not json"
        header = struct.pack("!I", len(bad_payload))
        with pytest.raises(CodecError, match="Invalid JSON"):
            StreamCodec.decode_frame(header + bad_payload)

    def test_decode_non_object_json_raises(self) -> None:
        payload = b'"just a string"'
        header = struct.pack("!I", len(payload))
        with pytest.raises(CodecError, match="Expected JSON object"):
            StreamCodec.decode_frame(header + payload)

    def test_decode_unknown_message_type_returns_none(self) -> None:
        frame = StreamCodec.encode({"type": "TOTALLY_UNKNOWN"})
        msg, _ = StreamCodec.decode_message(frame)
        assert msg is None


class TestStreamCodecTryDecode:
    """StreamCodec.try_decode_frame — non-raising variant."""

    def test_returns_none_on_incomplete_header(self) -> None:
        result, consumed = StreamCodec.try_decode_frame(b"\x00")
        assert result is None
        assert consumed == 0

    def test_returns_none_on_incomplete_body(self) -> None:
        header = struct.pack("!I", 100)
        result, consumed = StreamCodec.try_decode_frame(header + b"x")
        assert result is None
        assert consumed == 0

    def test_returns_data_on_complete_frame(self) -> None:
        frame = StreamCodec.encode({"ok": True})
        result, consumed = StreamCodec.try_decode_frame(frame)
        assert result == {"ok": True}
        assert consumed == len(frame)

    def test_raises_on_bad_json(self) -> None:
        payload = b"{bad"
        header = struct.pack("!I", len(payload))
        with pytest.raises(CodecError, match="Invalid JSON"):
            StreamCodec.try_decode_frame(header + payload)


# ============================================================================
# GossipSubCodec — plain JSON
# ============================================================================


class TestGossipSubCodec:
    """GossipSubCodec encode/decode."""

    def test_roundtrip(self) -> None:
        original = {"type": "SHARE_BELIEF", "beliefs": [{"content": "hello"}]}
        encoded = GossipSubCodec.encode(original)
        decoded = GossipSubCodec.decode(encoded)
        assert decoded == original

    def test_encode_is_compact(self) -> None:
        data = {"a": 1, "b": 2}
        encoded = GossipSubCodec.encode(data)
        # No spaces in compact JSON
        assert b" " not in encoded

    def test_encode_rejects_oversized(self) -> None:
        data = {"big": "x" * (MAX_FRAME_SIZE + 1)}
        with pytest.raises(CodecError, match="exceeds maximum"):
            GossipSubCodec.encode(data)

    def test_decode_rejects_oversized(self) -> None:
        with pytest.raises(CodecError, match="exceeds maximum"):
            GossipSubCodec.decode(b"x" * (MAX_FRAME_SIZE + 1))

    def test_decode_rejects_invalid_json(self) -> None:
        with pytest.raises(CodecError, match="Invalid JSON"):
            GossipSubCodec.decode(b"not json")

    def test_decode_rejects_non_object(self) -> None:
        with pytest.raises(CodecError, match="Expected JSON object"):
            GossipSubCodec.decode(b"[1, 2, 3]")

    def test_message_roundtrip(self) -> None:
        msg = AuthChallengeRequest(client_did="did:vkb:web:bob")
        encoded = GossipSubCodec.encode_message(msg)
        parsed = GossipSubCodec.decode_message(encoded)
        assert parsed is not None
        assert isinstance(parsed, AuthChallengeRequest)
        assert parsed.client_did == "did:vkb:web:bob"

    def test_decode_unknown_type_returns_none(self) -> None:
        encoded = GossipSubCodec.encode({"type": "UNKNOWN"})
        assert GossipSubCodec.decode_message(encoded) is None


# ============================================================================
# StreamBuffer — incremental frame decoder
# ============================================================================


class TestStreamBuffer:
    """StreamBuffer accumulation and draining."""

    def test_single_complete_frame(self) -> None:
        buf = StreamBuffer()
        frame = StreamCodec.encode({"msg": "hello"})
        buf.feed(frame)
        results = buf.drain()
        assert len(results) == 1
        assert results[0]["msg"] == "hello"

    def test_incremental_feeding(self) -> None:
        buf = StreamBuffer()
        frame = StreamCodec.encode({"step": "incremental"})

        # Feed byte by byte (worst case)
        for i in range(len(frame) - 1):
            buf.feed(frame[i : i + 1])
            assert buf.drain() == []  # Not complete yet

        buf.feed(frame[-1:])
        results = buf.drain()
        assert len(results) == 1
        assert results[0]["step"] == "incremental"

    def test_multiple_frames_in_one_feed(self) -> None:
        buf = StreamBuffer()
        f1 = StreamCodec.encode({"n": 1})
        f2 = StreamCodec.encode({"n": 2})
        f3 = StreamCodec.encode({"n": 3})

        buf.feed(f1 + f2 + f3)
        results = buf.drain()
        assert len(results) == 3
        assert [r["n"] for r in results] == [1, 2, 3]

    def test_partial_frame_preserved(self) -> None:
        buf = StreamBuffer()
        f1 = StreamCodec.encode({"n": 1})
        f2 = StreamCodec.encode({"n": 2})

        # Feed first frame + half of second
        half = len(f2) // 2
        buf.feed(f1 + f2[:half])
        results = buf.drain()
        assert len(results) == 1
        assert results[0]["n"] == 1

        # Feed rest of second frame
        buf.feed(f2[half:])
        results = buf.drain()
        assert len(results) == 1
        assert results[0]["n"] == 2

    def test_len_tracks_buffer_size(self) -> None:
        buf = StreamBuffer()
        assert len(buf) == 0
        buf.feed(b"hello")
        assert len(buf) == 5

    def test_empty_drain(self) -> None:
        buf = StreamBuffer()
        assert buf.drain() == []
