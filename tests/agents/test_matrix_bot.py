"""Tests for valence.agents.matrix_bot module.

Requires the 'matrix' extra: pip install valence[matrix]
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

# Skip all tests in this module if matrix-nio is not installed
try:
    import nio
    NIO_AVAILABLE = True
except ImportError:
    NIO_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not NIO_AVAILABLE,
    reason="matrix-nio not installed (install with: pip install valence[matrix])"
)


# ============================================================================
# Input Sanitization Tests (Security - Issue #44)
# ============================================================================

class TestInputSanitization:
    """Tests for subprocess command injection prevention.
    
    Issue #44: User input passed to subprocess must be sanitized to prevent
    command injection attacks.
    """

    def test_sanitize_removes_null_bytes(self):
        """Null bytes in input must be removed to prevent argument truncation."""
        from valence.agents.matrix_bot import sanitize_for_subprocess
        
        malicious = "hello\x00world"
        result = sanitize_for_subprocess(malicious, 1000, "test")
        
        assert "\x00" not in result
        assert result == "helloworld"

    def test_sanitize_removes_control_characters(self):
        """Control characters must be removed except whitespace."""
        from valence.agents.matrix_bot import sanitize_for_subprocess
        
        # Bell, backspace, escape, etc. should be removed
        malicious = "hello\x07\x08\x1bworld"
        result = sanitize_for_subprocess(malicious, 1000, "test")
        
        assert "\x07" not in result
        assert "\x08" not in result
        assert "\x1b" not in result
        assert result == "helloworld"

    def test_sanitize_preserves_normal_whitespace(self):
        """Normal whitespace (tab, newline) should be preserved."""
        from valence.agents.matrix_bot import sanitize_for_subprocess
        
        text = "hello\tworld\nfoo"
        result = sanitize_for_subprocess(text, 1000, "test")
        
        assert result == "hello\tworld\nfoo"

    def test_sanitize_enforces_length_limit(self):
        """Input exceeding max length must be truncated."""
        from valence.agents.matrix_bot import sanitize_for_subprocess
        
        long_text = "x" * 5000
        result = sanitize_for_subprocess(long_text, 100, "test")
        
        assert len(result) == 100

    def test_sanitize_handles_empty_input(self):
        """Empty input should return empty string."""
        from valence.agents.matrix_bot import sanitize_for_subprocess
        
        assert sanitize_for_subprocess("", 100, "test") == ""
        assert sanitize_for_subprocess(None, 100, "test") == ""  # type: ignore

    def test_validate_sender_accepts_valid_matrix_id(self):
        """Valid Matrix user IDs should pass validation."""
        from valence.agents.matrix_bot import validate_sender
        
        valid_ids = [
            "@user:matrix.org",
            "@alice:example.com",
            "@bot_123:server.net",
            "@user-name:host.domain.tld",
        ]
        
        for user_id in valid_ids:
            result = validate_sender(user_id)
            assert result == user_id

    def test_validate_sender_sanitizes_invalid_format(self):
        """Invalid Matrix user IDs should be sanitized, not rejected."""
        from valence.agents.matrix_bot import validate_sender
        
        # Contains shell metacharacters - should be sanitized
        result = validate_sender("@user$(whoami):evil.com")
        assert "$" not in result or result.count("$") == 0 or "_" in result

    def test_validate_sender_rejects_empty(self):
        """Empty sender should raise ValueError."""
        from valence.agents.matrix_bot import validate_sender
        
        with pytest.raises(ValueError, match="Empty sender"):
            validate_sender("")

    def test_validate_session_id_accepts_valid(self):
        """Valid session IDs should pass validation."""
        from valence.agents.matrix_bot import validate_session_id
        
        valid_ids = [
            "abc123",
            "session-id-123",
            "session_id_456",
            "A1B2C3",
        ]
        
        for session_id in valid_ids:
            result = validate_session_id(session_id)
            assert result == session_id

    def test_validate_session_id_accepts_none(self):
        """None session ID should return None."""
        from valence.agents.matrix_bot import validate_session_id
        
        assert validate_session_id(None) is None

    def test_validate_session_id_rejects_shell_injection(self):
        """Session ID with shell metacharacters must be rejected."""
        from valence.agents.matrix_bot import validate_session_id
        
        malicious_ids = [
            "session; rm -rf /",
            "session$(whoami)",
            "session`id`",
            "session|cat /etc/passwd",
            "session && echo pwned",
        ]
        
        for malicious in malicious_ids:
            with pytest.raises(ValueError, match="Invalid session ID"):
                validate_session_id(malicious)

    def test_sanitize_message_removes_dangerous_chars(self):
        """Messages with dangerous characters should be sanitized."""
        from valence.agents.matrix_bot import sanitize_message
        
        # Null bytes should be removed
        result = sanitize_message("hello\x00world")
        assert "\x00" not in result

    def test_sanitize_message_enforces_max_length(self):
        """Messages exceeding MAX_MESSAGE_LENGTH should be truncated."""
        from valence.agents.matrix_bot import sanitize_message, MAX_MESSAGE_LENGTH
        
        long_message = "x" * (MAX_MESSAGE_LENGTH + 1000)
        result = sanitize_message(long_message)
        
        assert len(result) <= MAX_MESSAGE_LENGTH


class TestCommandInjectionPrevention:
    """Tests that command injection attacks are blocked in _generate_response.
    
    Issue #44: Subprocess command injection risk in Matrix bot.
    """

    @pytest.fixture
    def bot(self):
        """Create bot for testing."""
        with patch("valence.agents.matrix_bot.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.user_id = "@bot:example.com"
            mock_client_cls.return_value = mock_client

            from valence.agents.matrix_bot import ValenceBot

            return ValenceBot(
                homeserver="https://matrix.example.com",
                user_id="@bot:example.com",
                password="secret",
            )

    @pytest.fixture
    def session(self):
        """Create test session."""
        from valence.agents.matrix_bot import RoomSession

        return RoomSession(
            room_id="!room:example.com",
            room_name="Test Room",
        )

    @pytest.mark.asyncio
    async def test_shell_injection_via_message_blocked(self, bot, session):
        """Shell injection attempts via message content must be blocked."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"result": "Response"})
        
        malicious_messages = [
            "; rm -rf /",
            "$(whoami)",
            "`id`",
            "| cat /etc/passwd",
            "&& echo pwned",
            "\x00; rm -rf /",  # Null byte injection
        ]
        
        for malicious in malicious_messages:
            with patch("subprocess.run", return_value=mock_result) as mock_run:
                with patch("os.path.exists", return_value=False):
                    await bot._generate_response(
                        session, 
                        "@user:example.com", 
                        malicious
                    )
                    
                    # Verify subprocess was called with sanitized input
                    # The command should be a list, not a string
                    call_args = mock_run.call_args
                    cmd = call_args[0][0]
                    
                    # Must be a list (no shell=True)
                    assert isinstance(cmd, list), "Command must be a list"
                    
                    # shell=True must not be present
                    assert call_args[1].get("shell") is not True, "shell=True is forbidden"
                    
                    # Null bytes must be removed from prompt
                    prompt_arg = cmd[cmd.index("-p") + 1]
                    assert "\x00" not in prompt_arg, "Null bytes must be removed"

    @pytest.mark.asyncio
    async def test_shell_injection_via_sender_blocked(self, bot, session):
        """Shell injection attempts via sender must be blocked."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"result": "Response"})
        
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            with patch("os.path.exists", return_value=False):
                # Malicious sender
                await bot._generate_response(
                    session,
                    "@user$(whoami):evil.com",
                    "Hello"
                )
                
                call_args = mock_run.call_args
                cmd = call_args[0][0]
                prompt_arg = cmd[cmd.index("-p") + 1]
                
                # The $(whoami) should be sanitized/escaped in the prompt
                # Either removed or the $ replaced
                assert "$(whoami)" not in prompt_arg or "_" in prompt_arg

    @pytest.mark.asyncio
    async def test_shell_injection_via_session_id_blocked(self, bot, session):
        """Shell injection attempts via session ID must be blocked."""
        # Set malicious session ID
        session.claude_session_id = "session; rm -rf /"
        
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"result": "Response"})
        
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            with patch("os.path.exists", return_value=False):
                await bot._generate_response(
                    session,
                    "@user:example.com",
                    "Hello"
                )
                
                call_args = mock_run.call_args
                cmd = call_args[0][0]
                
                # Either --resume shouldn't be in command (session was rejected)
                # Or the session ID should be sanitized
                if "--resume" in cmd:
                    resume_idx = cmd.index("--resume")
                    session_arg = cmd[resume_idx + 1]
                    # Must not contain shell metacharacters
                    assert ";" not in session_arg
                    assert " " not in session_arg

    @pytest.mark.asyncio
    async def test_invalid_session_id_starts_fresh(self, bot, session):
        """Invalid session ID should cause fresh session start."""
        session.claude_session_id = "session`id`"  # Backticks = command substitution
        
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "session_id": "new-safe-session",
            "result": "Response"
        })
        
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            with patch("os.path.exists", return_value=False):
                await bot._generate_response(
                    session,
                    "@user:example.com",
                    "Hello"
                )
                
                call_args = mock_run.call_args
                cmd = call_args[0][0]
                
                # --resume should not be present with malicious session
                # OR the session should have been cleared
                if "--resume" in cmd:
                    # If --resume is present, verify it's not the malicious value
                    resume_idx = cmd.index("--resume")
                    assert "`" not in cmd[resume_idx + 1]

    @pytest.mark.asyncio
    async def test_empty_message_after_sanitization_skipped(self, bot, session):
        """Messages that become empty after sanitization should be skipped."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"result": "Response"})
        
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            with patch("os.path.exists", return_value=False):
                # Message with only control characters
                result = await bot._generate_response(
                    session,
                    "@user:example.com",
                    "\x00\x01\x02"  # Only null and control chars
                )
                
                # Should return None without calling subprocess
                assert result is None

    @pytest.mark.asyncio  
    async def test_dos_via_long_message_prevented(self, bot, session):
        """Extremely long messages should be truncated to prevent DoS."""
        from valence.agents.matrix_bot import MAX_MESSAGE_LENGTH
        
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"result": "Response"})
        
        # Very long message (100KB)
        huge_message = "x" * 100000
        
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            with patch("os.path.exists", return_value=False):
                await bot._generate_response(
                    session,
                    "@user:example.com",
                    huge_message
                )
                
                call_args = mock_run.call_args
                cmd = call_args[0][0]
                prompt_arg = cmd[cmd.index("-p") + 1]
                
                # Prompt should be truncated
                # Account for "User @user:example.com says: " prefix
                assert len(prompt_arg) <= MAX_MESSAGE_LENGTH + 100


# ============================================================================
# RoomSession Tests
# ============================================================================

class TestRoomSession:
    """Tests for RoomSession dataclass."""

    def test_create(self):
        """Should create room session with required fields."""
        from valence.agents.matrix_bot import RoomSession

        session = RoomSession(
            room_id="!room123:example.com",
            room_name="Test Room",
        )

        assert session.room_id == "!room123:example.com"
        assert session.room_name == "Test Room"
        assert session.claude_session_id is None
        assert session.vkb_session_id is None
        assert session.message_count == 0
        assert session.last_active == 0

    def test_create_with_all_fields(self):
        """Should create room session with all fields."""
        from valence.agents.matrix_bot import RoomSession

        session = RoomSession(
            room_id="!room123:example.com",
            room_name="Test Room",
            claude_session_id="claude-123",
            vkb_session_id="vkb-456",
            message_count=10,
            last_active=1234567890.0,
        )

        assert session.claude_session_id == "claude-123"
        assert session.vkb_session_id == "vkb-456"
        assert session.message_count == 10
        assert session.last_active == 1234567890.0

    def test_mutable_fields(self):
        """Should allow updating mutable fields."""
        from valence.agents.matrix_bot import RoomSession

        session = RoomSession(
            room_id="!room123:example.com",
            room_name="Test Room",
        )

        session.claude_session_id = "new-claude-id"
        session.message_count = 5

        assert session.claude_session_id == "new-claude-id"
        assert session.message_count == 5


# ============================================================================
# SessionManager Tests
# ============================================================================

class TestSessionManager:
    """Tests for SessionManager class."""

    def test_init_default_plugin_dir(self):
        """Should use default plugin directory."""
        from valence.agents.matrix_bot import SessionManager

        manager = SessionManager()

        assert manager.plugin_dir == os.path.expanduser("~/.claude/plugins/valence")
        assert manager.room_sessions == {}

    def test_init_custom_plugin_dir(self):
        """Should use custom plugin directory."""
        from valence.agents.matrix_bot import SessionManager

        manager = SessionManager(plugin_dir="/custom/path")

        assert manager.plugin_dir == "/custom/path"

    def test_get_session_not_found(self):
        """Should return None for unknown room."""
        from valence.agents.matrix_bot import SessionManager

        manager = SessionManager()
        result = manager.get_session("!unknown:example.com")

        assert result is None

    def test_get_session_found(self):
        """Should return session for known room."""
        from valence.agents.matrix_bot import SessionManager, RoomSession

        manager = SessionManager()
        session = RoomSession(room_id="!room:example.com", room_name="Test")
        manager.room_sessions["!room:example.com"] = session

        result = manager.get_session("!room:example.com")

        assert result is session

    def test_create_session(self):
        """Should create and store new session."""
        from valence.agents.matrix_bot import SessionManager

        manager = SessionManager()

        session = manager.create_session("!room:example.com", "Test Room")

        assert session.room_id == "!room:example.com"
        assert session.room_name == "Test Room"
        assert manager.room_sessions["!room:example.com"] is session

    def test_update_claude_session(self):
        """Should update Claude session ID."""
        from valence.agents.matrix_bot import SessionManager

        manager = SessionManager()
        manager.create_session("!room:example.com", "Test")

        manager.update_claude_session("!room:example.com", "claude-session-123")

        assert manager.room_sessions["!room:example.com"].claude_session_id == "claude-session-123"

    def test_update_claude_session_unknown_room(self):
        """Should not error for unknown room."""
        from valence.agents.matrix_bot import SessionManager

        manager = SessionManager()

        # Should not raise
        manager.update_claude_session("!unknown:example.com", "claude-123")

    def test_update_vkb_session(self):
        """Should update VKB session ID."""
        from valence.agents.matrix_bot import SessionManager

        manager = SessionManager()
        manager.create_session("!room:example.com", "Test")

        manager.update_vkb_session("!room:example.com", "vkb-session-456")

        assert manager.room_sessions["!room:example.com"].vkb_session_id == "vkb-session-456"

    def test_update_vkb_session_unknown_room(self):
        """Should not error for unknown room."""
        from valence.agents.matrix_bot import SessionManager

        manager = SessionManager()

        # Should not raise
        manager.update_vkb_session("!unknown:example.com", "vkb-456")

    @pytest.fixture
    def mock_get_cursor(self):
        """Mock get_cursor context manager."""
        from contextlib import contextmanager

        mock_cursor = MagicMock()

        @contextmanager
        def fake_get_cursor():
            yield mock_cursor

        with patch("valence.agents.matrix_bot.get_cursor", fake_get_cursor):
            yield mock_cursor

    def test_load_from_db_success(self, mock_get_cursor):
        """Should load sessions from database."""
        from valence.agents.matrix_bot import SessionManager

        mock_get_cursor.fetchall.return_value = [
            {
                "external_room_id": "!room1:example.com",
                "claude_session_id": "claude-1",
                "vkb_session_id": "00000000-0000-0000-0000-000000000001",
            },
            {
                "external_room_id": "!room2:example.com",
                "claude_session_id": None,
                "vkb_session_id": "00000000-0000-0000-0000-000000000002",
            },
        ]

        manager = SessionManager()
        manager.load_from_db()

        assert len(manager.room_sessions) == 2
        assert manager.room_sessions["!room1:example.com"].claude_session_id == "claude-1"
        assert manager.room_sessions["!room2:example.com"].claude_session_id is None

    def test_load_from_db_updates_existing(self, mock_get_cursor):
        """Should update existing sessions from database."""
        from valence.agents.matrix_bot import SessionManager

        mock_get_cursor.fetchall.return_value = [
            {
                "external_room_id": "!room1:example.com",
                "claude_session_id": "updated-claude-id",
                "vkb_session_id": "00000000-0000-0000-0000-000000000001",
            },
        ]

        manager = SessionManager()
        manager.create_session("!room1:example.com", "Test Room")

        manager.load_from_db()

        assert manager.room_sessions["!room1:example.com"].claude_session_id == "updated-claude-id"

    def test_load_from_db_handles_error(self):
        """Should handle database errors gracefully."""
        from valence.agents.matrix_bot import SessionManager
        from valence.core.exceptions import DatabaseException

        with patch("valence.agents.matrix_bot.get_cursor") as mock_get_cursor:
            mock_get_cursor.side_effect = DatabaseException("Connection failed")

            manager = SessionManager()
            # Should not raise
            manager.load_from_db()

            assert len(manager.room_sessions) == 0


# ============================================================================
# ValenceBot Tests
# ============================================================================

class TestValenceBot:
    """Tests for ValenceBot class."""

    @pytest.fixture
    def mock_nio(self):
        """Mock nio library components."""
        with patch("valence.agents.matrix_bot.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.user_id = "@bot:example.com"
            mock_client.add_event_callback = MagicMock()
            mock_client_cls.return_value = mock_client
            yield mock_client

    def test_init(self, mock_nio):
        """Should initialize with correct settings."""
        from valence.agents.matrix_bot import ValenceBot

        bot = ValenceBot(
            homeserver="https://matrix.example.com",
            user_id="@bot:example.com",
            password="secret",
        )

        assert bot.homeserver == "https://matrix.example.com"
        assert bot.user_id == "@bot:example.com"
        assert bot.password == "secret"
        assert bot.device_name == "ValenceBot"
        assert bot._initial_sync_done is False

    def test_init_custom_device_name(self, mock_nio):
        """Should use custom device name."""
        from valence.agents.matrix_bot import ValenceBot

        bot = ValenceBot(
            homeserver="https://matrix.example.com",
            user_id="@bot:example.com",
            password="secret",
            device_name="CustomBot",
        )

        assert bot.device_name == "CustomBot"

    @pytest.mark.asyncio
    async def test_login_success(self, mock_nio):
        """Should login successfully."""
        from valence.agents.matrix_bot import ValenceBot
        from nio import LoginResponse

        mock_nio.login.return_value = LoginResponse(
            user_id="@bot:example.com",
            device_id="device123",
            access_token="token",
        )

        bot = ValenceBot(
            homeserver="https://matrix.example.com",
            user_id="@bot:example.com",
            password="secret",
        )

        with patch.object(bot.session_manager, "load_from_db"):
            result = await bot.login()

        assert result is True
        mock_nio.login.assert_called_once()

    @pytest.mark.asyncio
    async def test_login_failure(self, mock_nio):
        """Should handle login failure."""
        from valence.agents.matrix_bot import ValenceBot

        mock_nio.login.return_value = MagicMock(spec=[])  # Not a LoginResponse

        bot = ValenceBot(
            homeserver="https://matrix.example.com",
            user_id="@bot:example.com",
            password="wrong",
        )

        result = await bot.login()

        assert result is False


# ============================================================================
# ValenceBot._should_respond Tests
# ============================================================================

class TestShouldRespond:
    """Tests for ValenceBot._should_respond method."""

    @pytest.fixture
    def bot(self):
        """Create bot for testing."""
        with patch("valence.agents.matrix_bot.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.user_id = "@valence-bot:example.com"
            mock_client_cls.return_value = mock_client

            from valence.agents.matrix_bot import ValenceBot

            return ValenceBot(
                homeserver="https://matrix.example.com",
                user_id="@valence-bot:example.com",
                password="secret",
            )

    @pytest.fixture
    def mock_room(self):
        """Create mock room."""
        room = MagicMock()
        room.member_count = 5
        return room

    def test_responds_to_mention(self, bot, mock_room):
        """Should respond when mentioned by name."""
        result = bot._should_respond(mock_room, "Hey valence-bot, what's up?")
        assert result is True

    def test_responds_to_valence_mention(self, bot, mock_room):
        """Should respond to @valence mention."""
        result = bot._should_respond(mock_room, "Hey @valence, help me")
        assert result is True

    def test_responds_in_dm(self, bot, mock_room):
        """Should respond in DMs (2 member rooms)."""
        mock_room.member_count = 2
        result = bot._should_respond(mock_room, "Hello there")
        assert result is True

    def test_responds_to_command(self, bot, mock_room):
        """Should respond to ! commands."""
        result = bot._should_respond(mock_room, "!help")
        assert result is True

    def test_ignores_unmentioned_messages(self, bot, mock_room):
        """Should not respond to unmentioned messages in group rooms."""
        result = bot._should_respond(mock_room, "Hello everyone")
        assert result is False

    def test_mention_case_insensitive(self, bot, mock_room):
        """Should handle case-insensitive mentions."""
        result = bot._should_respond(mock_room, "VALENCE-BOT help please")
        assert result is True


# ============================================================================
# ValenceBot._generate_response Tests
# ============================================================================

class TestGenerateResponse:
    """Tests for ValenceBot._generate_response method."""

    @pytest.fixture
    def bot(self):
        """Create bot for testing."""
        with patch("valence.agents.matrix_bot.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.user_id = "@bot:example.com"
            mock_client_cls.return_value = mock_client

            from valence.agents.matrix_bot import ValenceBot

            return ValenceBot(
                homeserver="https://matrix.example.com",
                user_id="@bot:example.com",
                password="secret",
            )

    @pytest.fixture
    def session(self):
        """Create test session."""
        from valence.agents.matrix_bot import RoomSession

        return RoomSession(
            room_id="!room:example.com",
            room_name="Test Room",
        )

    @pytest.mark.asyncio
    async def test_success_json_response(self, bot, session):
        """Should parse JSON response from Claude."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "session_id": "new-session-123",
            "result": "Hello! How can I help?",
        })

        with patch("subprocess.run", return_value=mock_result):
            with patch("os.path.exists", return_value=False):
                response = await bot._generate_response(session, "@user:example.com", "Hi")

        assert response == "Hello! How can I help?"
        assert session.claude_session_id == "new-session-123"

    @pytest.mark.asyncio
    async def test_success_with_messages_array(self, bot, session):
        """Should extract response from messages array."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "session_id": "session-456",
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello there!"},
            ],
        })

        with patch("subprocess.run", return_value=mock_result):
            with patch("os.path.exists", return_value=False):
                response = await bot._generate_response(session, "@user:example.com", "Hi")

        assert response == "Hello there!"

    @pytest.mark.asyncio
    async def test_success_non_json(self, bot, session):
        """Should handle non-JSON output."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Plain text response"

        with patch("subprocess.run", return_value=mock_result):
            with patch("os.path.exists", return_value=False):
                response = await bot._generate_response(session, "@user:example.com", "Hi")

        assert response == "Plain text response"

    @pytest.mark.asyncio
    async def test_truncates_long_response(self, bot, session):
        """Should truncate very long responses."""
        long_text = "x" * 3000
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"result": long_text})

        with patch("subprocess.run", return_value=mock_result):
            with patch("os.path.exists", return_value=False):
                response = await bot._generate_response(session, "@user:example.com", "Hi")

        assert len(response) == 2000
        assert response.endswith("...")

    @pytest.mark.asyncio
    async def test_uses_resume_flag(self, bot, session):
        """Should use --resume flag for existing sessions."""
        session.claude_session_id = "existing-session-id"

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"result": "Response"})

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            with patch("os.path.exists", return_value=False):
                await bot._generate_response(session, "@user:example.com", "Hi")

        cmd = mock_run.call_args[0][0]
        assert "--resume" in cmd
        assert "existing-session-id" in cmd

    @pytest.mark.asyncio
    async def test_cli_error(self, bot, session):
        """Should return None on CLI error."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Error message"

        with patch("subprocess.run", return_value=mock_result):
            with patch("os.path.exists", return_value=False):
                response = await bot._generate_response(session, "@user:example.com", "Hi")

        assert response is None

    @pytest.mark.asyncio
    async def test_timeout(self, bot, session):
        """Should handle timeout."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 600)):
            with patch("os.path.exists", return_value=False):
                response = await bot._generate_response(session, "@user:example.com", "Hi")

        assert response is not None
        assert "took too long" in response

    @pytest.mark.asyncio
    async def test_cli_not_found(self, bot, session):
        """Should handle missing CLI."""
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            with patch("os.path.exists", return_value=False):
                response = await bot._generate_response(session, "@user:example.com", "Hi")

        assert response is None

    @pytest.mark.asyncio
    async def test_subprocess_error(self, bot, session):
        """Should handle subprocess errors."""
        with patch("subprocess.run", side_effect=subprocess.SubprocessError("Error")):
            with patch("os.path.exists", return_value=False):
                response = await bot._generate_response(session, "@user:example.com", "Hi")

        assert response is None

    @pytest.mark.asyncio
    async def test_adds_plugin_dir(self, bot, session):
        """Should add plugin dir if it exists."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"result": "Response"})

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            with patch("os.path.exists", return_value=True):
                await bot._generate_response(session, "@user:example.com", "Hi")

        cmd = mock_run.call_args[0][0]
        assert "--plugin-dir" in cmd


# ============================================================================
# ValenceBot Event Handling Tests
# ============================================================================

class TestOnMessage:
    """Tests for ValenceBot.on_message method."""

    @pytest.fixture
    def bot(self):
        """Create bot for testing."""
        with patch("valence.agents.matrix_bot.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.user_id = "@bot:example.com"
            mock_client.room_send = AsyncMock()
            mock_client_cls.return_value = mock_client

            from valence.agents.matrix_bot import ValenceBot

            bot = ValenceBot(
                homeserver="https://matrix.example.com",
                user_id="@bot:example.com",
                password="secret",
            )
            bot._initial_sync_done = True
            return bot

    @pytest.fixture
    def mock_room(self):
        """Create mock room."""
        room = MagicMock()
        room.room_id = "!room:example.com"
        room.display_name = "Test Room"
        room.member_count = 2
        return room

    @pytest.fixture
    def mock_event(self):
        """Create mock message event."""
        event = MagicMock()
        event.sender = "@user:example.com"
        event.body = "Hello bot"
        return event

    @pytest.mark.asyncio
    async def test_ignores_own_messages(self, bot, mock_room, mock_event):
        """Should ignore messages from self."""
        mock_event.sender = "@bot:example.com"

        with patch.object(bot, "_should_respond") as mock_should:
            await bot.on_message(mock_room, mock_event)
            mock_should.assert_not_called()

    @pytest.mark.asyncio
    async def test_ignores_before_initial_sync(self, bot, mock_room, mock_event):
        """Should ignore messages before initial sync."""
        bot._initial_sync_done = False

        with patch.object(bot, "_should_respond") as mock_should:
            await bot.on_message(mock_room, mock_event)
            mock_should.assert_not_called()

    @pytest.mark.asyncio
    async def test_ignores_non_matching_messages(self, bot, mock_room, mock_event):
        """Should not respond to non-matching messages."""
        mock_room.member_count = 10
        # Change the message body to something that won't match the bot name
        mock_event.body = "Hello everyone"

        with patch.object(bot, "_should_respond", return_value=False):
            with patch.object(bot, "_generate_response") as mock_gen:
                await bot.on_message(mock_room, mock_event)
                mock_gen.assert_not_called()

    @pytest.mark.asyncio
    async def test_responds_to_dm(self, bot, mock_room, mock_event):
        """Should respond to DMs."""
        mock_room.member_count = 2

        with patch.object(bot, "_get_or_create_session") as mock_session:
            from valence.agents.matrix_bot import RoomSession
            mock_session.return_value = RoomSession(
                room_id="!room:example.com",
                room_name="Test",
                vkb_session_id="vkb-123",
            )

            with patch.object(bot, "_record_exchange", new_callable=AsyncMock):
                with patch.object(bot, "_generate_response", new_callable=AsyncMock) as mock_gen:
                    mock_gen.return_value = "Hello!"
                    await bot.on_message(mock_room, mock_event)

                    mock_gen.assert_called_once()


# ============================================================================
# main() Tests
# ============================================================================

# ============================================================================
# ValenceBot._get_or_create_session Tests (Session Resumption)
# ============================================================================

class TestGetOrCreateSession:
    """Tests for ValenceBot._get_or_create_session method.

    This tests the session resumption fix that ensures the bot resumes
    existing active sessions on restart instead of creating duplicates.
    """

    @pytest.fixture
    def bot(self):
        """Create bot for testing."""
        with patch("valence.agents.matrix_bot.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.user_id = "@bot:example.com"
            mock_client_cls.return_value = mock_client

            from valence.agents.matrix_bot import ValenceBot

            return ValenceBot(
                homeserver="https://matrix.example.com",
                user_id="@bot:example.com",
                password="secret",
            )

    @pytest.fixture
    def mock_room(self):
        """Create mock Matrix room."""
        room = MagicMock()
        room.room_id = "!room123:example.com"
        room.display_name = "Test Room"
        return room

    @pytest.fixture
    def mock_get_cursor_for_bot(self):
        """Mock get_cursor specifically for matrix_bot module."""
        from contextlib import contextmanager

        mock_cursor = MagicMock()

        @contextmanager
        def fake_get_cursor():
            yield mock_cursor

        with patch("valence.agents.matrix_bot.get_cursor", fake_get_cursor):
            yield mock_cursor

    @pytest.mark.asyncio
    async def test_returns_existing_in_memory_session(self, bot, mock_room):
        """Should return existing session from in-memory cache."""
        from valence.agents.matrix_bot import RoomSession

        existing_session = RoomSession(
            room_id="!room123:example.com",
            room_name="Old Name",
            claude_session_id="claude-123",
            vkb_session_id="vkb-456",
        )
        bot.session_manager.room_sessions["!room123:example.com"] = existing_session

        result = await bot._get_or_create_session(mock_room)

        assert result is existing_session
        assert result.room_name == "Test Room"  # Updated to current name

    @pytest.mark.asyncio
    async def test_resumes_existing_db_session_on_restart(self, bot, mock_room, mock_get_cursor_for_bot):
        """Should resume existing active session from database on restart.

        This is the key test for the session resumption fix. When the bot restarts,
        if there's an active session in the database for this room, it should
        resume that session instead of creating a new one.
        """
        from uuid import uuid4

        existing_session_id = uuid4()

        # Simulate database returning an existing active session
        mock_get_cursor_for_bot.fetchone.return_value = {
            "id": existing_session_id,
            "claude_session_id": "claude-existing-session",
        }

        result = await bot._get_or_create_session(mock_room)

        # Verify it queried for existing session
        mock_get_cursor_for_bot.execute.assert_called()
        query = mock_get_cursor_for_bot.execute.call_args_list[0][0][0]
        assert "SELECT" in query
        assert "status = 'active'" in query
        assert "external_room_id" in query

        # Verify it resumed the existing session
        assert result.vkb_session_id == str(existing_session_id)
        assert result.claude_session_id == "claude-existing-session"
        assert result.room_id == "!room123:example.com"

    @pytest.mark.asyncio
    async def test_creates_new_session_when_none_exists(self, bot, mock_room, mock_get_cursor_for_bot):
        """Should create new session when no active session exists in database."""
        from uuid import uuid4

        new_session_id = uuid4()

        # First query returns None (no existing session)
        # Second query creates new session
        mock_get_cursor_for_bot.fetchone.side_effect = [
            None,  # No existing session
            {"id": new_session_id},  # New session created
        ]

        result = await bot._get_or_create_session(mock_room)

        # Verify it tried to find existing session first
        calls = mock_get_cursor_for_bot.execute.call_args_list
        assert len(calls) >= 2

        # First call should be SELECT
        assert "SELECT" in calls[0][0][0]

        # Second call should be INSERT
        assert "INSERT INTO vkb_sessions" in calls[1][0][0]

        # Verify new session was created
        assert result.vkb_session_id == str(new_session_id)

    @pytest.mark.asyncio
    async def test_handles_db_error_checking_existing_session(self, bot, mock_room):
        """Should handle database error when checking for existing session."""
        from valence.core.exceptions import DatabaseException
        from contextlib import contextmanager
        from uuid import uuid4

        new_session_id = uuid4()
        call_count = [0]

        @contextmanager
        def failing_then_succeeding_cursor():
            mock_cursor = MagicMock()
            call_count[0] += 1
            if call_count[0] == 1:
                # First call (checking for existing) raises error
                raise DatabaseException("Connection failed")
            else:
                # Subsequent calls succeed
                mock_cursor.fetchone.return_value = {"id": new_session_id}
                yield mock_cursor

        with patch("valence.agents.matrix_bot.get_cursor", failing_then_succeeding_cursor):
            result = await bot._get_or_create_session(mock_room)

        # Should still create a new session despite error
        assert result.vkb_session_id == str(new_session_id)

    @pytest.mark.asyncio
    async def test_resumes_session_with_null_claude_session_id(self, bot, mock_room, mock_get_cursor_for_bot):
        """Should handle resuming session that has no Claude session ID yet."""
        from uuid import uuid4

        existing_session_id = uuid4()

        mock_get_cursor_for_bot.fetchone.return_value = {
            "id": existing_session_id,
            "claude_session_id": None,  # No Claude session yet
        }

        result = await bot._get_or_create_session(mock_room)

        assert result.vkb_session_id == str(existing_session_id)
        assert result.claude_session_id is None

    @pytest.mark.asyncio
    async def test_uses_most_recent_active_session(self, bot, mock_room, mock_get_cursor_for_bot):
        """Should use ORDER BY started_at DESC to get most recent session."""
        mock_get_cursor_for_bot.fetchone.return_value = {
            "id": "most-recent-session-id",
            "claude_session_id": "claude-recent",
        }

        await bot._get_or_create_session(mock_room)

        query = mock_get_cursor_for_bot.execute.call_args_list[0][0][0]
        assert "ORDER BY started_at DESC" in query
        assert "LIMIT 1" in query


class TestSessionResumptionIntegration:
    """Integration tests for session resumption across bot restarts."""

    @pytest.fixture
    def mock_get_cursor_for_bot(self):
        """Mock get_cursor specifically for matrix_bot module."""
        from contextlib import contextmanager

        mock_cursor = MagicMock()

        @contextmanager
        def fake_get_cursor():
            yield mock_cursor

        with patch("valence.agents.matrix_bot.get_cursor", fake_get_cursor):
            yield mock_cursor

    def test_load_from_db_populates_sessions(self, mock_get_cursor_for_bot):
        """SessionManager.load_from_db should populate room_sessions."""
        from valence.agents.matrix_bot import SessionManager

        mock_get_cursor_for_bot.fetchall.return_value = [
            {
                "external_room_id": "!room1:example.com",
                "claude_session_id": "claude-1",
                "vkb_session_id": "vkb-1",
            },
            {
                "external_room_id": "!room2:example.com",
                "claude_session_id": "claude-2",
                "vkb_session_id": "vkb-2",
            },
        ]

        manager = SessionManager()
        manager.load_from_db()

        assert len(manager.room_sessions) == 2
        assert "!room1:example.com" in manager.room_sessions
        assert "!room2:example.com" in manager.room_sessions
        assert manager.room_sessions["!room1:example.com"].claude_session_id == "claude-1"

    @pytest.mark.asyncio
    async def test_full_restart_scenario(self, mock_get_cursor_for_bot):
        """Test complete restart scenario: bot stops, restarts, resumes session."""
        from uuid import uuid4

        # Session created in previous bot run
        existing_vkb_session = uuid4()
        existing_claude_session = "claude-session-before-restart"

        # First bot instance creates session
        with patch("valence.agents.matrix_bot.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.user_id = "@bot:example.com"
            mock_client_cls.return_value = mock_client

            from valence.agents.matrix_bot import ValenceBot

            bot1 = ValenceBot(
                homeserver="https://matrix.example.com",
                user_id="@bot:example.com",
                password="secret",
            )

            # Simulate first bot creating a session
            mock_get_cursor_for_bot.fetchone.return_value = None  # No existing session
            mock_get_cursor_for_bot.fetchone.side_effect = [
                None,  # No existing session
                {"id": existing_vkb_session},  # New session created
            ]

            mock_room = MagicMock()
            mock_room.room_id = "!persistent:example.com"
            mock_room.display_name = "Persistent Room"

            session1 = await bot1._get_or_create_session(mock_room)
            session1.claude_session_id = existing_claude_session

        # Reset mock for second bot
        mock_get_cursor_for_bot.reset_mock()
        mock_get_cursor_for_bot.fetchone.side_effect = None

        # Second bot instance (after restart) should resume session
        with patch("valence.agents.matrix_bot.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.user_id = "@bot:example.com"
            mock_client_cls.return_value = mock_client

            from valence.agents.matrix_bot import ValenceBot

            bot2 = ValenceBot(
                homeserver="https://matrix.example.com",
                user_id="@bot:example.com",
                password="secret",
            )

            # Simulate database returning the existing session
            mock_get_cursor_for_bot.fetchone.return_value = {
                "id": existing_vkb_session,
                "claude_session_id": existing_claude_session,
            }

            session2 = await bot2._get_or_create_session(mock_room)

            # Verify session was resumed, not created anew
            assert session2.vkb_session_id == str(existing_vkb_session)
            assert session2.claude_session_id == existing_claude_session


# ============================================================================
# main() Tests
# ============================================================================

class TestMain:
    """Tests for main entry point."""

    def test_missing_password(self, clean_env):
        """Should exit if MATRIX_PASSWORD not set."""
        from valence.agents.matrix_bot import main

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    def test_runs_bot(self, env_with_matrix_vars):
        """Should create and run bot."""
        with patch("valence.agents.matrix_bot.AsyncClient"):
            with patch("valence.agents.matrix_bot.ValenceBot") as mock_bot_cls:
                mock_bot = MagicMock()
                mock_bot_cls.return_value = mock_bot

                with patch("asyncio.run") as mock_run:
                    from valence.agents.matrix_bot import main
                    main()

                    mock_bot_cls.assert_called_once()
                    mock_run.assert_called_once()
