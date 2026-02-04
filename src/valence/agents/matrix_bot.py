"""Valence Matrix Bot - AI agent that lives in Matrix rooms.

Redesigned to use Claude Code session resumption for maintaining
conversation context across messages.

Key features:
- Session resumption via --resume flag
- Plugin integration for knowledge substrate
- Cross-session memory via substrate queries
- VKB conversation tracking

Requires the 'matrix' extra: pip install valence[matrix]
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any


# =============================================================================
# Input Sanitization (Security)
# =============================================================================

# Maximum lengths to prevent DoS via extremely long inputs
MAX_MESSAGE_LENGTH = 10000  # 10KB max message
MAX_SENDER_LENGTH = 256  # Matrix user IDs shouldn't be longer than this
MAX_SESSION_ID_LENGTH = 128  # Claude session IDs are typically short UUIDs

# Pattern for valid Matrix user IDs: @localpart:domain
MATRIX_USER_PATTERN = re.compile(r"^@[a-zA-Z0-9._=/+-]+:[a-zA-Z0-9.-]+$")

# Pattern for valid session IDs (alphanumeric, hyphens, underscores)
SESSION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


def sanitize_for_subprocess(text: str, max_length: int, field_name: str) -> str:
    """Sanitize text before passing to subprocess arguments.
    
    This function removes potentially dangerous characters and enforces
    length limits to prevent command injection and DoS attacks.
    
    Args:
        text: The text to sanitize
        max_length: Maximum allowed length
        field_name: Name of the field (for logging)
    
    Returns:
        Sanitized text safe for subprocess arguments
    
    Security measures:
        - Removes null bytes (can truncate arguments)
        - Removes control characters except newlines/tabs
        - Enforces length limits
        - Strips leading/trailing whitespace
    """
    if not text:
        return ""
    
    # Remove null bytes - these can cause argument truncation
    text = text.replace("\x00", "")
    
    # Remove control characters except common whitespace (newline, tab, carriage return)
    # Control chars are 0x00-0x1F and 0x7F, excluding \t (0x09), \n (0x0A), \r (0x0D)
    text = "".join(
        char for char in text
        if char >= " " or char in "\t\n\r"
    )
    
    # Enforce length limit
    if len(text) > max_length:
        logger.warning(
            f"Truncating {field_name} from {len(text)} to {max_length} characters"
        )
        text = text[:max_length]
    
    return text.strip()


def validate_sender(sender: str) -> str:
    """Validate and sanitize a Matrix sender ID.
    
    Args:
        sender: The Matrix user ID (e.g., @user:matrix.org)
    
    Returns:
        Validated sender ID
    
    Raises:
        ValueError: If sender format is invalid
    """
    sender = sanitize_for_subprocess(sender, MAX_SENDER_LENGTH, "sender")
    
    if not sender:
        raise ValueError("Empty sender ID")
    
    # Validate Matrix user ID format
    if not MATRIX_USER_PATTERN.match(sender):
        # Log the issue but use a safe fallback
        logger.warning(f"Invalid Matrix user ID format, using sanitized version")
        # Replace any remaining problematic characters
        sender = re.sub(r"[^\w@:._=/-]", "_", sender)
    
    return sender


def validate_session_id(session_id: str | None) -> str | None:
    """Validate a Claude session ID before using in subprocess.
    
    Args:
        session_id: The session ID to validate, or None
    
    Returns:
        Validated session ID or None
    
    Raises:
        ValueError: If session ID format is invalid
    """
    if session_id is None:
        return None
    
    session_id = sanitize_for_subprocess(session_id, MAX_SESSION_ID_LENGTH, "session_id")
    
    if not session_id:
        return None
    
    # Session IDs should be alphanumeric with hyphens/underscores only
    if not SESSION_ID_PATTERN.match(session_id):
        logger.warning(f"Invalid session ID format: {session_id[:20]}...")
        raise ValueError(f"Invalid session ID format")
    
    return session_id


def sanitize_message(message: str) -> str:
    """Sanitize a user message before passing to subprocess.
    
    Args:
        message: The user's message text
    
    Returns:
        Sanitized message safe for subprocess arguments
    """
    return sanitize_for_subprocess(message, MAX_MESSAGE_LENGTH, "message")

try:
    from nio import (
        AsyncClient,
        AsyncClientConfig,
        InviteMemberEvent,
        JoinError,
        LoginResponse,
        MatrixRoom,
        RoomMessageText,
        SyncResponse,
    )
    NIO_AVAILABLE = True
except ImportError:
    NIO_AVAILABLE = False
    # Provide stub types for type checking when nio is not installed
    AsyncClient = None  # type: ignore[assignment,misc]
    AsyncClientConfig = None  # type: ignore[assignment,misc]
    InviteMemberEvent = None  # type: ignore[assignment,misc]
    JoinError = None  # type: ignore[assignment,misc]
    LoginResponse = None  # type: ignore[assignment,misc]
    MatrixRoom = None  # type: ignore[assignment,misc]
    RoomMessageText = None  # type: ignore[assignment,misc]
    SyncResponse = None  # type: ignore[assignment,misc]

from ..core.db import get_cursor
from ..core.exceptions import DatabaseException

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class RoomSession:
    """Tracks session state for a Matrix room."""
    room_id: str
    room_name: str
    claude_session_id: str | None = None
    vkb_session_id: str | None = None
    message_count: int = 0
    last_active: float = 0


class SessionManager:
    """Manages Claude Code sessions per room."""

    def __init__(self, plugin_dir: str | None = None):
        self.plugin_dir = plugin_dir or os.path.expanduser("~/.claude/plugins/valence")
        self.room_sessions: dict[str, RoomSession] = {}

    def get_session(self, room_id: str) -> RoomSession | None:
        """Get existing session for a room."""
        return self.room_sessions.get(room_id)

    def create_session(self, room_id: str, room_name: str) -> RoomSession:
        """Create a new session for a room."""
        session = RoomSession(
            room_id=room_id,
            room_name=room_name,
        )
        self.room_sessions[room_id] = session
        return session

    def update_claude_session(self, room_id: str, claude_session_id: str) -> None:
        """Update the Claude session ID for a room."""
        if room_id in self.room_sessions:
            self.room_sessions[room_id].claude_session_id = claude_session_id

    def update_vkb_session(self, room_id: str, vkb_session_id: str) -> None:
        """Update the VKB session ID for a room."""
        if room_id in self.room_sessions:
            self.room_sessions[room_id].vkb_session_id = vkb_session_id

    def load_from_db(self) -> None:
        """Load active sessions from database."""
        try:
            with get_cursor() as cur:
                cur.execute("""
                    SELECT external_room_id, claude_session_id, id as vkb_session_id
                    FROM vkb_sessions
                    WHERE platform = 'matrix'
                    AND status = 'active'
                    AND external_room_id IS NOT NULL
                """)
                rows = cur.fetchall()

                for row in rows:
                    room_id = row["external_room_id"]
                    if room_id not in self.room_sessions:
                        self.room_sessions[room_id] = RoomSession(
                            room_id=room_id,
                            room_name=room_id,  # Will be updated when we see the room
                            claude_session_id=row.get("claude_session_id"),
                            vkb_session_id=str(row["vkb_session_id"]),
                        )
                    else:
                        self.room_sessions[room_id].claude_session_id = row.get("claude_session_id")
                        self.room_sessions[room_id].vkb_session_id = str(row["vkb_session_id"])

                logger.info(f"Loaded {len(rows)} active sessions from database")
        except DatabaseException as e:
            logger.warning(f"Could not load sessions from database: {e}")


def _check_nio_available() -> None:
    """Check that matrix-nio is installed, raise helpful error if not."""
    if not NIO_AVAILABLE:
        raise ImportError(
            "matrix-nio is required for the Matrix bot. "
            "Install it with: pip install valence[matrix]"
        )


class ValenceBot:
    """Matrix bot that uses Claude Code with session resumption."""

    def __init__(
        self,
        homeserver: str,
        user_id: str,
        password: str,
        device_name: str = "ValenceBot",
        plugin_dir: str | None = None,
    ):
        _check_nio_available()

        self.homeserver = homeserver
        self.user_id = user_id
        self.password = password
        self.device_name = device_name

        # Session management
        self.session_manager = SessionManager(plugin_dir)

        # Matrix client config
        config = AsyncClientConfig(
            max_limit_exceeded=0,
            max_timeouts=0,
            store_sync_tokens=True,
        )
        self.client = AsyncClient(
            homeserver,
            user_id,
            config=config,
        )
        self.client.add_event_callback(self.on_message, RoomMessageText)
        self.client.add_event_callback(self.on_invite, InviteMemberEvent)

        # Track when we're ready to process messages
        self._initial_sync_done = False

    async def login(self) -> bool:
        """Login to the Matrix homeserver."""
        logger.info(f"Logging in as {self.user_id} to {self.homeserver}")
        response = await self.client.login(self.password, device_name=self.device_name)

        if isinstance(response, LoginResponse):
            logger.info(f"Logged in successfully. Device ID: {response.device_id}")

            # Load existing sessions from database
            self.session_manager.load_from_db()

            return True
        else:
            logger.error(f"Login failed: {response}")
            return False

    async def on_invite(self, room: MatrixRoom, event: InviteMemberEvent) -> None:
        """Handle room invites by auto-joining."""
        # Only respond to invites for us
        if event.state_key != self.client.user_id:
            return

        logger.info(f"Received invite to room {room.room_id} from {event.sender}")

        # Auto-join the room
        result = await self.client.join(room.room_id)

        if isinstance(result, JoinError):
            logger.error(f"Failed to join room {room.room_id}: {result.message}")
        else:
            logger.info(f"Successfully joined room {room.room_id}")

    async def on_message(self, room: MatrixRoom, event: RoomMessageText) -> None:
        """Handle incoming messages."""
        # Ignore our own messages
        if event.sender == self.client.user_id:
            return

        # Ignore messages from before we started
        if not self._initial_sync_done:
            return

        sender = event.sender
        body = event.body
        room_id = room.room_id

        logger.info(f"[{room.display_name}] {sender}: {body[:100]}...")

        # Check if we should respond
        if not self._should_respond(room, body):
            return

        # Get or create session for this room
        session = await self._get_or_create_session(room)

        # Record user message to VKB
        await self._record_exchange(session, "user", f"{sender}: {body}")

        # Generate response using Claude with session resumption
        response_text = await self._generate_response(session, sender, body)

        if response_text:
            # Send the response
            await self.client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content={
                    "msgtype": "m.text",
                    "body": response_text,
                },
            )

            # Record assistant message to VKB
            await self._record_exchange(session, "assistant", response_text)

            session.message_count += 1

    def _should_respond(self, room: MatrixRoom, body: str) -> bool:
        """Determine if we should respond to this message."""
        # Always respond if mentioned
        bot_name = self.client.user_id.split(":")[0].lstrip("@")
        if bot_name.lower() in body.lower():
            return True

        # Respond to @valence mentions
        if "@valence" in body.lower():
            return True

        # Respond in DMs (rooms with only 2 members)
        if room.member_count == 2:
            return True

        # Respond if message starts with ! (command prefix)
        if body.startswith("!"):
            return True

        return False

    async def _get_or_create_session(self, room: MatrixRoom) -> RoomSession:
        """Get existing session or create new one for a room.

        On restart, attempts to resume existing active sessions from the database
        rather than creating orphaned duplicates.
        """
        session = self.session_manager.get_session(room.room_id)

        if session:
            session.room_name = room.display_name
            return session

        # Check database for existing active session (handles restart case)
        try:
            with get_cursor() as cur:
                cur.execute("""
                    SELECT id, claude_session_id
                    FROM vkb_sessions
                    WHERE platform = 'matrix'
                    AND external_room_id = %s
                    AND status = 'active'
                    ORDER BY started_at DESC
                    LIMIT 1
                """, (room.room_id,))
                row = cur.fetchone()

                if row:
                    # Resume existing session
                    session = self.session_manager.create_session(room.room_id, room.display_name)
                    session.vkb_session_id = str(row["id"])
                    session.claude_session_id = row.get("claude_session_id")
                    logger.info(f"Resumed existing VKB session {session.vkb_session_id} for room {room.display_name}")
                    return session
        except DatabaseException as e:
            logger.warning(f"Could not check for existing session: {e}")

        # Create new session
        session = self.session_manager.create_session(room.room_id, room.display_name)

        # Create VKB session in database
        try:
            with get_cursor() as cur:
                cur.execute("""
                    INSERT INTO vkb_sessions (platform, project_context, external_room_id, status)
                    VALUES ('matrix', %s, %s, 'active')
                    RETURNING id
                """, (f"matrix:{room.display_name}", room.room_id))
                row = cur.fetchone()
                session.vkb_session_id = str(row["id"])
                logger.info(f"Created VKB session {session.vkb_session_id} for room {room.display_name}")
        except DatabaseException as e:
            logger.warning(f"Could not create VKB session: {e}")

        return session

    async def _record_exchange(
        self,
        session: RoomSession,
        role: str,
        content: str,
    ) -> None:
        """Record an exchange to VKB."""
        if not session.vkb_session_id:
            return

        try:
            with get_cursor() as cur:
                cur.execute("""
                    INSERT INTO vkb_exchanges (session_id, sequence, role, content)
                    SELECT %s,
                           COALESCE(MAX(sequence), 0) + 1,
                           %s,
                           %s
                    FROM vkb_exchanges WHERE session_id = %s
                """, (session.vkb_session_id, role, content, session.vkb_session_id))
        except DatabaseException as e:
            logger.warning(f"Could not record exchange: {e}")

    async def _generate_response(
        self,
        session: RoomSession,
        sender: str,
        message: str,
    ) -> str | None:
        """Generate a response using Claude CLI with session resumption.
        
        Security: All user input (sender, message) and stored data (session_id)
        is sanitized before being passed to subprocess to prevent command injection.
        """
        try:
            # === SECURITY: Sanitize all inputs before subprocess ===
            try:
                safe_sender = validate_sender(sender)
            except ValueError as e:
                logger.warning(f"Invalid sender, using fallback: {e}")
                safe_sender = "unknown_user"
            
            safe_message = sanitize_message(message)
            if not safe_message:
                logger.warning("Empty message after sanitization, skipping")
                return None
            
            try:
                safe_session_id = validate_session_id(session.claude_session_id)
            except ValueError as e:
                logger.warning(f"Invalid session ID, starting fresh session: {e}")
                safe_session_id = None
                session.claude_session_id = None  # Clear invalid session
            
            # Build the prompt with sanitized inputs
            prompt = f"User {safe_sender} says: {safe_message}"

            # Build command - using list arguments (no shell=True) is secure
            # as long as inputs are sanitized
            cmd = [
                "claude",
                "-p", prompt,
                "--output-format", "json",
                "--permission-mode", "bypassPermissions",
            ]

            # Add plugin directory if available (this is from config, not user input)
            plugin_dir = self.session_manager.plugin_dir
            if os.path.exists(plugin_dir):
                cmd.extend(["--plugin-dir", plugin_dir])

            # Add resume flag if we have a validated previous session
            if safe_session_id:
                cmd.extend(["--resume", safe_session_id])

            logger.debug(f"Running: {' '.join(cmd)}")

            # Run Claude
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes
                env={**os.environ, "CLAUDE_NO_TELEMETRY": "1"},
                cwd=os.path.expanduser("~"),
            )

            if result.returncode == 0:
                # Parse JSON output to get session ID and response
                try:
                    output = json.loads(result.stdout)

                    # Extract session ID for future resumption
                    new_session_id = output.get("session_id")
                    if new_session_id:
                        session.claude_session_id = new_session_id
                        self.session_manager.update_claude_session(
                            session.room_id,
                            new_session_id
                        )

                        # Update database with Claude session ID
                        if session.vkb_session_id:
                            try:
                                with get_cursor() as cur:
                                    cur.execute("""
                                        UPDATE vkb_sessions
                                        SET claude_session_id = %s
                                        WHERE id = %s
                                    """, (new_session_id, session.vkb_session_id))
                            except DatabaseException as e:
                                logger.warning(f"Could not update Claude session ID in DB: {e}")

                    # Extract the response text
                    response = output.get("result", "")
                    if not response:
                        # Try to get from messages
                        messages = output.get("messages", [])
                        for msg in reversed(messages):
                            if msg.get("role") == "assistant":
                                response = msg.get("content", "")
                                break

                    # Truncate very long responses
                    if len(response) > 2000:
                        response = response[:1997] + "..."

                    return response

                except json.JSONDecodeError:
                    # If JSON parsing fails, return raw output
                    response = result.stdout.strip()
                    if len(response) > 2000:
                        response = response[:1997] + "..."
                    return response

            else:
                logger.error(f"Claude CLI error: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            logger.error("Claude CLI timed out")
            return "I'm sorry, I took too long to think about that. Could you try again?"
        except FileNotFoundError:
            logger.error("Claude CLI not found")
            return None
        except subprocess.SubprocessError as e:
            logger.error(f"Subprocess error: {e}")
            return None

    async def run(self) -> None:
        """Run the bot."""
        if not await self.login():
            logger.error("Failed to login, exiting")
            return

        logger.info("Starting sync loop...")

        # Initial sync to get current state
        await self.client.sync(timeout=30000, full_state=True)
        self._initial_sync_done = True
        logger.info("Initial sync complete, now processing messages")

        # Continuous sync
        while True:
            try:
                sync_response = await self.client.sync(timeout=30000)
                if isinstance(sync_response, SyncResponse):
                    # Events are handled by callbacks
                    pass
            except asyncio.CancelledError:
                logger.info("Sync cancelled, shutting down")
                break
            except ConnectionError as e:
                logger.error(f"Connection error: {e}")
                await asyncio.sleep(5)
            except Exception as e:
                logger.exception(f"Unexpected sync error: {e}")
                await asyncio.sleep(5)

    async def close(self) -> None:
        """Clean up and close sessions."""
        # End all active VKB sessions
        for room_id, session in self.session_manager.room_sessions.items():
            if session.vkb_session_id:
                try:
                    with get_cursor() as cur:
                        cur.execute("""
                            UPDATE vkb_sessions
                            SET status = 'completed',
                                ended_at = NOW(),
                                summary = 'Bot shutdown'
                            WHERE id = %s
                        """, (session.vkb_session_id,))
                except DatabaseException as e:
                    logger.warning(f"Error ending session {session.vkb_session_id}: {e}")

        await self.client.close()


def main() -> None:
    """Entry point for the Matrix bot."""
    if not NIO_AVAILABLE:
        logger.error(
            "matrix-nio is required for the Matrix bot. "
            "Install it with: pip install valence[matrix]"
        )
        sys.exit(1)

    # Get configuration from environment
    homeserver = os.environ.get("MATRIX_HOMESERVER", "https://pod.zonk1024.info")
    user_id = os.environ.get("MATRIX_USER", "@valence-bot:pod.zonk1024.info")
    password = os.environ.get("MATRIX_PASSWORD")
    plugin_dir = os.environ.get("VALENCE_PLUGIN_DIR")

    if not password:
        logger.error("MATRIX_PASSWORD environment variable required")
        sys.exit(1)

    bot = ValenceBot(
        homeserver,
        user_id,
        password,
        plugin_dir=plugin_dir,
    )

    async def run_bot():
        try:
            await bot.run()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            await bot.close()

    asyncio.run(run_bot())


if __name__ == "__main__":
    main()
