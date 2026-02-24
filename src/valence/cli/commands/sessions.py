# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Sessions commands — manage conversation session buffers and ingestion.

Commands:
    valence sessions start <session-id>                Start/resume a session
    valence sessions append <session-id>               Append message to session buffer
    valence sessions flush <session-id>                Flush unflushed messages to source
    valence sessions finalize <session-id>             Flush + complete + compile session
    valence sessions search "query"                    Search conversation sources
    valence sessions list                              List sessions
    valence sessions show <session-id>                 Show session details
    valence sessions compile <session-id>              Compile session into article
    valence sessions flush-stale                       Flush all stale sessions
"""

from __future__ import annotations

import argparse
import sys

from ..http_client import ValenceAPIError, ValenceConnectionError, get_client
from ..output import output_error, output_result


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the sessions sub-command group."""
    sessions_parser = subparsers.add_parser("sessions", help="Manage conversation sessions")
    sessions_sub = sessions_parser.add_subparsers(dest="sessions_command", required=True)

    # --- start ---
    start_p = sessions_sub.add_parser("start", help="Start or resume a session")
    start_p.add_argument("session_id", help="Session identifier")
    start_p.add_argument(
        "--platform",
        required=True,
        choices=["openclaw", "claude-code"],
        help="Platform name",
    )
    start_p.add_argument("--channel", help="Channel (e.g., discord, telegram, cli)")
    start_p.add_argument("--parent-session-id", help="Parent session ID for subagents")
    start_p.set_defaults(func=cmd_sessions_start)

    # --- append ---
    append_p = sessions_sub.add_parser("append", help="Append message to session buffer")
    append_p.add_argument("session_id", help="Session identifier")
    append_p.add_argument(
        "--role",
        required=True,
        choices=["user", "assistant", "system", "tool"],
        help="Message role",
    )
    append_p.add_argument("--speaker", required=True, help="Speaker name")
    append_p.add_argument("--content", help="Message content (or read from stdin)")
    append_p.set_defaults(func=cmd_sessions_append)

    # --- flush ---
    flush_p = sessions_sub.add_parser("flush", help="Flush unflushed messages to source")
    flush_p.add_argument("session_id", help="Session identifier")
    flush_p.add_argument("--compile", action="store_true", help="Trigger compilation after flush")
    flush_p.add_argument("--no-compile", action="store_false", dest="compile", help="Skip compilation")
    flush_p.set_defaults(func=cmd_sessions_flush, compile=True)

    # --- finalize ---
    finalize_p = sessions_sub.add_parser("finalize", help="Flush + complete + compile session")
    finalize_p.add_argument("session_id", help="Session identifier")
    finalize_p.set_defaults(func=cmd_sessions_finalize)

    # --- search ---
    search_p = sessions_sub.add_parser("search", help="Search conversation sources")
    search_p.add_argument("query", help="Search query")
    search_p.add_argument("--limit", "-n", type=int, default=10, help="Max results (default 10)")
    search_p.set_defaults(func=cmd_sessions_search)

    # --- list ---
    list_p = sessions_sub.add_parser("list", help="List sessions")
    list_p.add_argument("--active", action="store_const", const="active", dest="status", help="Filter to active sessions")
    list_p.add_argument("--stale", action="store_const", const="stale", dest="status", help="Filter to stale sessions")
    list_p.add_argument("--completed", action="store_const", const="completed", dest="status", help="Filter to completed sessions")
    list_p.add_argument("--since", help="Filter to sessions active since ISO timestamp (e.g., 2026-02-24)")
    list_p.add_argument("--limit", "-n", type=int, default=100, help="Max results (default 100)")
    list_p.set_defaults(func=cmd_sessions_list)

    # --- show ---
    show_p = sessions_sub.add_parser("show", help="Show session details")
    show_p.add_argument("session_id", help="Session identifier")
    show_p.add_argument("--messages", action="store_true", dest="show_messages", help="Include message transcript")
    show_p.add_argument("--no-messages", action="store_false", dest="show_messages", help="Exclude messages")
    show_p.set_defaults(func=cmd_sessions_show, show_messages=False)

    # --- compile ---
    compile_p = sessions_sub.add_parser("compile", help="Compile session sources into article")
    compile_p.add_argument("session_id", help="Session identifier")
    compile_p.set_defaults(func=cmd_sessions_compile)

    # --- flush-stale ---
    flush_stale_p = sessions_sub.add_parser("flush-stale", help="Flush all stale sessions")
    flush_stale_p.add_argument("--stale-minutes", type=int, default=30, help="Stale threshold in minutes (default 30)")
    flush_stale_p.set_defaults(func=cmd_sessions_flush_stale)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def cmd_sessions_start(args: argparse.Namespace) -> int:
    """Start or resume a session."""
    client = get_client()
    body: dict = {
        "session_id": args.session_id,
        "platform": args.platform,
    }
    if getattr(args, "channel", None):
        body["channel"] = args.channel
    if getattr(args, "parent_session_id", None):
        body["parent_session_id"] = args.parent_session_id

    try:
        result = client.post("/sessions", body=body)
        output_result(result)
        return 0
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1


def cmd_sessions_append(args: argparse.Namespace) -> int:
    """Append message to session buffer."""
    client = get_client()

    # Read content from stdin if not provided as --content
    content = getattr(args, "content", None)
    if not content:
        content = sys.stdin.read()

    if not content:
        output_error("No content provided (use --content or pipe to stdin)")
        return 1

    body: dict = {
        "speaker": args.speaker,
        "role": args.role,
        "content": content,
    }

    try:
        result = client.post(f"/sessions/{args.session_id}/messages", body=body)
        output_result(result)
        return 0
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1


def cmd_sessions_flush(args: argparse.Namespace) -> int:
    """Flush unflushed messages to source."""
    client = get_client()
    params: dict = {
        "compile": str(args.compile).lower(),
    }

    try:
        result = client.post(f"/sessions/{args.session_id}/flush", params=params)
        output_result(result)
        return 0
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1


def cmd_sessions_finalize(args: argparse.Namespace) -> int:
    """Flush + complete + compile session."""
    client = get_client()

    try:
        result = client.post(f"/sessions/{args.session_id}/finalize")
        output_result(result)
        return 0
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1


def cmd_sessions_search(args: argparse.Namespace) -> int:
    """Search conversation sources."""
    client = get_client()
    params: dict = {
        "q": args.query,
        "limit": args.limit,
    }

    try:
        result = client.get("/sessions/search", params=params)
        output_result(result)
        return 0
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1


def cmd_sessions_list(args: argparse.Namespace) -> int:
    """List sessions."""
    client = get_client()
    params: dict = {
        "limit": args.limit,
    }
    if getattr(args, "status", None):
        params["status"] = args.status
    if getattr(args, "since", None):
        params["since"] = args.since

    try:
        result = client.get("/sessions", params=params)
        output_result(result)
        return 0
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1


def cmd_sessions_show(args: argparse.Namespace) -> int:
    """Show session details."""
    client = get_client()

    try:
        result = client.get(f"/sessions/{args.session_id}")

        # If messages requested, fetch them separately
        if args.show_messages:
            messages_result = client.get(f"/sessions/{args.session_id}/messages")
            result["messages"] = messages_result

        output_result(result)
        return 0
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1


def cmd_sessions_compile(args: argparse.Namespace) -> int:
    """Compile session sources into article."""
    client = get_client()

    try:
        result = client.post(f"/sessions/{args.session_id}/compile")
        output_result(result)
        return 0
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1


def cmd_sessions_flush_stale(args: argparse.Namespace) -> int:
    """Flush all stale sessions."""
    client = get_client()
    params: dict = {
        "stale_minutes": args.stale_minutes,
    }

    try:
        result = client.post("/sessions/flush-stale", params=params)
        output_result(result)
        return 0
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1
