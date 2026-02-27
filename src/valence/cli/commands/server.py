# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Server management commands — restart, status, logs.

Commands:
    valence server status     Show server PID, uptime, version, code freshness
    valence server restart    Restart via LaunchAgent (unload + load)
    valence server logs       Tail server logs
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

from ..output import output_error, output_result

LAUNCHD_LABEL = "com.ourochronos.valence-server"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"
HEALTH_URL = "http://127.0.0.1:8420/health"
LOG_DIR = Path.home() / ".valence" / "logs"


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register server subcommands."""
    server_p = subparsers.add_parser("server", help="Server management")
    server_sub = server_p.add_subparsers(dest="server_command", required=True)

    # status
    status_p = server_sub.add_parser("status", help="Show server status")
    status_p.set_defaults(func=cmd_server_status)

    # restart
    restart_p = server_sub.add_parser("restart", help="Restart the server")
    restart_p.add_argument("--timeout", type=int, default=15, help="Seconds to wait for health check")
    restart_p.set_defaults(func=cmd_server_restart)

    # logs
    logs_p = server_sub.add_parser("logs", help="Tail server logs")
    logs_p.add_argument("-n", "--lines", type=int, default=50, help="Number of lines to show")
    logs_p.add_argument("-f", "--follow", action="store_true", help="Follow log output")
    logs_p.add_argument("--err", action="store_true", help="Show error log instead of stdout")
    logs_p.set_defaults(func=cmd_server_logs)


def _get_server_pid() -> int | None:
    """Get the PID of the running Valence server."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "uvicorn.*valence"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            pids = result.stdout.strip().split("\n")
            # Return the first (main) PID
            return int(pids[0]) if pids[0] else None
    except Exception:
        pass
    return None


def _check_health(timeout: float = 5.0) -> dict | None:
    """Hit the health endpoint and return response, or None on failure."""
    try:
        import httpx

        resp = httpx.get(HEALTH_URL, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def _get_git_head() -> str | None:
    """Get current git HEAD commit (short hash)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path.home() / "projects" / "valence",
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _launchd_loaded() -> bool:
    """Check if the LaunchAgent is loaded."""
    try:
        result = subprocess.run(
            ["launchctl", "list", LAUNCHD_LABEL],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def cmd_server_status(args: argparse.Namespace) -> int:
    """Show server status."""
    pid = _get_server_pid()
    health = _check_health()
    loaded = _launchd_loaded()
    git_head = _get_git_head()

    status: dict = {
        "running": pid is not None,
        "pid": pid,
        "launchd_loaded": loaded,
        "launchd_label": LAUNCHD_LABEL,
    }

    if health:
        status["health"] = health.get("status", "unknown")
        status["version"] = health.get("version", "unknown")
        status["database"] = health.get("database", "unknown")
        if "started_at" in health:
            status["started_at"] = health["started_at"]
        if "last_write_at" in health:
            status["last_write_at"] = health["last_write_at"]
        if "git_commit" in health:
            status["server_commit"] = health["git_commit"]
    else:
        status["health"] = "unreachable"

    if git_head:
        status["repo_head"] = git_head
        server_commit = status.get("server_commit")
        if server_commit and server_commit != git_head:
            status["code_stale"] = True
            status["warning"] = f"Server running {server_commit}, repo at {git_head} — restart needed"

    output_result(status)
    return 0


def cmd_server_restart(args: argparse.Namespace) -> int:
    """Restart the server via LaunchAgent."""
    timeout = args.timeout

    if not PLIST_PATH.exists():
        output_error(f"LaunchAgent plist not found: {PLIST_PATH}")
        return 1

    old_pid = _get_server_pid()
    print(f"Restarting Valence server (PID {old_pid or 'unknown'})...")

    # Try launchctl kickstart -k first (cleanest restart)
    try:
        result = subprocess.run(
            [
                "launchctl",
                "kickstart",
                "-k",
                f"gui/{os.getuid()}/{LAUNCHD_LABEL}",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return _wait_for_health(timeout, old_pid)
    except Exception:
        pass

    # Fallback: unload + load
    print("  Using unload/load fallback...")
    subprocess.run(
        ["launchctl", "unload", str(PLIST_PATH)],
        capture_output=True,
    )
    time.sleep(2)

    subprocess.run(
        ["launchctl", "load", str(PLIST_PATH)],
        capture_output=True,
    )

    return _wait_for_health(timeout, old_pid)


def _wait_for_health(timeout: int, old_pid: int | None) -> int:
    """Wait for the server to come up healthy with a new PID."""
    print("  Waiting for health check...", end="", flush=True)

    start = time.time()
    while time.time() - start < timeout:
        time.sleep(1)
        print(".", end="", flush=True)

        new_pid = _get_server_pid()
        if new_pid and new_pid != old_pid:
            health = _check_health(timeout=3.0)
            if health and health.get("status") == "healthy":
                print()
                print(f"  ✅ Server restarted (PID {new_pid})")
                if health.get("database") != "connected":
                    print(f"  ⚠️  Database: {health.get('database', 'unknown')}")
                return 0

    print()
    # Check if it came up at all
    new_pid = _get_server_pid()
    if new_pid:
        health = _check_health(timeout=3.0)
        if health:
            print(f"  ✅ Server running (PID {new_pid}) but took longer than expected")
            return 0
        else:
            print(f"  ⚠️  Server process exists (PID {new_pid}) but health check fails")
            print("     Check logs: valence server logs --err")
            return 1
    else:
        output_error("Server failed to start. Check: valence server logs --err")
        return 1


def cmd_server_logs(args: argparse.Namespace) -> int:
    """Tail server logs."""
    log_file = LOG_DIR / ("server.err.log" if args.err else "server.log")

    if not log_file.exists():
        # Check old location
        alt = Path.home() / "projects" / "valence" / "nohup.out"
        if alt.exists():
            log_file = alt
        else:
            output_error(f"Log file not found: {log_file}")
            return 1

    if args.follow:
        try:
            os.execvp("tail", ["tail", "-f", "-n", str(args.lines), str(log_file)])
        except Exception as e:
            output_error(f"Failed to tail: {e}")
            return 1
    else:
        try:
            result = subprocess.run(
                ["tail", "-n", str(args.lines), str(log_file)],
                capture_output=True,
                text=True,
            )
            print(result.stdout, end="")
            return 0
        except Exception as e:
            output_error(f"Failed to read log: {e}")
            return 1
