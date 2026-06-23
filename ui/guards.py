from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st

from config import (
    CACHE_DIR,
    MAX_CONCURRENT_USERS,
    MIN_REFRESH_INTERVAL_MINUTES,
    RETRAIN_COOLDOWN_HOURS,
)

_SESSIONS_FILE = CACHE_DIR / ".active_sessions.json"
_RETRAIN_LOCK_FILE = CACHE_DIR / ".retrain_lock"
_SESSION_TIMEOUT_MINUTES = 5


def _get_session_id() -> str | None:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        return str(ctx.session_id) if ctx else None
    except Exception:
        return None


def _load_sessions() -> dict:
    if not _SESSIONS_FILE.exists():
        return {}
    try:
        return json.loads(_SESSIONS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_sessions(sessions: dict) -> None:
    try:
        _SESSIONS_FILE.write_text(json.dumps(sessions), encoding="utf-8")
    except Exception:
        pass


def update_session_heartbeat() -> None:
    """Register or refresh the current session. Called once per page render."""
    session_id = _get_session_id()
    if session_id is None:
        return
    sessions = _load_sessions()
    now = time.time()
    cutoff = now - _SESSION_TIMEOUT_MINUTES * 60
    sessions = {k: v for k, v in sessions.items() if v > cutoff}
    sessions[session_id] = now
    _save_sessions(sessions)


def check_concurrent_users() -> None:
    """Stop rendering and show an error if the server is over capacity."""
    sessions = _load_sessions()
    now = time.time()
    cutoff = now - _SESSION_TIMEOUT_MINUTES * 60
    active_count = sum(1 for v in sessions.values() if v > cutoff)
    if active_count > MAX_CONCURRENT_USERS:
        st.error(
            f"Server is at capacity ({active_count} / {MAX_CONCURRENT_USERS} active users). "
            "Please try again in a few minutes."
        )
        st.info(
            "This dashboard runs on a free-tier server with limited resources. "
            "Sessions expire automatically after 5 minutes of inactivity."
        )
        st.stop()


def check_retrain_allowed() -> tuple[bool, str]:
    """Return (allowed, warning_message). False when cooldown is still active."""
    if not _RETRAIN_LOCK_FILE.exists():
        return True, ""
    age_hours = (time.time() - _RETRAIN_LOCK_FILE.stat().st_mtime) / 3600
    if age_hours < RETRAIN_COOLDOWN_HOURS:
        remaining = RETRAIN_COOLDOWN_HOURS - age_hours
        return False, (
            f"Models were retrained {age_hours:.1f}h ago. "
            f"Next retrain available in {remaining:.1f}h."
        )
    return True, ""


def record_retrain() -> None:
    """Touch the lock file to start the retrain cooldown timer."""
    try:
        _RETRAIN_LOCK_FILE.touch()
    except Exception:
        pass


def enforce_min_refresh(interval_minutes: int) -> int:
    """Clamp refresh interval to MIN_REFRESH_INTERVAL_MINUTES and warn if adjusted."""
    if 0 < interval_minutes < MIN_REFRESH_INTERVAL_MINUTES:
        st.sidebar.warning(
            f"Minimum refresh is {MIN_REFRESH_INTERVAL_MINUTES} min to protect server resources. "
            "Adjusted automatically."
        )
        return MIN_REFRESH_INTERVAL_MINUTES
    return interval_minutes
