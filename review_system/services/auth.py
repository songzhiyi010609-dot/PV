from __future__ import annotations

import json
import secrets
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


PLACEHOLDER_PASSWORDS = {"", "replace_with_password", "change_me"}


def load_users(users_path: Path) -> tuple[dict[str, dict[str, str]], bool]:
    if not users_path.exists():
        return {}, True
    payload = json.loads(users_path.read_text(encoding="utf-8-sig"))
    raw_users = payload.get("users", payload) if isinstance(payload, dict) else {}
    users: dict[str, dict[str, str]] = {}
    setup_needed = False
    for username, value in raw_users.items():
        if isinstance(value, dict):
            password = str(value.get("password", ""))
            display_name = str(value.get("display_name", username))
        else:
            password = str(value)
            display_name = str(username)
        if password.strip() in PLACEHOLDER_PASSWORDS:
            setup_needed = True
        users[str(username)] = {"password": password, "display_name": display_name}
    if not users:
        setup_needed = True
    return users, setup_needed


def verify_user(users: dict[str, dict[str, str]], username: str, password: str) -> bool:
    user = users.get(username)
    if not user:
        return False
    expected = user.get("password", "")
    if expected.strip() in PLACEHOLDER_PASSWORDS:
        return False
    return secrets.compare_digest(expected, password)


def create_session(conn: sqlite3.Connection, username: str, ttl_hours: int) -> str:
    token = secrets.token_urlsafe(32)
    expires = datetime.now() + timedelta(hours=ttl_hours)
    conn.execute(
        "insert into sessions(token, username, expires_at) values (?, ?, ?)",
        (token, username, expires.isoformat(timespec="seconds")),
    )
    conn.commit()
    return token


def get_session_user(conn: sqlite3.Connection, token: str | None) -> str | None:
    if not token:
        return None
    row = conn.execute("select username, expires_at from sessions where token = ?", (token,)).fetchone()
    if row is None:
        return None
    try:
        expires = datetime.fromisoformat(str(row["expires_at"]))
    except ValueError:
        return None
    if expires < datetime.now():
        conn.execute("delete from sessions where token = ?", (token,))
        conn.commit()
        return None
    return str(row["username"])


def delete_session(conn: sqlite3.Connection, token: str | None) -> None:
    if not token:
        return
    conn.execute("delete from sessions where token = ?", (token,))
    conn.commit()
