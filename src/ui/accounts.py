"""Demo user accounts for the public DocAudit site.

Stored in SQLite OUTSIDE the repository (default ~/.docaudit/users.sqlite,
override with DOCAUDIT_USERS_DB). Passwords are never stored: only a
PBKDF2-SHA256 hash with a per-user random salt. Simple abuse limits:
sign-ups per client IP per hour, a total account cap, and a short lock
after repeated failed logins for one username.
"""

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import threading
import time
from pathlib import Path

_ITER = 200_000
_USER_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,31}$")
MAX_USERS = int(os.environ.get("DOCAUDIT_MAX_USERS", "500"))
SIGNUPS_PER_IP_PER_HOUR = 5
FAILS_BEFORE_LOCK = 8
LOCK_SECONDS = 120

_lock = threading.Lock()
_signups = {}          # ip -> [timestamps]
_fails = {}            # username -> (count, first_fail_time)


class SignupError(ValueError):
    pass


def db_path() -> Path:
    return Path(os.environ.get("DOCAUDIT_USERS_DB",
                               Path.home() / ".docaudit" / "users.sqlite"))


def _conn():
    p = db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE IF NOT EXISTS users ("
              "username TEXT PRIMARY KEY, salt BLOB NOT NULL, "
              "hash BLOB NOT NULL, created REAL NOT NULL)")
    return c


def _hash(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITER)


def normalise(username: str) -> str:
    return (username or "").strip().lower()


def create_user(username: str, password: str, confirm: str = None,
                ip: str = "") -> str:
    """Create an account; returns the normalised username.

    Raises SignupError with a message that is safe to show the visitor.
    """
    u = normalise(username)
    if not _USER_RE.match(u):
        raise SignupError("Username: 3–32 characters, letters, digits, "
                          "'.', '_' or '-', starting with a letter or digit.")
    if len(password or "") < 8:
        raise SignupError("Password: at least 8 characters.")
    if len(password) > 128:
        raise SignupError("Password: at most 128 characters.")
    if confirm is not None and confirm != password:
        raise SignupError("The two passwords do not match.")
    now = time.time()
    with _lock:
        recent = [t for t in _signups.get(ip, []) if now - t < 3600]
        if ip and len(recent) >= SIGNUPS_PER_IP_PER_HOUR:
            raise SignupError("Too many new accounts from this network. "
                              "Try again in an hour.")
        with _conn() as c:
            (n,) = c.execute("SELECT COUNT(*) FROM users").fetchone()
            if n >= MAX_USERS:
                raise SignupError("The demo is full right now. "
                                  "Please contact the team.")
            if c.execute("SELECT 1 FROM users WHERE username=?",
                         (u,)).fetchone():
                raise SignupError("That username is taken.")
            salt = secrets.token_bytes(16)
            c.execute("INSERT INTO users VALUES (?,?,?,?)",
                      (u, salt, _hash(password, salt), now))
        _signups[ip] = recent + [now]
    return u


def verify(username: str, password: str) -> bool:
    """Login check used by Gradio's auth= (True = let in)."""
    u = normalise(username)
    now = time.time()
    with _lock:
        count, first = _fails.get(u, (0, now))
        if count >= FAILS_BEFORE_LOCK and now - first < LOCK_SECONDS:
            return False
        if now - first >= LOCK_SECONDS:
            count, first = 0, now
    with _conn() as c:
        row = c.execute("SELECT salt, hash FROM users WHERE username=?",
                        (u,)).fetchone()
    if row is None:
        _hash(password or "", b"0" * 16)        # same work for unknown users
        ok = False
    else:
        ok = hmac.compare_digest(_hash(password or "", row[0]), row[1])
    with _lock:
        if ok:
            _fails.pop(u, None)
        else:
            _fails[u] = (count + 1, first)
    return ok


def count_users() -> int:
    with _conn() as c:
        return c.execute("SELECT COUNT(*) FROM users").fetchone()[0]
