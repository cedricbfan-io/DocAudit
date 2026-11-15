"""Public site: sign-up, login gate, and nothing private served."""

from pathlib import Path

import pytest

pytest.importorskip("gradio")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCAUDIT_USERS_DB", str(tmp_path / "users.sqlite"))
    from src.ui import accounts, server
    accounts._signups.clear()
    accounts._fails.clear()
    return TestClient(server.create_app())


def test_password_is_hashed_not_stored(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCAUDIT_USERS_DB", str(tmp_path / "u.sqlite"))
    from src.ui import accounts
    accounts.create_user("Alice", "correct horse", "correct horse")
    raw = (tmp_path / "u.sqlite").read_bytes()
    assert b"correct horse" not in raw
    assert accounts.verify("alice", "correct horse")
    assert not accounts.verify("alice", "wrong password")
    assert not accounts.verify("nobody", "correct horse")


@pytest.mark.parametrize("u,p,c,msg", [
    ("ab", "longenough", "longenough", "Username"),
    ("bob", "short", "short", "at least 8"),
    ("bob", "longenough", "different1", "do not match"),
])
def test_signup_validation(tmp_path, monkeypatch, u, p, c, msg):
    monkeypatch.setenv("DOCAUDIT_USERS_DB", str(tmp_path / "u.sqlite"))
    from src.ui import accounts
    with pytest.raises(accounts.SignupError, match=msg):
        accounts.create_user(u, p, c)


def test_login_locks_after_repeated_failures(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCAUDIT_USERS_DB", str(tmp_path / "u.sqlite"))
    from src.ui import accounts
    accounts._fails.clear()
    accounts.create_user("carol", "goodpassword", "goodpassword")
    for _ in range(accounts.FAILS_BEFORE_LOCK):
        assert not accounts.verify("carol", "bad")
    assert not accounts.verify("carol", "goodpassword")      # locked for now
    accounts._fails.clear()


def test_signup_rate_limit_per_ip(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCAUDIT_USERS_DB", str(tmp_path / "u.sqlite"))
    from src.ui import accounts
    accounts._signups.clear()
    for i in range(accounts.SIGNUPS_PER_IP_PER_HOUR):
        accounts.create_user(f"user{i}", "goodpassword", ip="1.2.3.4")
    with pytest.raises(accounts.SignupError, match="Too many"):
        accounts.create_user("user99", "goodpassword", ip="1.2.3.4")
    accounts._signups.clear()


def _signup(client, name, pw="goodpassword"):
    return client.post("/signup", data={"username": name, "password": pw,
                                        "confirm": pw})


def _login(client, name, pw="goodpassword"):
    return client.post("/login", data={"username": name, "password": pw},
                       follow_redirects=False)


def test_site_flow(client):
    assert "DocAudit" in client.get("/").text
    assert client.get("/signup").status_code == 200
    assert "Welcome back" in client.get("/login").text
    # the app needs a login: page → our sign-in page, API → 401
    r = client.get("/app/", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login"
    assert client.get("/app/config").status_code in (401, 403)

    r = _signup(client, "dave")
    assert r.status_code == 200 and "Account created" in r.text
    assert _signup(client, "dave").status_code == 400

    r = _login(client, "dave", "wrongpass1")
    assert r.status_code == 401 and "Wrong username or password" in r.text
    assert client.get("/app/config").status_code in (401, 403)

    r = _login(client, "dave")
    assert r.status_code == 303 and r.headers["location"] == "/app/"
    assert "httponly" in r.headers["set-cookie"].lower()
    assert client.get("/app/config").status_code == 200
    assert client.get("/app/", follow_redirects=False).status_code == 200

    client.get("/logout")
    assert client.get("/app/config").status_code in (401, 403)


def test_tampered_or_expired_session_rejected(client):
    from src.ui import server
    assert _signup(client, "dave").status_code == 200
    good = server.make_session("dave")
    assert server.read_session(good) == "dave"
    user, exp, sig = good.rsplit("|", 2)
    assert server.read_session(f"admin|{exp}|{sig}") is None
    assert server.read_session(server.make_session("dave", now=0)) is None
    assert server.read_session("garbage") is None


def test_deleted_account_session_rejected(client):
    import sqlite3
    from src.ui import accounts, server
    assert _signup(client, "erin").status_code == 200
    assert _login(client, "erin").status_code == 303
    assert client.get("/app/config").status_code == 200
    with sqlite3.connect(accounts.db_path()) as c:
        c.execute("DELETE FROM users WHERE username='erin'")
    # the still-signed cookie no longer opens the app
    assert server.read_session(server.make_session("erin")) is None
    assert client.get("/app/config").status_code in (401, 403)
    r = client.get("/app/", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login"


def test_cross_site_forms_rejected(client):
    evil = {"origin": "https://evil.example"}
    r = client.post("/signup", headers=evil,
                    data={"username": "eve", "password": "goodpassword",
                          "confirm": "goodpassword"})
    assert r.status_code == 400
    _signup(client, "eve")
    r = client.post("/login", headers=evil,
                    data={"username": "eve", "password": "goodpassword"})
    assert r.status_code == 400


def test_project_files_are_not_served(client):
    _signup(client, "frank")
    _login(client, "frank")
    root = Path(__file__).resolve().parents[1]
    for rel in ("README.md", "config/thresholds.yaml"):
        r = client.get(f"/app/gradio_api/file={root / rel}")
        assert r.status_code in (403, 404), (rel, r.status_code)


# ── 15/11: health check + local-only status page ──
def _client_for_local():
    from fastapi.testclient import TestClient
    from src.ui import server
    return TestClient(server.create_app()), server


def test_healthz_needs_no_login():
    c, _ = _client_for_local()
    assert c.get("/healthz").json() == {"ok": True}


def test_local_status_page_is_hidden_from_tunnel_visitors(tmp_path, monkeypatch):
    c, server = _client_for_local()
    monkeypatch.setattr(server, "_is_local", lambda request: False)
    assert c.get("/local").status_code == 404


def test_is_local_rejects_requests_that_came_through_cloudflare():
    from types import SimpleNamespace as NS
    from src.ui.server import _is_local
    direct = NS(headers={}, client=NS(host="127.0.0.1"))
    tunnel = NS(headers={"cf-connecting-ip": "1.2.3.4"}, client=NS(host="127.0.0.1"))
    lan = NS(headers={}, client=NS(host="10.0.0.9"))
    assert _is_local(direct) and not _is_local(tunnel) and not _is_local(lan)


def test_local_status_page_shows_the_current_public_link(tmp_path, monkeypatch):
    c, server = _client_for_local()
    monkeypatch.setattr(server, "_is_local", lambda request: True)
    monkeypatch.setattr(server, "public_url", lambda: None)
    assert "No public link right now" in c.get("/local").text
    monkeypatch.setattr(server, "public_url",
                        lambda: "https://abc-def.trycloudflare.com")
    assert "https://abc-def.trycloudflare.com" in c.get("/local").text
