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
    good = server.make_session("dave")
    assert server.read_session(good) == "dave"
    user, exp, sig = good.rsplit("|", 2)
    assert server.read_session(f"admin|{exp}|{sig}") is None
    assert server.read_session(server.make_session("dave", now=0)) is None
    assert server.read_session("garbage") is None


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
