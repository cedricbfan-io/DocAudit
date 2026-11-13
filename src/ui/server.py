"""Public DocAudit site in one process (behind a Cloudflare Tunnel).

    python -m src.ui.server            # http://127.0.0.1:7860
      /          landing page (site/index.html)
      /signup    create a demo account
      /login     sign in (signed session cookie)
      /app/      the DocAudit app — redirects to /login without a session

Accounts: src/ui/accounts.py (hashed passwords, stored outside the repo).
The app runs in public mode: per-visitor sessions, no LLM button, temp
exports deleted after about an hour.
"""

import argparse
import hashlib
import hmac
import html
import os
import secrets
import sys
import time
from pathlib import Path

os.environ.setdefault("DOCAUDIT_PUBLIC", "1")
os.environ.setdefault("DOCAUDIT_ACCOUNTS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gradio as gr  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, Form, Request  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse  # noqa: E402

from src.ui import accounts  # noqa: E402
from src.ui import app as da  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site"

_SIGNUP_CSS = """
:root{--blue:#0249E1;--sky:#80B0EC;--white:#fff;--ink:#111111;--lime:#DAFB71;--red:#EE3D5A}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;display:grid;place-items:center;background:var(--blue);
     font:16px/1.5 "Urbanist",-apple-system,"Segoe UI","PingFang SC",sans-serif;color:var(--ink);padding:24px 16px}
.card{background:var(--white);border-radius:28px;width:100%;max-width:420px;padding:14px;
      box-shadow:0 30px 60px rgba(0,20,80,.35)}
.band{border-radius:18px;padding:18px;color:#fff;
      background:radial-gradient(circle at 15% 130%,var(--red) 0 24%,transparent 25%),
                 radial-gradient(circle at 85% -40%,var(--lime) 0 30%,transparent 31%),
                 linear-gradient(135deg,#0a3fc0,var(--blue))}
.band small{font-weight:700;letter-spacing:.08em;text-transform:uppercase;font-size:12px}
.band h1{margin:6px 0 0;font-size:26px}
.cut{position:relative;height:26px;margin:6px -14px;overflow:hidden}
.cut:after{content:"";position:absolute;left:26px;right:26px;top:12px;border-top:2px dotted var(--sky)}
.cut i{position:absolute;top:0;width:26px;height:26px;border-radius:50%;background:var(--blue)}
.cut i:first-child{left:-13px}.cut i:last-child{right:-13px}
form{padding:4px 10px 10px}
label{display:block;font-size:13px;color:#5b6170;margin:12px 0 4px}
input{width:100%;padding:12px 14px;border:1.5px solid #d7dbe3;border-radius:12px;font:inherit}
input:focus{outline:none;border-color:var(--blue)}
button{margin-top:18px;width:100%;border:0;border-radius:999px;padding:13px;font-family:inherit;font-weight:700;font-size:16px;
       background:var(--lime);color:var(--ink);cursor:pointer}
.err{background:#fff0f3;border-left:3px solid var(--red);border-radius:8px;padding:8px 10px;font-size:14px;margin:4px 10px}
.foot{text-align:center;font-size:14px;color:#5b6170;margin:6px 0 4px}
.foot a{color:var(--blue);font-weight:600}
.ok{background:#f3fde0;border-left:3px solid #7ea51a;border-radius:8px;padding:8px 10px;font-size:14px;margin:4px 10px}
"""


_FOOT_NOTE = ('<p class="foot" style="font-size:12px">No email needed. This is a '
              'course demo — use a password you do not use elsewhere.</p>')


def _card(title: str, band: str, body: str, notice: str = "",
          error: str = "") -> str:
    """One page in the same ticket-card style (signup and login)."""
    msg = ""
    if error:
        msg = f'<p class="err">{html.escape(error)}</p>'
    elif notice:
        msg = f'<p class="ok">{html.escape(notice)}</p>'
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} · DocAudit</title>
<link href="https://fonts.googleapis.com/css2?family=Urbanist:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>{_SIGNUP_CSS}</style></head><body>
<main class="card">
  <div class="band"><small>DocAudit · demo account</small><h1>{html.escape(band)}</h1></div>
  <div class="cut"><i></i><i></i></div>
  {msg}
  {body}
</main></body></html>"""


def _signup_page(error: str = "", username: str = "") -> str:
    body = f"""<form method="post" action="/signup" autocomplete="on">
    <label for="u">Username</label>
    <input id="u" name="username" required minlength="3" maxlength="32"
           pattern="[A-Za-z0-9][A-Za-z0-9_.\\-]{{2,31}}" value="{html.escape(username)}" autocomplete="username">
    <label for="p">Password (at least 8 characters)</label>
    <input id="p" name="password" type="password" required minlength="8" maxlength="128" autocomplete="new-password">
    <label for="c">Repeat password</label>
    <input id="c" name="confirm" type="password" required minlength="8" maxlength="128" autocomplete="new-password">
    <button type="submit">Create account</button>
  </form>
  <div class="cut"><i></i><i></i></div>
  <p class="foot">Already have one? <a href="/login">Sign in</a> · <a href="/">About DocAudit</a></p>
  {_FOOT_NOTE}"""
    return _card("Create account", "Create your account", body, error=error)


def _login_page(error: str = "", username: str = "", notice: str = "") -> str:
    body = f"""<form method="post" action="/login" autocomplete="on">
    <label for="u">Username</label>
    <input id="u" name="username" required maxlength="32" value="{html.escape(username)}"
           autocomplete="username" autofocus>
    <label for="p">Password</label>
    <input id="p" name="password" type="password" required maxlength="128" autocomplete="current-password">
    <button type="submit">Sign in</button>
  </form>
  <div class="cut"><i></i><i></i></div>
  <p class="foot">No account yet? <a href="/signup">Create one</a> · <a href="/">About DocAudit</a></p>"""
    return _card("Sign in", "Welcome back", body, notice=notice, error=error)


# ── login session: a signed cookie, checked for every /app request ──
COOKIE = "docaudit_session"
SESSION_SECONDS = 12 * 3600


def _secret() -> bytes:
    """Per-installation signing key, next to the accounts database."""
    p = accounts.db_path().parent / "session_secret"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_bytes(secrets.token_bytes(32))
        os.chmod(p, 0o600)
    return p.read_bytes()


def make_session(username: str, now: float = None) -> str:
    exp = int((time.time() if now is None else now) + SESSION_SECONDS)
    payload = f"{username}|{exp}"
    sig = hmac.new(_secret(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}|{sig}"


def read_session(value: str):
    """Username for a valid, unexpired cookie value; otherwise None."""
    try:
        username, exp, sig = (value or "").rsplit("|", 2)
        good = hmac.new(_secret(), f"{username}|{exp}".encode(),
                        hashlib.sha256).hexdigest()
        if hmac.compare_digest(sig, good) and int(exp) > time.time():
            return username
    except (ValueError, TypeError):
        pass
    return None


def _same_origin(request: Request) -> bool:
    origin = request.headers.get("origin")
    return not origin or origin.split("://", 1)[-1] == request.headers.get("host", "")


def _https(request: Request) -> bool:
    return (request.headers.get("x-forwarded-proto") == "https"
            or request.url.scheme == "https")


def _client_ip(request: Request) -> str:
    # Cloudflare puts the visitor's address here; fall back to the socket
    return (request.headers.get("cf-connecting-ip")
            or (request.client.host if request.client else ""))


def create_app() -> FastAPI:
    api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @api.middleware("http")
    async def app_needs_login(request: Request, call_next):
        # opening the app without a session → our sign-in page
        if (request.method == "GET" and request.url.path in ("/app", "/app/")
                and not read_session(request.cookies.get(COOKIE))):
            return RedirectResponse("/login", status_code=303)
        return await call_next(request)

    @api.get("/", response_class=HTMLResponse)
    def landing():
        return (SITE / "index.html").read_text(encoding="utf-8")

    @api.get("/screenshot.jpg")
    def screenshot():
        return FileResponse(SITE / "screenshot.jpg")

    @api.get("/signup", response_class=HTMLResponse)
    def signup_form():
        return _signup_page()

    @api.post("/signup")
    def signup(request: Request, username: str = Form(""),
               password: str = Form(""), confirm: str = Form("")):
        if not _same_origin(request):
            return HTMLResponse(_signup_page("Please submit the form from this site."),
                                status_code=400)
        try:
            u = accounts.create_user(username, password, confirm,
                                     ip=_client_ip(request))
        except accounts.SignupError as e:
            return HTMLResponse(_signup_page(str(e), username), status_code=400)
        return HTMLResponse(_login_page(username=u, notice=(
            "Account created — sign in with your new password.")))

    @api.get("/login", response_class=HTMLResponse)
    def login_form(request: Request):
        if read_session(request.cookies.get(COOKIE)):
            return RedirectResponse("/app/", status_code=303)
        return _login_page()

    @api.post("/login")
    def login(request: Request, username: str = Form(""),
              password: str = Form("")):
        if not _same_origin(request):
            return HTMLResponse(_login_page("Please submit the form from this site."),
                                status_code=400)
        if not accounts.verify(username, password):
            return HTMLResponse(_login_page(
                "Wrong username or password (too many tries lock the "
                "account for 2 minutes).", username), status_code=401)
        resp = RedirectResponse("/app/", status_code=303)
        resp.set_cookie(COOKIE, make_session(accounts.normalise(username)),
                        max_age=SESSION_SECONDS, httponly=True,
                        samesite="lax", secure=_https(request), path="/")
        return resp

    @api.get("/logout")
    def logout():
        resp = RedirectResponse("/login", status_code=303)
        resp.delete_cookie(COOKIE, path="/")
        return resp

    da.EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    demo = da.build_ui()
    return gr.mount_gradio_app(
        api, demo, path="/app",
        # every Gradio request is checked against our session cookie;
        # None → 401 (the page itself redirects to /login above)
        auth_dependency=lambda request: read_session(request.cookies.get(COOKIE)),
        allowed_paths=[str(da.EXPORT_ROOT.resolve())],
        # nothing inside the project folder (.env, data/, code) is served
        blocked_paths=[str(ROOT)],
        css=da._BRAND_CSS,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description="DocAudit public site")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args(argv)
    print(f"DocAudit public site → http://{a.host}:{a.port}/  "
          f"(app at /app/, sign-up at /signup; accounts in {accounts.db_path()})",
          flush=True)
    uvicorn.run(create_app(), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
