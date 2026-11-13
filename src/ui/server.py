"""Public DocAudit site in one process (behind a Cloudflare Tunnel).

    python -m src.ui.server            # http://127.0.0.1:7860
      /          landing page (site/index.html)
      /signup    create a demo account
      /app/      the DocAudit app — Gradio login page first

Accounts: src/ui/accounts.py (hashed passwords, stored outside the repo).
The app runs in public mode: per-visitor sessions, no LLM button, temp
exports deleted after about an hour.
"""

import argparse
import html
import os
import sys
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
"""


def _signup_page(error: str = "", username: str = "") -> str:
    err = f'<p class="err">{html.escape(error)}</p>' if error else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Create account · DocAudit</title>
<link href="https://fonts.googleapis.com/css2?family=Urbanist:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>{_SIGNUP_CSS}</style></head><body>
<main class="card">
  <div class="band"><small>DocAudit · demo account</small><h1>Create your account</h1></div>
  <div class="cut"><i></i><i></i></div>
  {err}
  <form method="post" action="/signup" autocomplete="on">
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
  <p class="foot">Already have one? <a href="/app/">Sign in</a> · <a href="/">About DocAudit</a></p>
  <p class="foot" style="font-size:12px">No email needed. This is a course demo — use a password you do not use elsewhere.</p>
</main></body></html>"""


def _client_ip(request: Request) -> str:
    # Cloudflare puts the visitor's address here; fall back to the socket
    return (request.headers.get("cf-connecting-ip")
            or (request.client.host if request.client else ""))


def create_app() -> FastAPI:
    api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

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
        origin = request.headers.get("origin")
        host = request.headers.get("host", "")
        if origin and origin.split("://", 1)[-1] != host:
            return HTMLResponse(_signup_page("Please submit the form from this site."),
                                status_code=400)
        try:
            accounts.create_user(username, password, confirm,
                                 ip=_client_ip(request))
        except accounts.SignupError as e:
            return HTMLResponse(_signup_page(str(e), username), status_code=400)
        return RedirectResponse("/app/?created=1", status_code=303)

    da.EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    demo = da.build_ui()
    return gr.mount_gradio_app(
        api, demo, path="/app",
        auth=accounts.verify,
        auth_message=("Sign in with your DocAudit demo account. "
                      "No account yet? Open /signup on this site to create one."),
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
