#!/usr/bin/env bash
# Public DocAudit site on this computer, reachable from the internet through
# a Cloudflare quick tunnel (free, no Cloudflare account or domain needed).
#
#   bash start_public.sh          (PYTHON=.venv/bin/python bash start_public.sh)
#
# Prints a https://<random>.trycloudflare.com address — share that link.
# The address changes every time this script starts. Stop with Ctrl+C.
# Visitors: landing page "/", sign-up "/signup", app "/app/".
# Accounts are stored in ~/.docaudit/users.sqlite (hashed passwords).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
PORT="${PORT:-7860}"
command -v cloudflared >/dev/null || { echo "Install cloudflared first: brew install cloudflared"; exit 1; }

"$PY" -m src.ui.server --port "$PORT" &
APP_PID=$!
trap 'kill $APP_PID 2>/dev/null' EXIT

for _ in $(seq 1 60); do
  curl -s -o /dev/null "http://127.0.0.1:$PORT/" && break
  sleep 1
done
echo "DocAudit is running locally on http://127.0.0.1:$PORT/ — opening the tunnel…"
echo "Look for the line with https://….trycloudflare.com below and share it."
cloudflared tunnel --no-autoupdate --protocol "${TUNNEL_PROTOCOL:-http2}" --url "http://127.0.0.1:$PORT"
