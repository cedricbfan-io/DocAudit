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
# The tunnel can lose its connection to Cloudflare for good (seen with a VPN /
# proxy app on this computer: "TLS handshake with edge error" repeated for
# hours while the old link stayed dead). When that happens, start a fresh
# tunnel. NOTE: a fresh quick tunnel has a NEW address — it is printed again
# and written to ~/.docaudit/public_url.txt.
LOG="$(mktemp -t docaudit_tunnel)"
mkdir -p "$HOME/.docaudit"
trap 'kill $APP_PID ${CF_PID:-} 2>/dev/null' EXIT
while true; do
  : > "$LOG"
  cloudflared tunnel --no-autoupdate --protocol "${TUNNEL_PROTOCOL:-http2}" \
    --url "http://127.0.0.1:$PORT" 2>&1 | tee "$LOG" &
  CF_PID=$!
  shown=""
  while kill -0 $CF_PID 2>/dev/null; do
    sleep 20
    url="$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$LOG" | head -1 || true)"
    if [ -n "$url" ] && [ "$url" != "$shown" ]; then
      shown="$url"; echo "$url" > "$HOME/.docaudit/public_url.txt"
      echo; echo "==> PUBLIC LINK: $url"; echo
    fi
    # 4 failed reconnects in a row with no successful registration after them
    if [ "$(tail -n 12 "$LOG" | grep -c 'Unable to establish connection')" -ge 4 ] \
       && ! tail -n 12 "$LOG" | grep -q 'Registered tunnel connection'; then
      echo "==> Tunnel lost its connection — starting a new one (the link will change)."
      pkill -P $CF_PID 2>/dev/null || true; pkill -f "cloudflared tunnel .*127.0.0.1:$PORT" 2>/dev/null || true
      break
    fi
  done
  wait $CF_PID 2>/dev/null || true
  sleep 3
done
