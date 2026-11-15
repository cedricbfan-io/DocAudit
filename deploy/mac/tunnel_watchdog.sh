#!/usr/bin/env bash
# Keeps a Cloudflare quick tunnel to the local DocAudit server alive.
# Run by launchd (see docauditctl); can also be run by hand.
#
# Health is read from cloudflared itself (its /ready endpoint says whether
# it holds a connection to Cloudflare) — NOT by opening the public link from
# this computer, which a VPN/proxy app here can make fail or succeed
# independently of what visitors see.
#
# A quick tunnel gets a NEW address every time cloudflared starts. The
# current one is always in ~/.docaudit/public_url.txt (history next to it).
set -u
PORT="${DOCAUDIT_PORT:-7860}"
METRICS="127.0.0.1:${DOCAUDIT_METRICS_PORT:-20241}"
STATE="$HOME/.docaudit"; LOGS="$STATE/logs"; mkdir -p "$LOGS"
URL_FILE="$STATE/public_url.txt"; HIST="$STATE/public_url_history.log"
CF_LOG="$LOGS/cloudflared.log"
CHECK_EVERY=15          # seconds between health checks
MAX_FAILS=6             # 6 × 15 s = 90 s without an edge connection → restart
PROTOS=(${DOCAUDIT_TUNNEL_PROTOCOLS:-http2 quic})
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

say() { echo "$(date '+%F %T') $*"; }
notify() { /usr/bin/osascript -e "display notification \"$1\" with title \"DocAudit\"" >/dev/null 2>&1 || true; }
CF_PID=""
HERE="$(cd "$(dirname "$0")" && pwd)"
publish() { "$HERE/publish_link.sh" "$1" >> "$LOGS/tunnel.log" 2>&1 & }   # fixed GitHub Pages link (opt-in)
cleanup() { [ -n "$CF_PID" ] && kill "$CF_PID" 2>/dev/null; rm -f "$URL_FILE"; exit 0; }
trap cleanup TERM INT

command -v cloudflared >/dev/null || { say "cloudflared is not installed (brew install cloudflared)"; sleep 300; exit 1; }

attempt=0
while true; do
  until curl -fs -o /dev/null -m 5 "http://127.0.0.1:$PORT/healthz"; do
    say "waiting for the local server on port $PORT…"; sleep 5
  done
  proto="${PROTOS[$((attempt % ${#PROTOS[@]}))]}"; attempt=$((attempt + 1))
  : > "$CF_LOG"; rm -f "$URL_FILE"
  say "starting tunnel (protocol $proto)"
  cloudflared tunnel --no-autoupdate --protocol "$proto" --metrics "$METRICS" \
    --url "http://127.0.0.1:$PORT" >> "$CF_LOG" 2>&1 &
  CF_PID=$!

  url=""
  for _ in $(seq 1 30); do                       # up to 60 s for the address
    url="$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$CF_LOG" | head -1)"
    [ -n "$url" ] && break
    kill -0 "$CF_PID" 2>/dev/null || break
    sleep 2
  done
  if [ -z "$url" ]; then
    say "no tunnel address after 60 s — retrying"
    kill "$CF_PID" 2>/dev/null; wait "$CF_PID" 2>/dev/null; sleep 10; continue
  fi

  fails=0; announced=""
  while kill -0 "$CF_PID" 2>/dev/null; do
    if curl -fs -o /dev/null -m 5 "http://$METRICS/ready"; then
      fails=0
      if [ -z "$announced" ]; then                # publish only once it really works
        announced=1; echo "$url" > "$URL_FILE"; echo "$(date '+%F %T') $url ($proto)" >> "$HIST"
        say "PUBLIC LINK: $url"; notify "Public link: $url"
        publish "$url"
      fi
    else
      fails=$((fails + 1)); say "tunnel not connected ($fails/$MAX_FAILS)"
      if [ "$fails" -ge "$MAX_FAILS" ]; then
        say "tunnel stayed down — starting a new one (the link will change)"
        notify "Public link is down — getting a new one"
        break
      fi
    fi
    sleep "$CHECK_EVERY"
  done
  kill "$CF_PID" 2>/dev/null; wait "$CF_PID" 2>/dev/null; CF_PID=""
  rm -f "$URL_FILE"
  sleep 5
done
