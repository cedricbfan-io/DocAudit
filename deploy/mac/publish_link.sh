#!/usr/bin/env bash
# Tell the fixed GitHub Pages address where the demo is right now.
#   publish_link.sh https://xxxx.trycloudflare.com      (or: publish_link.sh off)
# Rewrites url.json on the gh-pages branch of the repo named in
# ~/.docaudit/pages_repo (one line, "owner/repo"). Does nothing if that file
# is missing — the fixed link is opt-in (docauditctl pages-setup).
set -u
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
STATE="$HOME/.docaudit"; CONF="$STATE/pages_repo"
[ -s "$CONF" ] || exit 0
REPO="$(head -1 "$CONF" | tr -d '[:space:]')"; BRANCH=gh-pages
URL="${1:-off}"
case "$URL" in
  off) json='{"url": null, "updated": "'"$(date -u '+%Y-%m-%d %H:%M UTC')"'"}' ;;
  https://*.trycloudflare.com) json='{"url": "'"$URL"'", "updated": "'"$(date -u '+%Y-%m-%d %H:%M UTC')"'"}' ;;
  *) echo "publish_link: not a tunnel address: $URL"; exit 2 ;;
esac
b64="$(printf '%s\n' "$json" | base64 | tr -d '\n')"
for try in 1 2 3 4 5 6; do            # the GitHub API is flaky behind a proxy
  sha="$(gh api "repos/$REPO/contents/url.json?ref=$BRANCH" --jq .sha 2>/dev/null)" || sha=""
  if [ -n "$sha" ] && gh api -X PUT "repos/$REPO/contents/url.json" -f branch="$BRANCH" \
        -f message="demo address: ${URL#https://}" -f content="$b64" -f sha="$sha" >/dev/null 2>&1; then
    echo "$(date '+%F %T') fixed link updated → $URL"; exit 0
  fi
  sleep $((try * 5))
done
echo "$(date '+%F %T') could NOT update the fixed link (GitHub unreachable?) → $URL"; exit 1
