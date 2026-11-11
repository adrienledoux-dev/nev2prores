#!/usr/bin/env bash
set -euo pipefail

# Automates Google Search Console verification & sitemap submission via `gcloud` CLI:
#   1. Obtains an OAuth token with `siteverification` + `webmasters` scopes (prompting once in browser if needed)
#   2. Requests the official `<meta name="google-site-verification" ...>` tag from Google Site Verification API
#   3. Injects the meta tag into `docs/index.html`, commits & pushes to GitHub Pages, and waits until live
#   4. Calls Google Site Verification API to verify ownership of https://adrienledoux-dev.github.io/nev2prores/
#   5. Registers the property and submits `sitemap.xml` in Google Search Console API

export PATH="$HOME/google-cloud-sdk/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SITE_URL="https://adrienledoux-dev.github.io/nev2prores/"
SITEMAP_URL="${SITE_URL}sitemap.xml"
ENCODED_SITE_URL="https%3A%2F%2Fadrienledoux-dev.github.io%2Fnev2prores%2F"
ENCODED_SITEMAP_URL="https%3A%2F%2Fadrienledoux-dev.github.io%2Fnev2prores%2Fsitemap.xml"
GCP_PROJECT="$(gcloud config get-value project 2>/dev/null || echo "adrien-automation")"

REQUIRED_SCOPES="https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/siteverification,https://www.googleapis.com/auth/webmasters"

echo "[1/5] Checking Google Cloud CLI OAuth scopes for Search Console & Site Verification..."
TOKEN="$(gcloud auth application-default print-access-token 2>/dev/null || true)"
HAS_SCOPES="false"
if [ -n "$TOKEN" ]; then
    SCOPE_INFO="$(curl -s "https://oauth2.googleapis.com/tokeninfo?access_token=${TOKEN}" || true)"
    if echo "$SCOPE_INFO" | grep -q "siteverification" && echo "$SCOPE_INFO" | grep -q "webmasters"; then
        HAS_SCOPES="true"
    fi
fi

if [ "$HAS_SCOPES" != "true" ]; then
    echo "  -> Opening browser once to authorize Google Site Verification & Search Console scopes..."
    gcloud auth application-default login --scopes="$REQUIRED_SCOPES"
    TOKEN="$(gcloud auth application-default print-access-token)"
fi

echo "[2/5] Requesting HTML <meta> verification token for ${SITE_URL}..."
TOKEN_RESP="$(curl -sS -X POST \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "X-Goog-User-Project: ${GCP_PROJECT}" \
    -H "Content-Type: application/json" \
    -d "{\"site\":{\"type\":\"SITE\",\"identifier\":\"${SITE_URL}\"},\"verificationMethod\":\"META\"}" \
    "https://www.googleapis.com/siteVerification/v1/token")"

META_TAG="$(python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); print(d.get("token",""))' <<< "$TOKEN_RESP")"
if [ -z "$META_TAG" ]; then
    echo "❌ Failed to retrieve verification token from Google API:"
    echo "$TOKEN_RESP"
    exit 1
fi
echo "  -> Received meta tag: $META_TAG"

echo "[3/5] Injecting verification tag into docs/index.html & deploying to GitHub Pages..."
if ! grep -Fq "$META_TAG" docs/index.html; then
    python3 - "$META_TAG" << 'PYEOF'
import sys, re

meta_tag = sys.argv[1]
with open("docs/index.html", "r", encoding="utf-8") as f:
    html = f.read()

# Replace any existing google-site-verification tag or insert right after <meta name="robots" ...>
if 'name="google-site-verification"' in html:
    html = re.sub(r'<meta\s+name="google-site-verification"[^>]*>', meta_tag, html)
else:
    html = html.replace(
        '<meta name="robots" content="index, follow">',
        f'<meta name="robots" content="index, follow">\n  {meta_tag}'
    )

with open("docs/index.html", "w", encoding="utf-8") as f:
    f.write(html)
PYEOF

    git add docs/index.html
    git commit -m "Add Google Search Console site verification meta tag"
    git push origin HEAD
fi

echo "  -> Waiting for GitHub Pages (${SITE_URL}) to serve the verification tag..."
TOKEN_VALUE="$(python3 -c 'import re,sys; m=re.search(r"content=\"([^\"]+)\"", sys.argv[1]); print(m.group(1) if m else "")' "$META_TAG")"
for attempt in $(seq 1 24); do
    LIVE_HTML="$(curl -sL "${SITE_URL}?t=$(date +%s)" || true)"
    if echo "$LIVE_HTML" | grep -Fq "$TOKEN_VALUE"; then
        echo "  -> Verification tag is live on ${SITE_URL}!"
        break
    fi
    sleep 5
done

echo "[4/5] Verifying ownership with Google Site Verification API..."
curl -sS -X POST \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "X-Goog-User-Project: ${GCP_PROJECT}" \
    -H "Content-Type: application/json" \
    -d "{\"site\":{\"type\":\"SITE\",\"identifier\":\"${SITE_URL}\"}}" \
    "https://www.googleapis.com/siteVerification/v1/webResource?verificationMethod=META"
echo ""

echo "[5/5] Registering property & submitting sitemap (${SITEMAP_URL}) in Google Search Console..."
curl -sS -X PUT \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "X-Goog-User-Project: ${GCP_PROJECT}" \
    -H "Content-Length: 0" \
    "https://searchconsole.googleapis.com/webmasters/v3/sites/${ENCODED_SITE_URL}"

curl -sS -X PUT \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "X-Goog-User-Project: ${GCP_PROJECT}" \
    -H "Content-Length: 0" \
    "https://searchconsole.googleapis.com/webmasters/v3/sites/${ENCODED_SITE_URL}/sitemaps/${ENCODED_SITEMAP_URL}"

echo "================================================================"
echo "✅ Verified & registered in Google Search Console: ${SITE_URL}"
echo "✅ Submitted Sitemap to Google                  : ${SITEMAP_URL}"
echo "================================================================"
