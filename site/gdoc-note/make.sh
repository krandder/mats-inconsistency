#!/usr/bin/env bash
# Build /tmp/rdn/gdoc.html (Google-Docs-ready research direction note) and, with --push, import it into the
# existing Google Doc in place on deploy-host and verify the id and the image count.
#   RDN_SRC=<note.md> /tmp/rdn/make.sh [--push]
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
mkdir -p /tmp/rdn/img && cd /tmp/rdn
DOC_ID=[removed]
NAME='Inconsistency Is All You Need - research direction note (MATS, 2026-10-16)'
B=/home/agent/workspaces/mats/work/direction-page/direction/index.html
if [[ -f $B ]]; then timeout 200 python3 "$D/render_figs.py" | head -4; else rm -f figs.json; python3 "$D/render_mermaid.py"; fi
python3 "$D/extract.py"
timeout 300 python3 "$D/render_math.py" | grep -E 'errors|TOO WIDE' || true
python3 "$D/assemble.py"
[[ "${1:-}" == "--push" ]] || exit 0

ssh deploy-host mkdir -p /tmp/rdn-gdoc && scp -q gdoc.html deploy-host:/tmp/rdn-gdoc/gdoc.html
ssh deploy-host "cd /tmp/rdn-gdoc && N='$NAME' && \
  rclone copyto gdoc.html \"gdrive-futarchy:\$N.html\" --drive-import-formats html --drive-export-formats html -v 2>&1 | grep -v NOTICE; \
  rclone lsjson gdrive-futarchy: --files-only --include \"/\$N.*\" 2>/dev/null | grep -o '\"ID\":\"[^\"]*\"\|\"ModTime\":\"[^\"]*\"'; \
  rclone copyto \"gdrive-futarchy:\$N.html\" check.html --drive-export-formats html 2>/dev/null; \
  echo \"exported <img> count: \$(grep -o '<img' check.html | wc -l)\""
