#!/usr/bin/env bash
# Commit the staged mats/ tree in one worktree. Usage: [ALLOW_IGNORED_RE=regex] commit.sh <worktree> <message-file>
# Force-adds ignored *.log files; other ignored files abort the commit unless they match ALLOW_IGNORED_RE.
set -euo pipefail
WT=$1; MSG=$2
cd "$WT"
# refuse if anything outside mats/ changed
if git status --porcelain --untracked-files=all | grep -v ' mats/' | grep -q .; then echo "changes outside mats/"; git status --porcelain | grep -v ' mats/' | head; exit 1; fi
git add -A mats
# force-add ignored files under mats/ (only *.log are expected)
IGN=$(git status --porcelain --ignored --untracked-files=all -- mats | grep '^!!' | cut -c4- || true)
if [ -n "$IGN" ]; then
  if echo "$IGN" | grep -v '\.log$' | grep -vE "${ALLOW_IGNORED_RE:-^\$}" | grep -q .; then echo "unexpected ignored files:"; echo "$IGN" | grep -v '\.log$' | head; exit 1; fi
  echo "$IGN" | tr '\n' '\0' | xargs -0 git add -f --
fi
git -c core.quotepath=off status --porcelain --ignored --untracked-files=all -- mats | grep -v '^A ' | head -5 || true
git commit -q -F "$MSG"
git log -1 --stat --format='%H %an <%ae>%n%s' | tail -3
git show --stat --format='' HEAD | tail -1
