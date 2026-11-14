#!/usr/bin/env bash
# Shrink finished L4 run dirs under /tmp/l4runs: keep runs/, gpu-run.out and GPU-LOG.md at the same paths,
# drop the ~400M repo checkout (a git worktree, restorable from git). Only dirs whose gpu-run.out ends
# with "GPU-RUN EXIT" are touched. Usage: l4_slim.sh [run-id ...]  (default: all finished)
set -uo pipefail
REPO=/home/agent/workspaces/inconsistency-search
cd /tmp/l4runs || exit 0
ids=("$@"); [ ${#ids[@]} -eq 0 ] && ids=($(ls))
for id in "${ids[@]}"; do
  d=/tmp/l4runs/$id
  [ -d "$d" ] || continue
  tail -1 "$d/gpu-run.out" 2>/dev/null | grep -q '^GPU-RUN EXIT' || continue
  [ -e "$d/.git" ] || continue            # already slim
  keep=$(mktemp -d /tmp/l4slim.XXXX)
  for f in runs gpu-run.out GPU-LOG.md; do [ -e "$d/$f" ] && cp -a "$d/$f" "$keep/"; done
  git -C "$REPO" worktree remove --force "$d" 2>/dev/null || { echo "skip $id (worktree remove failed)"; rm -r "$keep"; continue; }
  mkdir -p "$d" && cp -a "$keep/." "$d/" && rm -r "$keep"
  echo "slimmed $id"
done
git -C "$REPO" worktree prune
