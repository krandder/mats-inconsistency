#!/usr/bin/env bash
# usage: paused.sh CMD...   Pause (SIGSTOP) the newest running phase-R job while CMD runs, so that at most 4 compute
# processes run at any time; resume it (SIGCONT) afterwards, also on error.
cd "$(dirname "$0")"
pid=$(pgrep -n -f '^/home/agent/.venvs/aggext/bin/python (truth_fz|chain_fz|eps_lb|score_cb|gc_tail|surr)\.py')
[ -n "$pid" ] && kill -STOP "$pid"
trap '[ -n "$pid" ] && kill -CONT "$pid"' EXIT
OMP_NUM_THREADS=1 "$@"
