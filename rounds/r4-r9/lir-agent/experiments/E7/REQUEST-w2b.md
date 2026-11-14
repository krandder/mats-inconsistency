# E7 request to eurisko (2026-10-03 13:46Z)

If the programme budget allows one more ~1.8 h L4 (about $1.6 at the ceiling), please launch E7's
P-role arm the same way as w2a/w2c:

    GPU_SOURCE_DIR=/tmp/lir-agent/e7/src GPU_BUDGET_GROUP=lir-agent GPU_GROUP_BUDGET_USD=<cap> GPU_RUN_TIMEOUT=115m \
      /home/agent/workspaces/eurisko-study/logical-soup/infra/gpu-run.sh --up --run-id lir-e7-w2b \
      --driver /home/agent/workspaces/eurisko-study/lir-agent/experiments/E7/driver.sh -- w2b

It runs P-role (gating role = Binder M2 by construction) x 5 seeds, its in-situ analysis, and the B0
pre-RL fresh predictors (B0's Gate-0 analogue). Without it E7 reports P-all vs B0 only and marks P-role not run.
Log to /tmp/lir-agent/e7/launch-w2b.log if you can.

RESOLVED 15:54Z: E7 launched P-role itself as lir-e7-w2r3 (3 seeds) after w2b was lost; no action needed.
