# Consistency research tasks

The project is tracked under Beads epic **`hub-t8k9c`**. [Open the live task graph](https://consistency.futarchy.ai/plan.html) for statuses refreshed every five minutes.

This repository includes a [project-only task snapshot](../project/beads.json), so every repository collaborator can read the task descriptions and dependencies. No access to the private Hub or fleet dashboard is needed.

To propose work or a status change, [open a repository issue](https://github.com/futarchy-fi/consistency-ft/issues/new) and put the Beads ID in its title, or reference that ID in a pull request. A project maintainer applies accepted changes to the canonical tracker; GitHub issues do not sync automatically.

Snapshot exported: 2026-09-13T14:49:33.953231+00:00. The statuses below are recorded task states, not a fresh verification of whether an old operational blocker still exists.

| Beads ID | Task | Status | Prerequisites |
|---|---|---|---|
| `hub-t8k9c` | RLEF × Forecasting — consistency-ft research program | open | — |
| `hub-t8k9c.1` | Fireworks account unblock + backfill (minimax-m2p7, inkling, muse retry, reasoning probe) | open | — |
| `hub-t8k9c.2` | Cutoff audit: collect advertised knowledge cutoffs for all benchmarked models | closed | — |
| `hub-t8k9c.3` | Dashboard: cutoff-verification section | open | `hub-t8k9c.9` |
| `hub-t8k9c.4` | Define temporally-clean Brier windows from verified cutoffs | open | `hub-t8k9c.9` |
| `hub-t8k9c.5` | Implement Exploit(c,S) arbitrage LP metric + tests | closed | — |
| `hub-t8k9c.6` | Episode builder: tuples → training episodes with constraint set S | closed | — |
| `hub-t8k9c.7` | GATE 2: write up or extend (2028 benchmark, larger models) | open | `hub-t8k9c.17` |
| `hub-t8k9c.8` | Cutoff audit: build high-salience post-cutoff probe set | closed | `hub-t8k9c.2` |
| `hub-t8k9c.9` | Cutoff audit: contamination runner + verified-cutoff table | in_progress | `hub-t8k9c.8`, `hub-t8k9c.1` |
| `hub-t8k9c.10` | Reasoning-mode sensitivity sweep (none/low vs medium/high) | in_progress | `hub-t8k9c.1` |
| `hub-t8k9c.11` | Base-belief snapshots + KL anchor implementation (arm A2) | in_progress | `hub-t8k9c.1` |
| `hub-t8k9c.12` | Brier anchor implementation (arm A1) | open | `hub-t8k9c.4` |
| `hub-t8k9c.13` | Training platform decision: managed RFT vs self-host; base model choice | in_progress | `hub-t8k9c.1`, `hub-t8k9c.10` |
| `hub-t8k9c.14` | Eval battery automation | open | `hub-t8k9c.4`, `hub-t8k9c.5` |
| `hub-t8k9c.15` | Pilot run: control + A2 (KL) arms, ~5k episodes, 1 seed | open | `hub-t8k9c.6`, `hub-t8k9c.13`, `hub-t8k9c.5`, `hub-t8k9c.11` |
| `hub-t8k9c.16` | GATE 1: pilot review with Jessica | open | `hub-t8k9c.15`, `hub-t8k9c.14`, `hub-t8k9c.9` |
| `hub-t8k9c.17` | Full run: all arms, 2-3 seeds | open | `hub-t8k9c.12`, `hub-t8k9c.16` |
| `hub-t8k9c.18` | Add GLM-5.3, Grok, Astra and all Luna reasoning levels to consistency ranking | closed | — |

Maintainers refresh this export with `python3 scripts/export_project_beads.py` from an authorized fleet environment, review the diff, and commit both generated files.
