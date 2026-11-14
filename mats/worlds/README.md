# Worlds

Synthetic prior worlds with an exact Bayes oracle, the small transformers trained on them, and the run outputs. Copied from `/home/agent/workspaces/mats/work/<folder>/` on General on 2026-10-28 at 20:28 UTC; `world-fa/` again at 22:59 UTC and on 2026-10-29 at 00:54 UTC. The code is unchanged. Checkpoints (`*.pt`) and arrays (`*.npz`, `*.pkl`) stayed on General and are listed in `mats/ARTIFACTS.tsv`. All results are exploratory (`docs/PROTOCOL.md` section 10) and mostly unreviewed; the reports named below say which parts had a review.

| Folder | What it is | Report (in `mats/reports/`) |
|---|---|---|
| `pfn-dedup/` | Set-style PFN on a source world with copied rows, depths 1 to 8, with and without group keys. Does attention deduplicate? | `pfn-dedup-depth-2026-10-26.md`, `pfn-dedup-review-2-2026-10-28.md` |
| `tiny-pfn/` | A grid of small PFNs on a three-source reliability world, searching for the smallest one within 0.02 nats of Bayes, plus the plateau diagnostic | `tiny-pfn-2026-10-27.md` |
| `worlds/` | World-agnostic harness (`harness.py`) and three worlds from `reports/world-critique/DESIGN-v1.md`: W1 hierarchical junta (`junta.py`), W2 market (`world2_market.py`, fast generator `w2fast.py`), W3 chains (`world3_chains.py`, `looped.py`) | `world1-junta-pilot-2026-10-28.md`, `world2-market-pilot-2026-10-28.md`, `world3-chains-pilot-2026-10-28.md` |
| `world-fa/` | World FA (forecast aggregation over a hidden correlated information structure): spec, preregistration, generator and exact Bayes (`world_fa.py`, `fa_core.py`, brute-force check `fa_brute.py`), training (`train_fa.py`), audit sweeps, design critiques (`r1-*.md`), the GPU pilot, binding and decomposition runs, and the A7 pipeline that was written but never run (`run_a7_fa.sh`, `fa_tests.py`). Parked | `world-fa-2026-10-28.md`, `world-fa-2026-10-29.md` |
| `w2-mech/` | Mechanism study of the W2b models (crux C4): locate, cut and patch | `w2-market-mechanism-2026-10-28.md` and its Codex review |

## Where each table comes from

Every report has a "Files" section (or names its files inline) that ties its tables to scripts and run folders. In short:

| Results | Written by | Per-seed logs and JSON |
|---|---|---|
| `pfn-dedup/runs_final/{summary,tables,split_tables}.md` and the analysis JSON | `run_all.sh` (train and eval, then `summarize.py`), `mech2.py`, `mech3.py`, `mech4.py`, `analysis2.py`, `make_tables.py`, `split.py`, `split_copy.py`, `split_tables.py`, `split_indist.py` | `runs_1026c/`, `runs_efg/`, `runs_l8/` (`L<depth>_k<keys>_s<seed>.log`, `.eval.log`, `<run>/eval.json`); `runs_final/` links to them |
| Track A2 (`pfn-dedup/runs_a2/`, control `runs_a2ctl/`) | `a2_tables.py`, `split_indist.py` | `runs_a2/L*_k0_s*.log` |
| `tiny-pfn/runs/summary*.md`, `runs/plateau.md` | `run_all.sh`, `run_followup.sh`, `run_ws.sh` (CPU on General), then `analyze.py`, `analyze2.py`, `analyze3.py`, `plateau.py` | `runs/<config>_s<seed>/` (`meta.json`, training log) |
| W1: `worlds/runs_w1pilot/tables.md`, `runs_w1b/tables_100k.md`, `runs_w1b2/tables_200k.md` | `run_pilot.sh`, `run_w1b.sh`, `run_w1b2.sh` (via `launch*.sh`), then `report_w1.py`, `report_w1b.py`, `diag_junta.py` | `runs_w1*/<arm>_s<seed>.log`, `.eval.log`, `.diag.log`; `<run>/eval/summary.json` |
| W2, W2b, W2c: `worlds/runs_w2*/**/summary.md`, `curves.md`, `cue.md` | `run_w2.sh`, `run_w2b.sh`, `run_w2c.sh`, then `w2_summarize.py`, `w2b_curves.py`, `w2c_cue.py` | `runs_w2_1028a/`, `runs_w2b_1028/{a,b}/main/`, `runs_w2c_1028/{a,b}/main/` |
| W3: `worlds/results-w3/*/summary.md` | `run_w3.sh`, `run_w3_screen.sh` (the dense-supervision arm uses `w3_dense.py`), then `w3_summarize.py` | `results-w3/eq-pilot/`, `plain-xor-s0/`, `screens/` |
| World FA pilot, binding arms and decomposition; `world-fa/runs/bind-all/SUMMARY.md` | `run_pilot_fa.sh`, `run_binding_fa.sh`, `run_decomp_fa.sh` (via `launch_fa.sh`), then `diag_fa.py`; `summarize_bind.py runs/bind-all --md runs/bind-all/SUMMARY.md`; `pilot_gap.py`; audit sweeps by `audit_fa.py`, `sweep_fa.py`, `gate_scan.py` | `runs/fa-pilot-1028b/`, `runs/fa-bind-1028d/`, `runs/fa-bind-1028e/`, `runs/fa-decomp-1028f/` (`<arm>_s<seed>.log`, `.eval.log`, `.diag.log`, `<arm>/diag.json`, `meta.json`); `runs/bind-all/` links the seven binding arms; `audit/*.jsonl` |
| W2 mechanism (`w2-mech/out/*.json`) | `prep.py`, `locate.py`, `c4.py`, `intervene.py`, `intervene2.py` | `w2-mech/out/*.log` |

Preregistrations and their hashes sit next to the code: `pfn-dedup/PREREG.md` (`code_sha256.txt`, `a2_code_sha256.txt`), `tiny-pfn/PREREG.md`, `worlds/PREREG-W1.md`, `PREREG-W2.md` (with `PREREG-W2*.sha256`), `PREREG-W3.md`, `world-fa/PREREG-FA.md`, `w2-mech/PREREG.md`. `worlds/PREREG.md` is the shared file that the W1, W2 and W3 lanes wrote into; the per-world files are the copies that were hashed.

## State at copy time

- World FA: parked on 2026-10-29 at 00:35 UTC, and `world-fa/` was synced again at 00:54 UTC, so this is its final state for now. All seven binding arms are here (`runs/fa-bind-1028d/` and the dense arm rerun alone after an out-of-memory error, `runs/fa-bind-1028e/`), joined in `runs/bind-all/` by relative links, and so is the decomposition run (`runs/fa-decomp-1028f/`, PREREG amendments A5 and A6). PREREG amendment A7 and its code are written but were never run; only a local smoke test ran, and it was stopped at the park.
- 44 symlinks in `worlds/runs_w2b_1028/` and `runs_w2c_1028/` pointed to evaluation sets on deleted VMs (`/root/soup/w2b/es/...`) and were dropped. Symlinks inside a copied folder were kept as relative links.
- `w2-mech/` reads the W2b checkpoints from `/home/agent/workspaces/mats/work/worlds` (hard-coded), and several launch scripts `cd` into General paths. Those scripts run as they are only on General.

## Tests

Run from each folder with torch, numpy, scipy and pytest (on General: `/home/agent/.venvs/aggext/bin/python -m pytest -q ...`). Results of a run from this tree on General (CPU) on 2026-10-28:

| Folder | Command | Result |
|---|---|---|
| `pfn-dedup/` | `pytest -q test_world.py` | 4 passed (3 s) |
| `tiny-pfn/` | `pytest -q test_world.py` | 9 passed (4 s) |
| `worlds/` | `pytest -q test_junta.py test_harness.py test_looped.py test_w2fast.py test_world2_market.py test_world3_chains.py` | 77 passed (160 s) |
| `world-fa/` | `pytest -q test_fa.py` | 60 passed, 2 skipped by design (signed enumeration only up to N = 3) (17 min; the VM runs used the faster `-k` subsets in the `run_*_fa.sh` scripts); rerun after the 2026-10-29 00:54 UTC sync |
| `w2-mech/` | no tests | |

No test needs a GPU. The training runs do: each `launch*.sh` starts one L4 VM through `mats/infra/run_on_l4.sh`.

Rerunning `pfn-dedup/summarize.py runs_final` and `worlds/report_w1.py runs_w1pilot` on the committed JSON (in a scratch copy) reproduces `pfn-dedup/runs_final/summary.md` and `worlds/runs_w1pilot/tables.md` exactly, and `world-fa/summarize_bind.py runs/bind-all` reproduces `world-fa/runs/bind-all/SUMMARY.md`.
