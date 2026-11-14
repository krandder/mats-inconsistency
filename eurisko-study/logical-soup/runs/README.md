# runs/

One directory per run: `runs/<run-id>/`, where run-id is `<date>-<name>-s<seed>`
(e.g. `2026-09-28-baseline-k14-s1`), plus `runs/smoke/` rewritten by `make check`.

- `config.json`: every knob (scripts/run.sh writes it; older runs have `config.txt`).
- `metrics.csv`: one row per epoch, header first (soup.bend `header()`, columns in docs/metrics.md).
- `timing.csv`: per chunk wall seconds, instructions, instructions/s.
- `soup.dump`: state after the last finished chunk (resume point; not in git).
- `plots/`: `make plot RUN=<run-id>` (scripts/plot.py) writes one PNG per metric group plus overview.png.
- `notes.md` (optional): what the run showed; the claim lines go to RESULTS.md.

Large raw dumps (> 50 MB) stay out of git (see .gitignore).
