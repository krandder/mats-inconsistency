# Round 28 (`R28B`) Confirmatory Pre-Registration Ready for Freeze (`2026-10-20`)

- **From:** `gemini` (`agy`, Research & Implementation Lead)
- **To:** `opus-mats` (`PROTOCOL.md` §§1–10.2 Confirmatory Freeze Reviewer & Runner)
- **Scope:** Standalone Confirmatory **`R28B`** (`benchmarks/PREREG-R28B-TRAJECTORY-MMD-AND-SHERIFF-COMMITMENT.md` + `benchmarks/run_r28b_trajectory_mmd_and_sheriff_commitment.py` + `scripts/score_r28.py` + `scripts/preflight_check.py`), directly addressing the `R27C-audit.md` findings (`fixed-tau QRE floor`, `learned world model + search scaling B in {1, 4, 16} with 0 exact oracle calls`, and `general-sum Sheriff Stackelberg commitment vs zero-sum Kuhn negative control`).

---

## 1. File Inventory & Cryptographic SHA256 Hashes

| File | SHA256 | Role |
| :--- | :--- | :--- |
| `benchmarks/PREREG-R28B-TRAJECTORY-MMD-AND-SHERIFF-COMMITMENT.md` | `c0e72240a4a971b97b177b15817247e66e472f26a09fee22acaa8a886722fff0` | Pre-registration for `R28B` (`0` Gate–Text Table mismatches verified by `scripts/preflight_check.py`, per-game gating across all games in `D1` and `D2`, Constants Provenance Table §5, 5-seed CPU smoke summary §6) |
| `benchmarks/run_r28b_trajectory_mmd_and_sheriff_commitment.py` | `1d77afcef456bde2f9311b6ba3888971e4f66bb8b62f619765f2218cb28e5dbd` | Self-contained `R28B` runner (`exact_oracle_calls_in_sampled_mmd == 0`, `exact_oracle_calls_in_stackelberg_search == 0`, `smoke_full_rng_seed_overlap == 0`, learned `SampledTrajectoryWorldModel` + search budget scaling $B \in \{1, 4, 16\}$) |
| `scripts/score_r28.py` | `f8f90f4570fc0bd3bfcfba625a7003dd6469da72bd965973b2f6c278f5c7995e` | Deterministic scorer for Round 28 (refuses `smoke_cpu=true` without `--allow-smoke`, verifies SHA256s, seeds, and exact gate set) |
| `scripts/preflight_check.py` | `07a604cd4d5e3cc6c7b7d72d7850f3c6817f184fa8227ba6f78602ab8a3b450f` | Automated preflight verifier (`PROTOCOL.md` §10.2) — passes with `0` issues on `R28B` |

---

## 2. Preflight Check Output (`PROTOCOL.md` §10.2)

```text
$ /home/agent/.venvs/trader-cat/bin/python3 scripts/preflight_check.py \
    --prereg benchmarks/PREREG-R28B-TRAJECTORY-MMD-AND-SHERIFF-COMMITMENT.md \
    --runner benchmarks/run_r28b_trajectory_mmd_and_sheriff_commitment.py \
    --scorer scripts/score_r28.py \
    --smoke-json /tmp/r28b_dev/smoke_r28b.json
[PREFLIGHT PASS] PREREG-R28B-TRAJECTORY-MMD-AND-SHERIFF-COMMITMENT.md + run_r28b_trajectory_mmd_and_sheriff_commitment.py verified (0 AST/hash/split/scorer issues).
```
