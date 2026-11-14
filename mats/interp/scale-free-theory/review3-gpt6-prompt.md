You are checking an addendum before its claims are reported. Working directory: /home/agent/workspaces/mats.
Read: work/scale-free-theory/PREREG-SF.md (pre-registration, hashed in work/scale-free-theory/prereg_hashes.txt before
any outcome), the code (work/scale-free-theory/sf_net.py, m3_subst.py, m4_bottleneck.py, m5_sink.py, sf_analyze.py),
the outputs (m3_subst.json, m4_bottleneck.json, m5_sink.json, sf_scores.json) and the addendum section "Addendum:
M3, M4, M5 on the existing nets" at the end of outbox/scale-free-theory-2026-10-29.md.

Check, briefly:
1. Does the code implement what PREREG-SF.md registered (nets, episodes, variants, metrics, thresholds)? Any bug that
   could change an outcome (e.g. the misaligned permutation, the two-way fit, the KL direction, the bottleneck
   placement, the noisy-net attention read-out)?
2. Are the reported counts and numbers in the addendum the ones in the JSON files and sf_scores.json?
3. Is each interpretation supported and not overstated? In particular: what M3's ladder and layer roles do and do not
   show about routing vs weighting; what the M4 bottleneck k means (a shared per-token rank, not a per-source message);
   what M5's correlations and slopes mean for the "prior as sink" identity; any post hoc result presented as
   pre-registered.
4. Are the statuses (MEASURED, post hoc, CONJECTURE) right?
Output: numbered findings with severity and the fix. Do not edit files. Be concise.
