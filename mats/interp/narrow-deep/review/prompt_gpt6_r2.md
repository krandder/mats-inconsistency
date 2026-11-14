You are an adversarial referee (deep-learning theory, Gaussian closures / moment methods, Monte Carlo). Review ONE new
section of a research report and its code. Read-only: do not edit any file. You may run short read-only Python
checks (CPU only, at most 2 threads, under 3 minutes each, prefixed with `nice -n 19`), using
/home/agent/.venvs/arc-noisy/bin/python and the code in /home/agent/workspaces/mats/work/narrow-deep/lib.

Report: /home/agent/workspaces/mats/outbox/narrow-deep-2026-10-29.md, section 10 ("Follow-up: why K2 fails on
trained narrow nets"). Sections 1-9 were reviewed before; review only section 10 and what it relies on.
Code: work/narrow-deep/lib/k2mix.py (Gaussian-mixture closure with gate splits), lib/k2.py, lib/tnets.py,
scratch/p5_locate.py, scratch/p5_split.py, scratch/p5_rb.py, scratch/p5_tailcheck.py, and scratch/p5_rbvar.py (added
after the pre-registered runs: RB variance from 2^20 directions vs the exact MC variance). Results: runs/p5/*.json,
logs/p5_*.log. Pre-registration: PREREG.md "Part 2" (hash in prereg_hashes.txt, 2026-10-30T01:14:09Z).

The question (coordinator): why does K2 (ARC's Gaussian closure: mean + covariance through each ReLU) fail on the
trained nets (width-16 residual sorters of depth 64; width-16 RNNs, T = 64, 2nd argmax), and is the failure
concentrated (a few neurons or layers) or spread out? Then a gate-aligned fix: split at the k worst-gate units, a
mixture over their on/off states, k = 1, 2, 4, 8; refuted if no k <= 8 gives 3x over K2. Then exact radius
integration (positive homogeneity) for tails.

Check, and report every problem you find, most serious first:
1. The split closure (k2mix.py): are the truncated-moment updates (split_piece) and the forced-gate ReLU moments
   (relu_mom_forced; Stein cross-covariances Cov(h, b) = Cov(h, z) g) correct? Does k = 0 reproduce K2 exactly? Is
   there any bug that could make the splits look useless (e.g. splits applied at the wrong layer index, pieces
   dropped, weights not renormalised)? Verify with a small numerical check if you can (e.g. one split at layer 0 of a
   residual net, where the input is exactly Gaussian, against MC of the next layer's moments).
2. The diagnostics (p5_locate.py): accumulated vs one-step gate errors, the suffix error E_out(l0) (paired with the
   truth's own paths), concentration shares. Do the conclusions ("spread" for the sorters; for the RNNs gate errors
   concentrated early in few units but the output error coming from the last steps) follow from the numbers?
3. Tails: the Imhof-based closure tails in p5_split.py show numerical warnings for the RNNs; p5_tailcheck.py re-does
   them by sampling the closure's output law. Which numbers should be trusted? Is the RB estimator (p5_rb.py) valid
   for the stated events (positive homogeneity of |output|, max_i |out_i - sort(x)_i| and the RNN's wrong-answer
   margin), and is its bias check adequate, with noise sigma = 0.01 approximate?
4. FLOP accounting of the mixture and the RQMC comparisons at equal FLOPs.
5. Pre-registration: are P11-P16 scored correctly in section 10?
6. Spot-check at least 8 numbers in section 10 against logs/ and runs/p5/.
Write a numbered list of findings with file:line references, then a short verdict.
