# Atom energy: bounded weights, and model + data vs data alone (2026-10-25/26)

Follow-up to `atom-energy-2026-10-25.md` and its adversarial review. CPU only, exact enumeration, n = 3 and 4 binary
variables. Code and raw output: `work/gls-energy/atom2/` (`lib.py`, `partA.py`, `partB.py`, `sumA.py`, `sumB.py`,
`partA.json`, `partB.json`, `partA_summary.txt`, `partB_summary.txt`); the original `atom/atom_energy.py` is reused
unchanged. Not reviewed.

## Part A: bounded weights

Every estimated-weight arm now:
- clamps its variance to the learned-C training-target range (log variance in [-6.91, 6.12]);
- sets w = 1/v, normalises to mean 1, clips to [0.05, 20];
- shrinks toward uniform with lambda chosen on training worlds (seeds 1000-1299).

The chosen lambdas are 0.75 for self-estimated, and 1.0 for learned C, 100 samples and 1000 samples. The per-type
learned arm is dropped, so no arm sees the test world's label.

Test worlds use fresh seeds (7000+, and 7300/7600/7900/8200 for the variants), 200 per cell. There are 90 cells per
variant (60 for the two correlated variants, gauss and tempered only).

Ratios are taken against the uniform-weight atom energy. "Geo" is the geometric mean over cells of the ratio of cell
means; "grand" is the ratio of grand means.

| variant | arm | geo | grand | worst cell | cells worse (2 se) | worlds with KL > 1 |
|---|---|---|---|---|---|---|
| base | uniform | 1 | 1 | | | 3 |
| base | oracle (reference, uses truth) | 0.129 | 0.228 | 1.45 | 6 | 0 |
| base | learned C, unbounded (old recipe) | 0.724 | 1.667 | **74.9** | 11 | 14 |
| base | learned C, bounded | 0.682 | 0.698 | 1.36 | 11 | 11 |
| base | self-estimated, bounded | 0.772 | 0.778 | 1.42 | 27 | 0 |
| base | IRLS, bounded | 0.853 | 0.914 | 5.28 | 34 | 11 |
| base | data 100, bounded | 0.673 | 0.524 | 2.72 | 18 | 0 |
| base | data 1000, bounded | 0.328 | 0.286 | 1.00 | 0 | 0 |
| base | empirical joint, 1000 samples | 0.294 | 0.210 | 2.94 | 10 | 0 |
| Dirichlet 0.3 | learned C, bounded | 0.744 | 0.825 | 1.93 | 12 | 10 |
| Dirichlet 0.3 | data 1000, bounded | 0.392 | 0.358 | 1.12 | 1 | 0 |
| Dirichlet 3.0 | learned C, bounded | 0.629 | 0.545 | 1.18 | 11 | 9 |
| Dirichlet 3.0 | data 1000, bounded | 0.297 | 0.252 | 0.95 | 0 | 0 |
| correlated within atom | learned C, bounded | 0.803 | 0.835 | 1.31 | 13 | 0 |
| correlated per variable | learned C, bounded | 0.860 | 0.876 | 1.34 | 8 | 0 |

The other arms in the variants follow the same pattern; see `partA_summary.txt`.

## Part B: model + data

The model+data estimate is mu = argmin -sum cnt log mu + t·F_w(mu), i.e. N·KL(p_hat || mu) + t·F_w + const.
- Data: the first N of one 1000-sample draw per world (nested).
- Cells: 60 = patterns × noises × sigma pairs {0.1/0.5, 0.05/1.0} × n ∈ {3, 4}, 200 fresh worlds each.
- The trust t comes either from 4-fold held-out likelihood on the N samples ("cv", no truth) or, fixed per N, from
  training worlds ("tr": t = 30 at N = 10, 100 otherwise).

Geometric mean over cells of (arm KL / empirical KL); cells worse / better than empirical (paired 2 se); worlds with
KL > 1:

| arm | N=10 | N=30 | N=100 | N=300 | N=1000 | cells worse at N=1000 | KL>1 worlds at N=10 |
|---|---|---|---|---|---|---|---|
| empirical (+0.5 smoothing) | 1 | 1 | 1 | 1 | 1 | | 2 |
| model only, uniform weights | 0.080 | 0.157 | 0.427 | 1.18 | 3.84 | 49 | 3 |
| model only, bounded learned C | 0.049 | 0.097 | 0.264 | 0.729 | 2.37 | 40 | 10 |
| model + data, learned C, t from training worlds | **0.051** | **0.086** | **0.200** | **0.421** | **0.796** | 4 | 6 |
| model + data, learned C, t by CV | 0.568 | 0.414 | 0.445 | 0.574 | 0.827 | 12 | 457 |
| model + data, uniform weights, t by CV | 0.537 | 0.446 | 0.580 | 0.857 | 1.18 | 33 | 436 |
| model + data, data-calibrated weights, t by CV | 1.17 | 0.766 | 0.670 | 0.689 | 0.727 | 6 | 777 |
| model + data, oracle weights (truth), t by CV | 0.626 | 0.384 | 0.305 | 0.274 | 0.330 | 4 | 526 |

For "model + data, learned C, t from training worlds", the cells worse than empirical are 0/60 at N ≤ 100, 1/60 at
N = 300 and 4/60 at N = 1000. Worst cell at N = 1000: 2.11× (n4|fullstiff|tempered); the four losing cells are three
tempered (too-flat) cells and one t3 cell.

## Findings
1. Bounding fixes the mean-level failure of learned C: on the same seeds the worst cell drops from 74.9× to 1.36× and
   no aggregate gets worse. It does not remove rare per-world blow-ups under heavy-tailed noise: learned C and IRLS
   each still have about 10 worlds with KL > 1, against 3 for uniform weights. These are worlds where uniform weights
   are fine, so the estimated weights trust an atom that happens to carry a heavy-tailed error.
2. With bounding, 1000-sample data calibration is never worse than uniform weights in any cell of any variant
   (worst cell 1.00-1.12×). On its own it is still only comparable to the plain empirical joint of the same samples.
3. Model + data beats data alone at small N by a wide margin: about 20× lower KL at N = 10 and 5× at N = 100, with
   trust fixed from training worlds. It stays ahead on average up to N = 1000 (0.80×), and it beats the model alone
   from N = 30 on.
4. It does not fully converge to the data at large N with a fixed trust: at N = 1000 it is worse than empirical in
   4/60 cells (up to 2.1×), almost all with too-flat (biased) atoms. Trust that shrinks as N grows would be needed.
5. Choosing the trust by cross-validation on the same small sample is unreliable. The typical choice is fine (median
   t = 100), but at N = 10-30 it sometimes picks near-zero trust in peaked worlds, giving hundreds of worlds with KL > 1
   (436-777 at N = 10). This holds even with oracle weights, so the cause is the trust selection, not the weights.
6. At large N, better weights matter. With oracle weights plus data the ratio is 0.33 at N = 1000, against 0.80 for
   learned C and 0.73 for data-calibrated weights. Learning C is still the main gap; the energy and the data
   combination work.

## Caveats
- One generator family (logit-noise atoms on a random Dirichlet joint); n ≤ 4; learned C is tested on worlds like its
  training worlds (fresh seeds, not other families).
- The oracle arms use the truth, and in tempered worlds they read the true per-context logit (as the earlier review
  noted). They are references only.
- The trust t "from training worlds" uses the truth on training worlds only; it is a fixed number per N (30 or 100),
  not per test world.
- The data-calibrated + data arm uses the same samples for calibration and likelihood in the final fit; its CV
  re-calibrates inside each fold, so the trust selection is honest, but the final estimate double-uses the data.
- 4-fold CV at N = 10 holds out 2-3 samples per fold; a one-standard-error rule toward higher trust might fix
  finding 5 but was not tested, to avoid tuning on these test results.
- No independent review of this run yet.
