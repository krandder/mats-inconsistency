Model: GPT-6 (Codex)

Read-only review, checked 2026-10-31; no Python or inference run. References to draft lines mean update75_draft.md.

1. **High: cancellation is not established.** In `results/e66/ctrl.json:e37_ctrl_s0`, sign_pc is 0.0203/0.0466 after single removals and **0.0516 after both**; sign_kl rises to 0.3268. The smaller share therefore coexists with greater absolute sign cost. P257 passes its registered ratio test, but draft:16,51 overinterpret it. Fix: report numerator and denominator alongside shares; call cancellation compatible with the intervention, not confirmed.

2. **High: M2's algebra and necessity claim fail.** Draft:50,53 omits the baseline mean and softmax normalization: for fixed base weights, the small-beta expansion is E_w[z] + beta Var_w(z), z=r/n, not beta sum(z²). Furthermore n need not be even: ||c+ur||² includes 2r c·u. Opposite score coefficients do not establish cancellation through unequal value/output maps. Fix: give the conditional expansion, acknowledge residual sign routes, and remove “symmetry only when two heads pair up.”

3. **High: necessity is overstated as computational attribution.** Draft:3–7,15,18,20 turns large off-distribution damage into “most” reliability computation and norm-route sufficiency. Own full-removal KL factors are 38–55 (`results/e66_tables.md`); near-zero rom alone cannot establish successful correction. Ctrl B's rom changes little. Fix: state sensitivity to this surgery, restrict departure increases to A, and retain norm-route attribution as a hypothesis. Say “KL from Bayes,” not “times further from the correct answer.”

4. **Medium: verification scope and wording.** `e66_plumb.py:39,59` checks two exploratory nets and one untrained own net, using duplicated logit code. Fix draft:25's “each forward” accordingly; block 2's operations remain unchanged, but its activations change. Hash timestamp is 04:46:21 (`prereg_hashes.txt:805`), not 04:45.

Survive: static inspection supports the registered surgery, preserved key bias/LN beta/eps and own mask (`e66_run.py:43–77`); conditions, family/rom evaluators, P256–P259 scoring and reported tables agree. Need rewording: cancellation, reliability attribution, departure and plain-language claims. Fail: the stated quadratic identity and exclusive two-head mechanism. Exact forward equality on every registered checkpoint remains unverified.
