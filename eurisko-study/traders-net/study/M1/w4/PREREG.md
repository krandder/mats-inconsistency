# M1 item (d), worker 4: where the books leak, and AttnLRP-rule books

Written 2026-09-27 ~23:20Z before any run.

**Question.** The books (wealth-net `books.py`, v1) conserve revenue only up to a residual. Which
operations make it (QK bilinear, softmax, AV bilinear, LayerNorm denominators), and do
AttnLRP-rule books (Achtibat et al. 2024, https://arxiv.org/abs/2402.05602) close it and change who is rich?

**Substrate.** Wealth-net train runs, read-only weights: `../../../../wealth-net/results/weights/train-s{1,2,3}-step{8000,16000,24000,30000}.pt`
(3 seeds x 4 checkpoints). Eval = the held-out tail (last 20% positions, 2048 examples, the same
`ev_idx` as wealth-net train.py). Model/streams copied to this dir unchanged.

**Arms (rule sets; unit income = activation x (modified) gradient of R = sum_q z_q (y_q - p_q), s detached):**
1. `live`: plain gradient x input, LN and attention live.
2. `v1`: LN denominators detached (wealth-net books.py). Equals AttnLRP's LN identity rule (Prop. 3.4).
3. `cp`: LN detached and attention pattern detached (CP-LRP, Ali et al. 2022, https://arxiv.org/abs/2202.07304). Exactly conservative by construction.
4. `attnlrp`: LN detached, uniform 1/2 rule on both bilinear products (Q.K^T and A.V, Props. 3.2-3.3), softmax rule Prop. 3.1 (= gradient x input on the modified upstream gradient), ReLU identity, biases counted as units (epsilon rule with eps=0).

**Metrics.**
- Relative conservation residual at each residual-stream cut, resid_k / |R|.
- Per-op split: local defect (relevance out - relevance in) of each LN (ln1, ln2, lnf), QK, softmax and AV per layer. Closure check: sum of local defects downstream of cut k equals resid_k (must hold to < 1e-4 relative, else the split is wrong).
- Unit books per rule (heads 8, neurons 512): Spearman between rules, and Spearman against the held-out zero-ablation Delta BCE of each unit on the same eval set (a local sanity check only; the full oracle is M2's).

**Seeds.** 3 (training seeds 1-3); report mean +- sd over seeds per checkpoint.

**Stop rule.** One pass over the 12 checkpoints; no tuning of rules after seeing the ablation
correlations. If closure fails, fix the split code once and rerun all; report the failure.
