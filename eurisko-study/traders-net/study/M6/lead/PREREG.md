# M6 lead merge pass: pre-registration of the added runs

Written 2026-09-28 ~08:15Z by the M6 lead, before either run below.

1. **w3 completion.** w3's 24 runs died after 3/24 (no process at 08:02Z). Rerun `w3/run_all.sh` unchanged
   (it skips finished runs); w3's own PREREG and stop rule apply as written.
2. **In-harness kill controls for w2.** The plan's stop rule drops a pool that is >= (a)+1% learnable BCE in 3/3
   seeds *with no protection beyond the kill controls*. w2's prereg tested only the BCE clause and ran no kill
   control in its harness (T=60000). Added arms, same stream, trunk, seeds 1-3 and metrics as `w2/pool.py`
   (code `pool_kc.py` imports it unchanged):
   - `wd`: arm a trained with AdamW(lr 1e-3, weight_decay 0.1) (w1's WD);
   - `boost`: z = l0 + 0.1 * sum_i d_i (w1's BOOST, nu = 0.1).
   Early stopping is undefined in a single prequential pass (as w3 noted).
   Protection clause, evaluated per pool arm: the pool has "protection beyond the kill controls" iff its
   noise excess max(BCE - ln2) over {ca30, prng} is below min(wd, boost) in 3/3 paired seeds. A pool is dropped
   iff it fails the BCE clause in 3/3 seeds and has no such protection.
3. `gate_flags.base_arms_survive` = at least one of (b)-(g) (or w5's committed arms) is not dropped under the rule.
