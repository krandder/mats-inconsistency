## FU-B2-2: Do P10's learned events still collapse to constants with a non-degenerate encoder init, a rate warm-up, or both?

Status: No gain (pre-registered; all gates passed; verified, verifier agrees)

Setup: P10's M2 learned-event arm (S6 6x6 puzzles, 3 learned encoders + 5 reference slots, 20000 steps, CPU), rerun with a centred (data-dependent) encoder init, a 4000-step rate warm-up, or both; cvae arm on 10 seeds (power calculation), auction arm on 5. Primary: share of learned slots non-constant on test at the end, paired sign-flip test, Holm over 3. Gates G0 (centred init non-constant), G1 (A/A with P10), G2 (reg2 NLL) enforced by the driver. PREREG frozen at 23:11Z on 09-29.

Result: Yes, they still collapse. The collapse is not an init or rate artifact.
- Primary (the cvae arm, 10 paired seeds): the share of the 3 learned slots that are non-constant on test at the end.
  - Original init: 0/30.
  - Centred init: 0/30.
  - Rate warm-up (no rate gradient into the encoder for 4000 steps): 1/30, diff +0.033 [-0.042, 0.109].
  - Both fixes: 1/30, the same diff.
  - Holm p = 1.0 for all three fixes. The 95% upper bounds on the per-slot survival rate are 0.095 for 0/30 and 0.149 for 1/30.
- The centred init does its job: 30/30 slots are non-constant at step 0, against 9/15 constant with P10's init. The slots are then constant again within 50-100 steps (median). Slots that started non-constant went constant before any rate gradient in 10/10 both-fixes seeds, 5/5 auction_both seeds and 6/10 warm-only seeds (the other 4 were constant from step 0). So the straight-through NLL gradient alone is enough to cause the collapse.
- The one cvae survivor (seed 0, slot 0, in both warm arms) is the non-cell atom `bl_row(0,4,0)` exactly (NMI 1.00), with G 0.083 / 0.060 against a fee of 0.005.
- Secondary, the auction with both fixes (5 seeds): 0.73 [0.39, 1.08] of learned slots are non-constant at the end, with G > fee in 5/5 seeds. The original auction has 0.00. Post hoc, only slots spawned at step 14000 survive (10/15, against 0/15 for slots spawned at steps 0 and 7000). Those slots had a low cosine LR (4.1e-4, falling to 0), a trained decoder and only 6000 steps of age.
- The A/A replication of P10 (seeds 0-4, CPU instead of GPU) shows no clear difference from P10 (not an equivalence test). The NLL differences are cvae -0.020 [-0.170, 0.129], auction -0.007 [-0.102, 0.087] and reg2 +0.001 [-0.097, 0.100].
- Caveats:
  - The pre-registered "aged" auction secondary is undefined (n = 0). Every auction_both slot failed its check at 7000 and again at 14000 and was respawned, so none was old enough.
  - The step-14000 survival is post hoc and confounded, so it is not evidence that any fix works.
  - G can exceed the fee for a constant slot (a resampling artifact), so "G > fee" alone overstates usefulness.
  - NLL is underpowered (n = 5-10).
  - Untested next step: a low encoder LR from step 0, or a damped straight-through gradient, with every slot scored after a full rate phase.
- Cost $3.02 at the ceiling rate (2.51 VM-hours on a c2d-highcpu-32 at $1.20/h). The VM and disk were deleted; the instances and disks lists showed 0 items at 01:28Z on 09-30.

Files: [followups-batch2/item2-p10/RESULTS.md](followups-batch2/item2-p10/RESULTS.md), `followups-batch2/item2-p10/{PREREG.md,analyze.py,posthoc.py,results/summary.txt,results/posthoc.txt}`, `GPU-LEDGER.md` (batch-2 follow-ups section)
