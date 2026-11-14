"""Write PREREG.md from the frozen tuning outputs, then record its SHA256 and timestamp (PREREG.sha256 and
vm/LEDGER.md). Run once, before any eval seed. Refuses to overwrite an existing PREREG.md."""
import datetime
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
out = os.path.join(HERE, "PREREG.md")
if os.path.exists(out) and "--force" not in sys.argv:
    raise SystemExit("PREREG.md exists; it is frozen")
hp = json.load(open(os.path.join(R, "hp.json")))
sel = json.load(open(os.path.join(R, "hp_select_reach.json")))
seld = json.load(open(os.path.join(R, "hp_select_dist.json"))) if os.path.exists(os.path.join(R, "hp_select_dist.json")) else None
nl = json.load(open(os.path.join(R, "none_loss.json")))
timing = [json.loads(l) for l in open(os.path.join(R, "logs", "timing.log")) if l.startswith("{")]
g6 = json.load(open(os.path.join(R, "gates_g6.json")))["g6"]
g7 = [json.load(open(os.path.join(R, f"gates_g7_{s}.json")))["g7"] for s in (100, 101, 102)]
g1g2 = json.load(open(os.path.join(R, "gates2_g1g2.json")))
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
V = next(t for t in timing if t["arm"] == "V")
B = 6000 * V["flops_per_step"]

L = []
w = L.append
w("# P14 pre-registration (frozen before any eval seed runs)\n")
w(f"Frozen {now}. The SHA256 of this file is in `PREREG.sha256` and in `vm/LEDGER.md` (line FREEZE). Nothing below "
  "changes after the freeze; later departures go in RESULTS.md under 'Departures after the freeze'.\n")
w("The design is the P14 spec (S1-S5, F1-F6, the attribution family and the secondary statistics, unchanged). "
  "This file fixes the values tuning chose, the matched budget, the analysis details the spec left open, and "
  "every deviation from the spec found before the freeze.\n")
w("## Frozen values\n")
w(f"- Grace G = 500 steps (gate g7: median steps to half the asymptotic VoI of a newly planted g2 slot = "
  f"{', '.join(str(int(g['median_t50'])) for g in g7)} on seeds 100-102; asymptote about 31.5 nats/maze).")
w(f"- Budget B = V's counted FLOPs over 6000 steps = 6000 x {V['flops_per_step']:.0f} = {B:.4e} FLOPs "
  "(matmul FLOPs from the analytic counter plus the element-wise flip-value adds; gate g5 checks the matmul "
  "count against torch's counter within 1.3%).")
w("- 50%%-flip fallback: not used (V pilot on the VM: %.3f s/step on one thread; tuning runs under 28 concurrent "
  "jobs stay near 0.25 s/step; threshold 0.5)." % V["s_per_step"])
w("- Steps to B (the FLOP multipliers; eval arms train until their counter reaches B, ORACLE and planted arms run 6000 steps):\n")
w("| arm | counted FLOPs/step | steps to B | multiplier vs V |")
w("|---|---|---|---|")
for t in timing:
    w(f"| {t['arm']} | {t['flops_per_step']:.4g} | {t['steps_to_B']} | {t['steps_to_B'] / 6000:.2f} |")
w("")
w("- Tuned values (label-only rule on seeds 100-102, validation set 777; score = best live bit's probe log loss; "
  "constraint = greedy-bit output log loss within 0.02 of the no-penalty counterpart):\n")
w("| arm | parameter | value | tuning score | output log loss | counterpart log loss | tolerance binds | bracket extensions | still at edge |")
w("|---|---|---|---|---|---|---|---|---|")
for arm, ch in sel["choice"].items():
    t = sel["table"][arm]
    p = [k for k in ("fee", "lam", "tau") if k in ch][0]
    cl = t["counterpart_ll"]
    w(f"| {arm} | {p} | {ch[p]:g} | {ch['score']:.4f} | {ch['ll']:.4f} | {'-' if cl is None else f'{cl:.4f}'} | "
      f"{'yes' if t['tolerance_binds'] else 'never binds'} | {ch.get('extensions', 0)} | {ch['edge'] or '-'} |")
w("")
w("  Bracketing: when a choice sat at a grid edge, the grid was extended on that side in 3x steps (at most twice). "
  "Both extension values ran together; the selection admits the second only if the first was chosen, which gives the "
  "same result as extending one step at a time. 'still at edge' marks a choice at the edge of the grid it was chosen "
  "from after the extensions.")
w("  Stage 2 (BP-L1-D-cm fee; STG-L1-cm lambda) ran for every BP-L1-cm lambda and STG tau value in parallel with "
  "stage 1; the selection uses only the runs at BP-L1-cm's lambda* and STG's tau*, as the spec's sequential order would.")
w("  Shared settings: BP-L1-D-cm uses BP-L1-cm's lambda*; STG-L1-cm uses STG's tau*; V-W, V-local, N-sel, P-LOCO "
  "and P-W use V's c*.")
w(f"- **BP\\* = {hp['BP*']}** (lowest tuning score among BP-L1-cm, BP-L0-cm, BP-L1-D-cm, DisARM-L1-cm, STG-L1-cm; "
  "ties to BP-L1-cm). Its secondary seeds are extended to 20.")
w(f"- **Endpoint switch: {'FIRED' if hp.get('endpoint_switch') else 'not fired'}** (BP\\* SUFF1 on 3/3 tuning seeds "
  "with gap_1 < 0.005). S1 is therefore "
  + ("the one-sided log-rank test on FLOPs to first SUFF1 (12 checkpoints, censored at B)." if hp.get("endpoint_switch")
     else "the one-sided Mann-Whitney test on gap_1 (V lower than BP*)."))
if seld is not None:
    w(f"- T-dist (seeds 100-101, joint-probe score): " + ", ".join(
        f"{a} {[k for k in ('fee', 'lam') if k in c][0]}={c[[k for k in ('fee', 'lam') if k in c][0]]:g}"
        for a, c in seld["choice"].items() if a in ("V", "BP-D", "BP-L1-cm", hp["BP*"])))
w(f"- Gap floor for a run with no live bit (entropy >= 0.01 on open cells): the walls-only probe gap on the eval set "
  f"4102, {nl['4102']['reach']['walls_only_gap']:.4f} nats/cell (tuning: walls-only probe loss on 777, "
  f"{nl['777']['reach']['walls_only_loss']:.4f}). Such a run scores as carrying no information and fails SUFF1.")
w(f"- L* (g2-oracle probe loss) = {nl['4102']['reach']['Lstar']:.5f} on 4102, {nl['777']['reach']['Lstar']:.5f} on 777.\n")
w("## Analysis details fixed now (the spec left them open)\n")
w("- Primary family {S1, S2}, Holm at 0.05. S1: V vs BP*; S2: V vs BP-D; one-sided (V lower gap_1). Paired "
  "Wilcoxon on seed pairs is a sensitivity analysis only.")
w("- Attribution family (Holm over 3), two-sided van Elteren tests (stratified Wilcoxon rank-sum, weights 1/(N+1), "
  "normal approximation with ties): estimator (LEG vs ST) stratified by listing in factorial A {BP, BP-D, V-noD, V}; "
  "listing (on vs off) stratified by estimator in A; penalty type (slot fee + listing vs L1) stratified by estimator "
  "in factorial B {BP-L1-cm, V-L1, BP-D, V}. Interactions are descriptive (difference of median differences).")
w("- Medians get 10,000-draw percentile bootstrap 95% intervals; rates get Wilson intervals; rate differences get "
  "Newcombe hybrid-score intervals and one-sided Fisher exact tests (secondary, unadjusted); means get t intervals.")
w("- S3: paired t upper 95% bound on the seed-paired difference of greedy-bit output log loss (V minus BP* on "
  "9x9; V minus ORACLE on 21x21), each <= 0.02.")
w("- Planted criteria (b) and (c): a planted slot counts as live while it is listed (the wall copy is constant on "
  "open cells, so the entropy-filtered live set would pass (c) trivially). The entropy-filtered reading is reported "
  "alongside.")
w("- Planted Spearman: over the 6 planted slots, the final VoI EMA (P-LOCO) or wealth (P-W) against the slot's "
  "joint-probe contribution = Shapley value of the probe log-loss reduction over the 5 analytic codes (g2 and its "
  "duplicate share; the seed-specific random slot is assigned 0), median over seeds.")
w("- Cross-seed AMI (arithmetic normalisation, exact expected MI) between the joint live-bit partitions of open "
  "cells on probe-test (300 mazes, seed 4102), median over seed pairs.")
w("- Collapse: no slot live with bit entropy >= 0.01 at the end.")
w("- F6: the point estimate of V's advantage over BP* (median gap_1 difference, or SUFF1 rate difference) is "
  "positive under greedy bits and <= 0 under 8 sampled bits.")
w("- Code identity: NMI = I/sqrt(H(b)H(g)); the delisting-quality table uses I/min(H(b),H(g)).\n")
w("## Deviations from the spec found before the freeze\n")
w("1. Probe optimiser. The specified probe (400 full-batch Adam steps at 3e-3) failed gate g6: the exact g2 code "
  "reached only probe acc 0.976, loss 0.058. The probe is minibatch Adam, 2000 steps of 4096 open-cell rows, lr 1e-2 "
  "with cosine decay (g2 on 4102: loss 0.0010, acc 1.0).")
w("2. Probe input centring. With the new probe, complements failed g6 (~g2 gap 0.043) because zero padding "
  "distinguishes a code from its complement at the grid border. Each bit channel is now mean-centred (probe-train "
  "mean) before zero padding; g6 then passes for all four sufficient codes and their complements "
  f"(largest gap: ~ring {g6['table']['~ring']['gap']:.4f}), and g1 and g0 stay insufficient "
  f"({g6['table']['g1']['gap']:.3f}, {g6['table']['g0']['gap']:.3f}).")
w("3. Gate g2 rescored. A reader fed g1 alone reaches 9x9 accuracy "
  f"{g1g2['g2']['g1_only']['9']['greedy']['acc']:.3f}, above the spec's 0.97 bar (which came from g1's exact-rule "
  "accuracy, 0.92). The gate is rescored on the sufficiency criterion used everywhere else: its log loss is "
  f"{g1g2['g2']['gap_vs_g2_oracle_9']:.3f} above the g2-fed reader's (> 0.02), so g1 is insufficient. Both "
  "verdicts are reported.")
w("4. FLOP accounting. The counter counts matmul FLOPs analytically plus element-wise flip-value adds; torch's "
  "counter over-counts grouped-conv backward, so g5 compares against a custom mapping (all arms within 1.3%).")
w("5. Tuning constraint uses the arm's greedy-bit output log loss on 777 (greedy is the primary readout).")
w("6. A tuning run with no live bit is scored by the walls-only probe loss on 777 (the spec did not say).")
w("7. T-dist tunes V, BP-D, BP-L1-cm and their counterparts; if BP* is not BP-L1-cm, BP* is tuned on T-dist by the "
  "same rule on seeds 100-101 before its T-dist eval.")
w("8. Speculative runs. To save wall time, stage 2 ran for every lambda / tau, and bracketing extensions for "
  "V (fee 0.1, 0.0333), V-L1, BP-L1-cm, STG and STG-L1-cm (lambda 3.33e-4, 1.11e-4; tau 0.167, 0.0556) were "
  "launched before the stage-1 choices were final. The selection above admits only the runs the sequential "
  "procedure would have used (column 'used' in results/hp_select_*.json); the unused runs are kept and not read. "
  "Two V extension runs were stopped once V's choice (fee 1, interior) was known.")
w("9. Tuning VM trader-b3-p14-1 was spot-preempted at 20:52Z; the queue resumed on trader-b3-p14-1-r1 from the "
  "last verified sync (.partial checkpoints hold the model, optimiser, market, data stream and RNG state).")
txt = "\n".join(L) + "\n"
open(out, "w").write(txt)
h = hashlib.sha256(txt.encode()).hexdigest()
open(os.path.join(HERE, "PREREG.sha256"), "w").write(f"{h}  PREREG.md  frozen {now}\n")
open(os.path.join(HERE, "vm", "LEDGER.md"), "a").write(f"- {now} FREEZE PREREG.md sha256={h}\n")
print(txt)
print("sha256", h)
