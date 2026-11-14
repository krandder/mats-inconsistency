"""Tables for the output-martingale backward certificate (amendment 8).
usage: python bc_summ.py -> results/bc_tables.md
"""
import json, math, os
import numpy as np

out = []
nets = []
for s in ('0.5', '0.3'):
    p = f'results/bc_small_s{s}.json'
    if os.path.exists(p): nets.append((f'small net σ {s}', json.load(open(p))))
pp = 'runs/L2_s0.05/bc.json'
pfn = json.load(open(pp)) if os.path.exists(pp) else None
RUNGS = [-6, -10, -14, -16, -20, -26, -30, -36, -40]


def pick(rows, lp):
    i = int(np.argmin([abs(r['log2p'] - lp) for r in rows])); return rows[i]


out.append('### Per-site sensitivity: search sup Λ_l vs rms over prior paths (||∇φ_l||)\n')
for nm, j in nets + ([('PFN 2L σ 0.05', pfn)] if pfn else []):
    S = j['sites']
    out.append(f"**{nm}** (σ√ΣΛ² = {j['sd_cert']:.3f}, σ√Σrms² = {j['sd_rms']:.3f}{', last-layer ||w|| = %.2f' % j['w_norm'] if 'w_norm' in j else ''})\n")
    out.append('| site | rms (prior paths) | max sampled | max after ascent | Λ used | Λ / rms |')
    out.append('|---|---|---|---|---|---|')
    for k, v in S.items():
        out.append(f"| {k} | {v['rms_prior']:.3f} | {v['max_sampled']:.3f} | {v['max_ascent']:.3f} | {v['Lambda']:.3f} | {v['Lambda'] / v['rms_prior']:.1f} |")
    out.append('')

for nm, j in nets:
    out.append(f"### {nm}: certified log2 P(A) vs truth (bits kept = certified / true rarity bits)\n")
    out.append('| true log2 p | t | certificate (bits kept) | O1: rms sensitivities (not a bound) | O2: input alone, log2 P(φ_0 > t) | plain sampling 95% bound at equal compute |')
    out.append('|---|---|---|---|---|---|')
    for lp in RUNGS:
        r = pick(j['rows'], lp)
        out.append(f"| {r['log2p']:.1f} | {r['t']:.3f} | {r['cert_log2']:.1f} ({100 * r['cert_bits_kept']:.0f}%) | {r['O1_log2']:.1f} | {r['O2_log2']:.1f} | {r['samp_log2']:.1f} |")
    out.append(f"\nCompute: {j['t_cert']:.0f} s (= {j['n_eq']:.2e} plain forward passes at the measured rate). IS: {j['n_IS']} inputs, {j['n_centres']} shift centres.\n")

if pfn:
    out.append('### PFN 2L σ 0.05: certified log2 P(A) vs truth (event: predictive mean of one noisy pass > t)\n')
    out.append(f"Output ceiling by search over reachable final query states: {pfn['ceiling']:.4f} (without the noise ball {pfn['ceiling_no_noise']:.4f}); "
               f"max over 4,096 tail-tilted real paths {pfn['max_tail_paths']:.4f}; max of φ_0 over the IS episodes {pfn['phi0_max']:.4f}.\n")
    out.append('| true log2 p | t | ceiling − t | certificate (bits kept) | O1 (not a bound) | O2: log2 P(φ_0 > t) | plain sampling at equal compute |')
    out.append('|---|---|---|---|---|---|---|')
    seen = set()
    for r in pfn['rows']:
        key = round(r['log2p'], 1)
        if key in seen: continue
        seen.add(key)
        out.append(f"| {r['log2p']:.1f} | {r['t']:.4f} | {r['ceiling_minus_t']:.3f} | {r['cert_log2']:.1f} ({100 * r['cert_bits_kept']:.0f}%) | {r['O1_log2']:.1f} | {r['O2_log2']:.1f} | {r['samp_log2']:.1f} |")
    out.append(f"\nCompute: {pfn['t_cert']:.0f} s (= {pfn['n_eq']:.2e} forward passes). IS: {pfn['n_IS']} episodes, φ_0 from {pfn['K_phi0']} noise draws each.\n")

# predictions
out.append('### Predictions (amendment 8)\n')
for nm, j in nets:
    ratio = max(v['Lambda'] / v['rms_prior'] for v in j['sites'].values())
    r40, r10 = pick(j['rows'], -40), pick(j['rows'], -10)
    beat = [r['cert_log2'] < r['samp_log2'] for r in j['rows'] if r['log2p'] <= -30]
    o1 = [abs(r['O1_log2'] - r['log2p']) <= 3 for r in j['rows'] if r['log2p'] >= -20.5]
    out.append(f"- {nm}: max Λ/rms {ratio:.1f}; bits kept at 2^-40 {100 * r40['cert_bits_kept']:.0f}%, at 2^-10 {100 * r10['cert_bits_kept']:.0f}% "
               f"(P43: ratio ≥ 2, 40–80% at 2^-40, ≥ 70% at 2^-10); beats sampling at all rungs ≤ 2^-30: {all(beat)} (P44); "
               f"O1 within 3 bits down to 2^-20: {all(o1)} (P46)")
if pfn:
    r40, r10 = pick(pfn['rows'], -40), pick(pfn['rows'], -10)
    o1 = [r['O1_log2'] - r['log2p'] for r in pfn['rows'] if r['log2p'] < -16]
    out.append(f"- PFN: ceiling − t(2^-40) = {r40['ceiling_minus_t']:.3f} (P45a: ≤ 0.06); bits kept at 2^-10 {100 * r10['cert_bits_kept']:.0f}% (≥ 70%), "
               f"at 2^-40 {100 * r40['cert_bits_kept']:.0f}% (≤ 50%); O1 over-prediction beyond 2^-16: min {min(o1):.1f} bits (P46: > 3)")
txt = '\n'.join(out)
open('results/bc_tables.md', 'w').write(txt)
print(txt)
