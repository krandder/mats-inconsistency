#!/usr/bin/env python3
"""X1 worker 3: render CLAIMS.md + results/claims.json from results/{static_check,repro,skeptic}.json.
Each claim gets: status (verified | partly verified | refuted | unverified), basis, checked_at, evidence, rerun.
Status rule: any static MISMATCH/ERROR -> refuted (quote). Else if seed-4 rows exist: all verified -> verified;
any refuted -> refuted; any missing -> unverified; else partly verified. Claims with no seed-4 row: 'verified'
with basis 'file re-read' (the quoted number equals its file; the run itself is not repeated).
Judgement / untested claims carry a fixed status with a reason. usage: python3 make_claims.py"""
import json, os, datetime
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, 'results')
J = lambda f: json.load(open(os.path.join(R, f)))
SC = J('static_check.json'); RP = J('repro.json'); SK = J('skeptic.json')
sc = SC['checks']; rp = {r['id']: r for r in RP['rows']}
ST = 'study/'
RE_A = 'python3 X1/w3/check_w3.py'; RE_B = 'python3 X1/w3/compare_w3.py'; RE_C = 'python3 X1/w3/skeptic_w3.py'
# id, workstream, claim text, static id prefixes, repro ids, evidence, fixed (status, reason) or None
C = [
 ('M5-1', 'M5', 'Prequential net exploitable by online MLP probe L: phi 0.092+-0.0486 (P4) / 0.118+-6.63e-4 (P16) nats/q-step; E_C log-log b 1.64 / 1.28 (linear)',
  ['M5.P4.mlp.phi', 'M5.P16.mlp.phi', 'M5.P4.mlp.b', 'M5.P16.mlp.b'], ['R-M5a', 'R-M5b', 'R-M5d.P4', 'R-M5d.P16'], 'M5/results/headline.csv; M5/w4/results/summary.csv; M5/w4/out/preq-*-win-s*.json', None),
 ('M5-2', 'M5', 'The edge is LFSR16 plus pi, and pi is a shifted-copy artefact of the stream', ['M5.P4.mlp.phi'], [],
  'M5/w4/results/family.csv; S0/SUBSTRATE-w2.md; X1/w1 pi_offsets finding (STATUS 12:14Z)',
  ('unverified', 'LFSR16+pi share re-read from family.csv; the artefact reading is not tested (no probe on the PRNG-parity family). Seed 4 draws close pi offsets and the probe pi edge drops to 0.12/0.11 from 0.19/0.17: consistent with the artefact reading, not a test of it (results/skeptic.json).')),
 ('M5-3', 'M5', 'Calibration leak: cal-online-fam invasion 8.26e-4+-1.11e-4 (P4) / 1.42e-3+-8.89e-5 (P16), fixed bins sublinear b 0.535/0.475',
  ['M5.P4.cal', 'M5.P16.cal'], [], 'M5/w1/SUMMARY-w1.csv; M5/results/headline.csv', None),
 ('M5-4', 'M5', 'Price-only probe 0.00164 / 0.00174, about 2% of probe L', ['M5.P4.mlp.P', 'M5.P16.mlp.P'], ['R-M5c.P4', 'R-M5c.P16'],
  'M5/w4/results/summary.csv', None),
 ('M5-5', 'M5', 'Pattern-2/4/8 in Kelly currency does not exploit the net: best phi -0.00151 / -0.00108 (3/3 negative); one-share 0.00761 / 0.00922',
  ['M5.P4.pat', 'M5.P16.pat'], [], 'M5/w3/results/summary.json', None),
 ('M5-6', 'M5', 'Not coherent before resolution: violations 0.477/0.545/0.536 vs 0.5 uninformed; bundle/twin arbitrage figures', ['M5.li-'], [],
  'M5/w2/results/summary.csv', None),
 ('M5-7', 'M5', 'Frozen wealth-net checkpoints get more exploitable with training; price-only 0.439 ~ probe L 0.44 at final', ['M5.WN'], [],
  'M5/results/headline.csv', None),
 ('M5-8', 'M5', 'Merge fix: four held-out BCEs are one market under different clips (0.972 raw ... 1.07 2nd half)', ['M5.clip'], [], 'M5/results/clip_bce.csv', None),
 ('M5-9', 'M5', 'Soup baseline pat4 0.128 -> -0.0148 over K/Q 1..64; arb 0.0468-0.175', ['M5.soup'], [],
  'M5/w3/results/soup_pat_agg.json; M5/w2/results/soup_arb_agg.json', None),
 ('M6-1', 'M6', 'Gate B base_arms_survive=true: 10 survive, 5 dropped, 0 undecided', ['M6.gateB', 'M6.n_'], [], 'M6/lead/M6-verdict.json', None),
 ('M6-2', 'M6', '(b)-(a) learnable BCE ties: P +0.0074+-0.0107, ME +0.0008+-0.0012', ['M6.w1.P.b_minus_a', 'M6.w1.ME.b_minus_a'], ['R-M6a', 'R-M6b.B'],
  'M6/w1/results/PAIRED-w1.csv; M6/w1/results/{P,ME}-{A,B}-s*.json', None),
 ('M6-3', 'M6', 'ME BOOST noise 0.0017+-0.0008 vs (a) 0.0108+-0.0006; ES -0.0050+-0.0004 vs (a), 3/3', ['M6.w1.ME.BOOST', 'M6.w1.ME.A', 'M6.w1.ME.ES'],
  ['R-M6b.BOOST', 'R-M6b.A', 'R-M6b.ES'], 'M6/w1/results/SUMMARY-w1.csv; PAIRED-w1.csv', None),
 ('M6-4', 'M6', 'w2 harness: cg noise 0.0019/0.0023/0.0023 < BOOST 0.0033/0.0035/0.0040 (paired 3/3); cg learnable +35/+40/+44% vs (a)',
  ['M6.w2.cg', 'M6.w2.boost.noise.s'], ['R-M6c.cg', 'R-M6c.boost', 'R-M6c.paired'], 'M6/lead/M6-verdict.json; M6/w2/results/cg-s*.json; M6/lead/results/boost-s*.json', None),
 ('M6-5', 'M6', 'Kill controls: WD learnable 0.195+-0.055, noise 0.0094+-0.0006; BOOST 0.215+-0.039, noise 0.0036+-0.0004',
  ['M6.w2.wd', 'M6.w2.boost.learn', 'M6.w2.boost.noise_mean'], [], 'M6/lead/M6-verdict.json', None),
 ('M6-6', 'M6', 'Excluding LFSR16, cg ties BOOST (0.1779/0.1339/0.1561 vs 0.1806/0.1359/0.1528)', ['M6.w2.exlfsr'], [], 'M6/lead/M6-verdict.json w2_harness_exlfsr', None),
 ('M6-7', 'M6', 'Whether a run learns LFSR16 varies by seed, and that decides the BCE clause', [], [], 'X1/w3/results/skeptic.json M6',
  ('partly verified', 'Deciding role confirmed (ex-LFSR16 tie). "Varies by seed" does not hold within arm in the w2 harness: (a) learnt LFSR16 3/3, cg 0/3 (0/4 with seed 4), cq 0/3 (Fisher one-sided p=0.05), so it is arm-systematic, not seed luck.')),
 ('M6-8', 'M6', 'w3: elog noise 0.0140+-0.0013, eexp 0.0150+-0.0004 dropped; fb01 noise 0.0031+-0.0002 but +8.1/+6.7/+7.3% BCE vs (b)', ['M6.w3'], [],
  'M6/w3/SUMMARY-w3.json; M6/lead/w3_detail.json', None),
 ('M6-9', 'M6', 'Fixed point not kept: violations vs 2-pass +0.17/+0.12/-0.08% (needs -50%); Newton 2.05+-0.08 iterations', ['M6.w4'], ['R-M6d', 'R-M6d.newton'],
  'M6/w4/VERDICT.json; M6/w4/SUMMARY.csv; M6/w4/results/{li,base}-*-s*.json', None),
 ('M6-10', 'M6', 'Committed heads Cm/Cw -0.3/-1.3/-0.9% vs U; clone corr 0.9934+-0.0015; Spearman wealth vs marginal -0.0016+-0.0221; Cs dropped (+1.8/+1.3/+2.0%)',
  ['M6.w5'], ['R-M6e.rel', 'R-M6e.corr'], 'M6/w5/results/stoprule.json; summary.csv', None),
 ('M6-11', 'M6', 'GPU spend $0.316 ceiling, audited', ['M6.gpu'], [], 'G1/COST.md row 31', None),
 ('M6-12', 'M6', 'Surviving pools cg/cq/fb01 pass only on a ~0.001-nat noise-protection margin', [], [], 'X1/w3/results/skeptic.json M6',
  ('verified', 'Margin cg-BOOST = -0.0014+-0.00025 nats (paired t=-9.8, 3/3), seed 4 -0.0022 (4/4): small but consistent. The M6 stress pass (M6/lead/M6-stress-verdict.json) later drops cg and fb01.')),
 ('M7-1', 'M7', 'Stage 0: router gradient max rel err 1.13e-15', ['M7.stage0'], [], 'M7/w1/results/stage0.json', None),
 ('M7-2', 'M7', 'dsfw vs DeepSeek u1e-3 bce_all paired: E8 -1.30% (sd 0.0009, 3/3), E64 -1.87% (sd 0.0049, 3/3), max load 0.24/0.11',
  ['M7.dsfw.E8', 'M7.dsfw.E64.rel', 'M7.dsfw.E64.sd', 'M7.dsfw.E64.wins', 'M7.dsfw.E64.maxload'], ['R-M7a', 'R-M7b'], 'M7/merge/arms.csv; M7/w2/results/all/{dsfw,ds3}-E*-s*.json', None),
 ('M7-3', 'M7', 'dsfw bce_last10k E64 -1.33% only 2/3; vs none E64 2/3', ['M7.dsfw.last10k', 'M7.dsfw.E64.wins_vs_none'], [], 'M7/merge/arms.csv', None),
 ('M7-4', 'M7', 'dsfw post-shift BCE worse vs ds3: +0.041..+0.054 (E8), +0.047..+0.092 (E64), 3/3', ['M7.dsfw.postshift'], ['R-M7a.postshift', 'R-M7b.postshift'],
  'M7/w2/results/paired.csv', None),
 ('M7-5', 'M7', 'exp3 max load E8 0.775+-0.074 vs DeepSeek 0.368+-0.054, collapsed 3/3', ['M7.exp3', 'M7.ds.E8.maxload'], [], 'M7/merge/arms.csv; M7/w3/results/summary.csv', None),
 ('M7-6', 'M7', 'Global wealth prior beta=1000: max load 1.00 at E8/E64, MI 0 bits', ['M7.global'], [], 'M7/w4/summary.csv', None),
 ('M7-7', 'M7', 'Bankruptcy+birth: E8 collapsed 3/3 (max load 0.57); E64 +4.54% (fresh +2.82%)', ['M7.bb'], [], 'M7/merge/arms.csv', None),
 ('M7-8', 'M7', 'none vs DeepSeek paired: -0.40% E8 (w1), -0.71% E64 (w4), 3/3', ['M7.none'], [], 'M7/merge/arms.csv', None),
 ('M7-9', 'M7', 'Dense FLOP-matched E8 bce_2nd_half 0.3083+-0.0236 vs DeepSeek 0.3314+-0.0197', ['M7.dense', 'M7.ds.E8.bce2h'], [], 'M7/w1/results/controls.csv', None),
 ('M7-10', 'M7', 'GPU M7 stage 1 audited $1.41 of $5 cap', ['M7.gpu'], [], 'G1/COST.md rows 32-41', None),
 ('M7-11', 'M7', 'wealth_beats_deepseek=false (lead judgement; R1/R2 pass, R3 fails for dsfw)', [], [], 'M7/merge/gate.json; X1/w3/results/skeptic.json M7',
  ('verified', 'gate.json records R1 pass at E8/E64, R2 pass, R3 fail for w2:dsfw, so the flag is a judgement across conflicting rules as stated. Skeptic: with seeds 1-3 the E64 edge was weak (paired t=-2.55, df 2, p=0.13; best of 3 dsfw variants passing 3/3 by luck p=0.33). Seed 4 (CPU) replicates the edge at both E (E8 -1.38%, E64 -3.70%; 4-seed paired t -12.8 / -3.3, p 0.001 / 0.047) and the post-shift cost (+0.056 / +0.077). So the edge is real on this substrate; the false flag rests on the cost and family-label reasons, which also replicate. Defensible, but not because the edge is noise.')),
]
out = []
for cid, ws, text, pre, rids, ev, fixed in C:
    chk = [x for x in sc if any(x['id'].startswith(p) for p in pre)]
    bad = [x for x in chk if x['status'] != 'match']
    rows = [rp[r] for r in rids if r in rp]
    if bad:
        st, basis, note = 'refuted', 'file re-read', 'quote mismatch: ' + ', '.join(x['id'] for x in bad)
    elif fixed:
        st, basis, note = fixed[0], 'analysis', fixed[1]
    elif rows:
        ss = [r['status'] for r in rows]
        st = ('refuted' if 'refuted' in ss else 'unverified' if 'unverified' in ss else 'verified' if all(s == 'verified' for s in ss) else 'partly verified')
        basis = 'file re-read + seed-4 rerun'
        note = '; '.join(f"{r['id']} {r['status']}" + ('' if r['seed4'] is None else f" (s4 {r['seed4']:.4g}, z {r['z']:+.2f}, mean {r['mean']:.4g}+-{r['sd']:.2g})") for r in rows)
    else:
        st, basis, note = 'verified', 'file re-read', ''
    out.append(dict(id=cid, ws=ws, claim=text, status=st, basis=basis, n_static=len(chk), note=note,
                    checked_at=max(SC['checked_at'], RP['checked_at'] if rows else '', SK['checked_at'] if fixed else ''),
                    evidence=ev, rerun=' ; '.join(filter(None, [RE_A if chk else '', RE_B if rows else '', RE_C if fixed else '']))))
now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
json.dump(dict(rendered_at=now, claims=out), open(os.path.join(R, 'claims.json'), 'w'), indent=1)
L = [f'# X1 worker 3: claims check for M5, M6, M7', '',
     f'Rendered {now} by `python3 X1/w3/make_claims.py` from results/static_check.json ({SC["checked_at"]}), results/repro.json ({RP["checked_at"]}) and results/skeptic.json ({SK["checked_at"]}). Paths are relative to `study/`; run the commands from `study/`. Do not edit this file by hand: rerun the scripts.', '',
     'Status: **verified** = the quoted numbers equal their files, and where a seed-4 rerun exists it lands within mean +- 2 sd with the qualitative claim holding; **partly verified** = one of the two holds; **refuted** = neither, or a quote mismatch; **unverified** = not tested. Basis says how far the check went: "file re-read" checks the quote only, not the run.', '',
     'Summary of the file re-read (part A): ' + '; '.join(f"{k} {v['match']}/{sum(v.values())} match" for k, v in SC['summary'].items()) + '.', '']
cnt = {}
for c in out: cnt[c['status']] = cnt.get(c['status'], 0) + 1
L += ['Claim statuses: ' + ', '.join(f'{k} {v}' for k, v in sorted(cnt.items())) + '.', '']
for ws in ('M5', 'M6', 'M7'):
    L += [f'## {ws}', '', '| id | claim | status | basis | checked_at | evidence | rerun | notes |', '|---|---|---|---|---|---|---|---|']
    for c in out:
        if c['ws'] == ws:
            L.append(f"| {c['id']} | {c['claim']} | **{c['status']}** | {c['basis']} ({c['n_static']} numbers) | {c['checked_at']} | {c['evidence']} | `{c['rerun']}` | {c['note']} |")
    L.append('')
L += ['## Seed-4 reproduction rows', '', '| row | what | seeds 1-3 mean +- sd | seed 4 | z | qualitative | status |', '|---|---|---|---|---|---|---|']
for r in RP['rows']:
    L.append(f"| {r['id']} | {r['what']} | {r['mean']:.5g} +- {r['sd']:.2g} | {'' if r['seed4'] is None else f'{r[chr(115)+chr(101)+chr(101)+chr(100)+chr(52)]:.5g}'} | {'' if r['z'] is None else f'{r[chr(122)]:+.2f}'} | {r['qual']} -> {r['qual_holds']} | {r['status']} |")
L += ['', '## Skeptic notes (results/skeptic.json)', '']
for k in ('M5', 'M6', 'M7'):
    L.append(f"- **{k}**: {SK[k]['note']}")
s4 = SK['M7'].get('seed4', {})
L += ['', '## Still unverified', '',
      '- M5: pi as a shifted-copy artefact (no probe on the PRNG-parity family); O(eps) invasion effect for w3 pattern traders; w2 bundle arbitrageur = Exploit LP; soup vs net comparison is order-of-magnitude; V1 panel 12.',
      '- M6: the noise margin in a stronger memorisation regime (partly addressed by the M6 stress pass, not by X1); Cm/Cw kill control in w5; the architecture test; Friedman 2001 and Herbster-Warmuth 1998 citations (unverified).',
      '- M7: GPU $ figures rest on the $0.87/h L4 rate (UNVERIFIED-ceiling in G1/GPU-LOG.md); DeepSeek bias update from recollection of arXiv 2408.15664 (https://arxiv.org/abs/2408.15664, unverified); stage-2 family-label proxy; seed 4 reruns ran on CPU (originals on GPU), so bitwise identity is not expected.',
      '']
open(os.path.join(H, 'CLAIMS.md'), 'w').write('\n'.join(L))
print(cnt)
