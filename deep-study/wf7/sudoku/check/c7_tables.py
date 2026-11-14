"""c7_tables.py (checker) -- collects the numbers of my check into one tables file (numbers only; each block names its source).
Usage: python c7_tables.py > out/c7_tables.out"""
import json, os, math
J = lambda f: json.load(open(f"out/{f}")) if os.path.exists(f"out/{f}") else None
print("Tables of the independent check of wf7/sudoku. Numbers only; produced by c7_tables.py from the outputs named in each block.")
print("Model: Claude Fable 5.2 (claude-fable-5-2[1m]). The prose findings are the structured result of the checker step;")
print("no REPORT.md exists: the harness does not let a subagent write report files.\n")
p = J("c1_pairs.json")
if p:
    print("[A] one-cell contexts, all of them (c1_pairs.out)")
    for k in ("mA6", "mA4"):
        r = p[k]; print(f"   {k}: contexts {r['contexts']}; impossible atoms {r['n_impossible_atoms']}, largest log q {r['L_bad']:.3f}; possible atoms {r['n_possible_atoms']}, smallest log q {r['L_good']:.3f}; gap {r['gap']:.2f}; "
                        f"rule graph equals peers {r['graph_equals_peer_relation']}; valid grids score {r['valid_score_min']:.4f} to {r['valid_score_max']:.4f}")
s = J("c2_4x4_sym.json")
if s:
    print("\n[B] 4x4, exact: excess on all 288 / held-out 96 / training 192 / mass on valid (c2_4x4_sym.out)")
    for k, r in s["RES"].items():
        if "all288" in r:
            print(f"   {k:72s} {r['all288']:8.4f} {r['heldout96']:8.4f} {r['train192']:8.4f} {r['valid_mass']:9.6f}")
    for k, w in s["WALK"].items():
        print(f"   walk {k[:44]:44s}: boards {w['n']}, largest invalid T {w['max_invalid']:.3e}, e^tau {w['exp_tau']:.2e}, pruned {w['pruned']:.1e}, invalid accepted {w['invalid_accepted']}, valid rejected {w['valid_rejected']}")
    for k, d in s["DESC"].items():
        print(f"   route {k[:26]:26s}: free choices {d['free_per_route']:.2f} at mean step {d['mean_step_free']:.2f}; forced taken {d['forced_taken_when_avail']:.3f}; excess free {d['excess_free']:.4f} / forced-not-local {d['excess_forced_nonlocal']:.4f} / forced-local {d['excess_forced_local']:.4f}")
f = J("c2_4x4_full.json")
if f:
    print(f"\n[C] 4x4 trust MDP on every reachable state, no symmetry ({f['nstates']} states; c2_4x4_full.out)")
    print(f"   {'reality the MDP was solved on':24s} {'distinct grids':>14s} {'fit excess':>11s} {'KL(true||T)':>12s} {'held-out orbit':>15s} {'train orbit':>12s} {'grids not in sample':>20s}")
    for k, r in f.items():
        if isinstance(r, dict):
            print(f"   {k:24s} {r['n_distinct']:14d} {r['fit_excess']:11.4f} {r['kl_true']:12.4f} {r['heldout']:15.4f} {r['train']:12.4f} {r['unseen']:20.4f}")
r = J("c3_routes.json"); pop = J("c3_population.json")
if r:
    print("\n[D] 6x6 routes, 800 fresh TEST-orbit grids: excess [95%], minus pi_trust [95%] (c3_routes.out)")
    for k, v in r.items():
        if "excess" in v:
            print(f"   {k:44s} {v['excess']:7.4f} [{v['lo']:.3f}, {v['hi']:.3f}]  {v['d']:+.4f} [{v['dlo']:+.3f}, {v['dhi']:+.3f}]")
if pop:
    print("\n[E] 6x6 routes, 600 grids uniform over ALL grids = KL(reality || T): value [95%], minus pi_trust [95%], by orbit set train/val/test (c3_population.out, c3_population_pairs.out)")
    for k, v in pop.items():
        print(f"   {k:44s} {v['kl']:7.4f} [{v['lo']:.3f}, {v['hi']:.3f}]  {v['d']:+.4f} [{v['dlo']:+.3f}, {v['dhi']:+.3f}]  " + " / ".join(f"{x:.3f}" for x in v["by_split"]))
print("\n[F] search on HARD, fresh seeds, pair check from the table (c3_search_*.out, c6_agree.out)")
for a in ("S1", "S2", "S3", "S4"):
    q = J(f"c3_search_{a}.json")
    if q:
        extra = f"; route threshold on the same candidates: puzzles ending on a wrong answer {q['route_wrong_answers']}, accuracy at 6.0e9 FLOPs {q['anytime_route'][2]:.4f}" if "route_wrong_answers" in q else ""
        print(f"   {a}: n {q['n']}; accepted {q['accepted']:.4f}; accepted correct {q['accepted_correct']:.4f}; candidate 0 correct {q['first_correct']:.4f}; candidates {q['cands_mean']:.2f}; wrong candidates {q['wrong_candidates']}, accepted {q['wrong_accepted_pair']}; "
              f"FLOPs mean {q['flops_calls_mean']:.3e} (proposals only {q['flops_table_mean']:.3e}); accuracy at 6.0e9 FLOPs {q['anytime_calls'][2]:.4f}{extra}")
print("   (the S1 row above is my second seed, 777; c3_search_S1.json was overwritten by it. Both seeds, from the .out files:)")
for f_ in ("c3_search_S1.out", "c3_search_S1_seed2.out"):
    if os.path.exists(f"out/{f_}"):
        for l in open(f"out/{f_}"):
            if "accepted within" in l or "same candidates" in l or "one call per context read:" in l:
                print(f"   {f_}: {l.strip()}")
ag = J("c6_agree.json")
if ag:
    print("   same answers: " + "; ".join(f"{k} {v['same']}/{v['both']} (FLOPs ratio of means {v['ratio_mean']:.2f})" for k, v in ag.items()))
b = J("c3_base.json"); c = J("c3_conf.json"); fr = J("c3_fresh.json"); fr2 = J("c3_fresh2.json")
if b:
    print("\n[G] one pass on the worker's 300 HARD puzzles (c3_base.out)")
    for k, v in b.items():
        if "p_wrong_mean" in v:
            print(f"   {k:32s} correct {v['acc']:.4f}; at first error: probability on the wrong value {v['p_wrong_mean']:.3f}, >= 0.9 in {v['share_above_09']:.2f} of failures, surprise at truth {v['surprise_true']:.2f}")
    print("   best of k passes of M_B by own score: " + "; ".join(f"k={k[3:]}: {v['acc']:.4f} (solution among them {v['any']:.4f})" for k, v in b.items() if k.startswith("K2_")))
    print(f"   KT {b['KT']['acc']:.4f}; rule T_B > 1/2 accepts {b['KT']['rule_accepts']}, wrong {b['KT']['rule_wrong']}")
if c:
    print("\n[H] calibration on puzzle contexts (c3_conf.out): atoms >= 0.9 / wrong; atoms >= 0.99 / wrong; share right by stated probability")
    for k, v in c.items():
        print(f"   {k:40s} {v['conf']} / {v['conf_wrong']}; {v['conf99']} / {v['conf99_wrong']}; " + ", ".join(f"{lo:.2f}-{min(hi,1):.2f}: {a:.3f}" for lo, hi, n, a in v["cal"] if a is not None))
if fr:
    print(f"\n[I] my own new hard puzzles (c3_fresh.out): n {fr['n']} of {fr['made']} minimal; one pass M_A {fr['k0']:.4f}, M_B margin {fr['k1']['margin']:.4f} / entropy {fr['k1']['entropy']:.4f} / top probability {fr['k1']['maxprob']:.4f}; S1 + pair check accepted {fr['accepted']}, correct {fr['correct']}, wrong candidates {fr['wrong']} accepted {fr['wrong_accepted']}")
if fr2:
    print(f"   all fresh puzzles (c3_fresh2.out): n {fr2['n']}; " + "; ".join(f"{k}: M_A {v['MA']:.4f} M_B {v['MB']:.4f} (only M_A {v['only_MA']}, only M_B {v['only_MB']}, p {v['p']:.3f})" for k, v in fr2.items() if isinstance(v, dict))
          + f"; S1 accepted {fr2['accepted']}, all correct {fr2['all_correct']}, candidates {fr2['cands']:.2f}; accuracy at 6.0e9/1.2e10/3e10/1e11 FLOPs {fr2['any_calls']}")
st = J("c3_stiff.json"); dr = J("c3_draws.json"); rc = J("c4_recurrence.json")
if st:
    print("\n[J] what M rules out, by cells revealed (c3_stiff.out): share ruled out; real values ruled out; ruled-out-but-possible (120 states); rule-breaking caught; dead ends caught (count)")
    for v in st["rows"]:
        print(f"   {v['depth']:3d}: {v['ruled_out']:.4f}; {v['real_ruled_out']:.1e}; {v['ruled_out_but_possible']}; {v['direct_recall']:.4f}; " + (f"{v['dead_recall']:.4f} ({v['n_dead']})" if v['dead_recall'] is not None else "none exist"))
    print(f"   rule-breaking mass per atom mean {st['leak_direct_mean']:.2e}, max {st['leak_direct_max']:.2e}; confident atoms {st['conf']['n']}, confidently wrong {st['conf']['wrong']}")
if dr:
    print("\n[K] fresh draws from T (c3_draws.out)")
    for k, v in dr.items():
        if "frac_invalid" in v:
            print(f"   T by {k:15s}: draws {v['n']}, invalid {v['invalid']} ({100*v['frac_invalid']:.2f}%), pass the route threshold {v['route_accepts_invalid']}, pass the pair check {v['pair_accepts_invalid']} (largest pair score {v['pair_max_invalid']:.2f})")
if rc:
    print("\n[L] reading the rule from real data by recurrence (c4_recurrence.out)")
    for k, v in rc.items():
        if k.startswith("rec"):
            print(f"   a real {k[3:]}-cell context is found in {v['mean']:.1f} of 4,000 real grids on average (5th percentile {v['p5']:.0f})")
        else:
            print(f"   n = {k:>5s} real grids: each one-cell context recurs >= {v['min_recurrence']}; never-seen values {v['data_zeros']}, false zeros {v['false_zeros']}; equal to M's 2,592 vetoes: {v['data_zeros_equal_M_vetoes']}")
for f_ in ("c3_splits.out", "c3_rater.out", "c5_orbit.out", "c3_valorbits.out"):
    if os.path.exists(f"out/{f_}"):
        print(f"\n[{f_}]"); print("".join(l for l in open(f"out/{f_}") if "per orbit (n;" not in l and not l.startswith("exit")).rstrip())
fx = J("c8_fixed.json")
if fx:
    print("\n[M] 4x4, a small class of routes learned from n real grids, scored exactly against the whole law (c8_fixed.out): n; best single of 256 fixed orders picked on the sample, mean [range]; EM mixture of the 256, mean [range]")
    for k, v in fx.items():
        if k != "whole":
            print(f"   {k:>5s}: {v['single_mean']:.4f} [{v['single_min']:.4f}, {v['single_max']:.4f}]; {v['mix_mean']:.4f} [{v['mix_min']:.4f}, {v['mix_max']:.4f}]")
    print(f"   whole law: {fx['whole']['single']:.4f}; {fx['whole']['mix']:.4f}")
for f_ in ("c3_population_pairs.out",):
    if os.path.exists(f"out/{f_}"):
        print(f"\n[{f_}]"); print(open(f"out/{f_}").read().rstrip())
