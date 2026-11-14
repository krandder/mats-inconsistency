"""M2 lead: re-read every number quoted in BAKEOFF.md from its source file (policy 15).
Usage: python3 check_bakeoff.py   (stdlib only; exit 1 on any mismatch). Tolerance: the quoted rounding.
"""
import csv, json, os, sys
H = os.path.dirname(os.path.abspath(__file__))
def rows(p): return list(csv.DictReader(open(os.path.join(H, p))))
SP, CO = rows("w3/results/spearman_summary.csv"), rows("w3/results/coalition_summary.csv")
FR, AD = rows("w3/results/frontier_summary.csv"), rows("w3/results/additivity_summary.csv")
T1, T2 = rows("w4/results/summary_metrics.csv"), rows("w4/results/s2_summary_metrics.csv")
W1 = json.load(open(os.path.join(H, "w1/results/summary.json")))
SR = json.load(open(os.path.join(H, "w3/results/stoprule.json")))
bad = n = 0
def chk(label, got, want, tol):
    global bad, n
    n += 1; ok = got is not None and abs(float(got) - want) <= tol
    if not ok: bad += 1; print(f"MISMATCH {label}: file={got} quoted={want}")
def one(R, **kw):
    m = [r for r in R if all(r[k] == v for k, v in kw.items())]
    return m[0] if len(m) == 1 else None
def sp(ck, g, p, m, s, t=6e-4):
    r = one(SP, ckpt=ck, gran=g, predictor=p, ablation="zero")
    chk(f"spearman {ck} {g} {p}", r and r["rho_mean"], m, t); chk(f"spearman-sd {ck} {g} {p}", r and r["rho_sd"], s, t)
# neurons x 4 checkpoints (section 1)
N = {"abl_train": [(.560,.085),(.492,.116),(.439,.113),(.412,.106)], "settle_loo_train": [None,(.458,.043),None,(.464,.041)],
     "absag_int": [(.421,.166),(.354,.090),(.322,.128),(.313,.115)], "v2_lmsr_inst": [(.277,.099),(.226,.029),(.227,.052),(.199,.075)],
     "v2_lmsr_int": [(.277,.099),(.149,.028),(.162,.045),(.162,.044)], "v1_wealth": [(.172,.009),(.065,.067),(.060,.070),(.060,.063)],
     "taylor_abs": [(.386,.168),(.220,.086),(.138,.104),(.103,.084)], "taylor": [(.211,.174),(.171,.059),(.137,.159),(.076,.053)],
     "movement": [(.162,.035),(-.061,.054),(-.077,.077),(-.089,.082)], "wout_norm": [(.131,.039),(.020,.037),(-.015,.054),(-.022,.066)]}
for p, v in N.items():
    for ck, x in zip(["8000", "20000", "30000", "final"], v):
        if x: sp(ck, "neuron", p, *x)
H_ = {"abl_train": (.929,.083), "shap_head_exact": (.865,.060), "absag_int": (.714,.095), "v2_lmsr_inst": (.698,.339),
      "v2_lmsr_int": (.667,.401), "v1_wealth": (.325,.387), "taylor_abs": (.809,.143), "taylor": (.111,.730), "movement": (.571,.227)}
for p, x in H_.items(): sp("final", "head", p, *x)
sp("20000", "head", "v2_lmsr_inst", .675, .387); sp("20000", "head", "taylor_abs", .762, .063)
R_ = {"abl_train": (.742,.020), "absag_int": (-.754,.014), "v2_lmsr_inst": (.187,.033), "v2_lmsr_int": (.303,.022),
      "taylor_abs": (-.750,.017), "movement": (.814,.027), "wout_norm": (.724,.049)}
for p, x in R_.items(): sp("final", "readout", p, *x)
sp("20000", "readout", "v2_lmsr_int", .353, .017); sp("20000", "readout", "movement", .824, .011)
# stop rule json (final)
for g, st in [("neuron", "win"), ("head", "tie"), ("readout", "loss"), ("neuron_settle", "win")]:
    n += 1
    if SR[g]["status"] != st: bad += 1; print("MISMATCH stoprule", g, SR[g]["status"], st)
chk("stoprule neuron diff", SR["neuron"]["diff"], .0959, 1e-4); chk("stoprule neuron margin", SR["neuron"]["margin"], .0839, 1e-4)
n += 1
if SR["coalition_win"]["any"] is not True: bad += 1; print("MISMATCH coalition_win.any")
# coalitions (random kinds, final + 20k): kind 'random' rows
for ck, p, m, s in [("final","v2_lmsr_int",.818,.027),("final","taylor_abs",.800,.010),("final","movement",.804,.023),
                    ("final","taylor",.115,.132),("final","settle_loo_train",.816,.050),("final","abl_train",.908,.033),
                    ("final","shap_head_exact",.935,.028),("20000","v2_lmsr_int",.833,.043),("20000","taylor_abs",.822,.024)]:
    r = one(CO, ckpt=ck, kind="random", predictor=p)
    chk(f"coalition {ck} {p}", r and r["rho_mean"], m, 6e-4); chk(f"coalition-sd {ck} {p}", r and r["rho_sd"], s, 6e-4)
for ck, k, m in [("final","random",.974),("final","structured",.900),("20000","random",.963)]:
    r = one(AD, ckpt=ck, kind=k, ablation="zero"); chk(f"additivity rho {ck} {k}", r and r["rho_sum_singles_mean"], m, 6e-4)
r = one(AD, ckpt="final", kind="random", ablation="zero"); chk("median rel gap", r["median_rel_gap_mean"], .181, 6e-4)
chk("frac superadditive", r["frac_superadditive_mean"], .425, 6e-4)
# frontier AUC (neurons)
for ck, p, m, s in [("final","abl_train",.109,.030),("final","absag_acc",.155,.077),("final","settle_loo_train",.170,.018),
                    ("final","taylor_abs",.200,.044),("final","movement",.295,.129),("final","random",.451,.218),
                    ("final","v1_wealth",.580,.372),("final","v2_lmsr_inst",.863,.645),("final","taylor",.939,.824),
                    ("final","oracle_zero",.003,.061),("20000","settle_loo_train",.146,.009),("20000","taylor_abs",.159,.114),
                    ("20000","random",.485,.188),("20000","v2_lmsr_inst",.723,.805)]:
    r = one(FR, ckpt=ck, gran="neuron", predictor=p)
    chk(f"frontier {ck} {p}", r and r["auc_mean"], m, 6e-4); chk(f"frontier-sd {ck} {p}", r and r["auc_sd"], s, 6e-4)
# oracle reliability (w1)
for k, m in [("final/neuron/all/split_half_zero", .916), ("step20000/neuron/all/split_half_zero", .925),
             ("step8000/neuron/all/split_half_zero", .945), ("final/head/all/split_half_zero", .73)]:
    chk("w1 " + k, W1[k]["mean"], m, 6e-3)
# tracr stage 1 (beta 5, POOLED_nodyck) and stage 2 (POOLED)
for g, met, p, m, s in [("head","neg_on_important","taylor",.67,.29),("neuron","neg_on_important","taylor",.42,.12),
                        ("neuron","neg_on_important","loo_direct",0,0),("neuron","auroc_used","magnitude",.81,0),
                        ("head","auroc_used","magnitude",.32,0),("neuron","auroc_used","loo_direct",.60,0),
                        ("neuron","auroc_used","taylor",.99,0),("neuron","sp_live","taylor",.26,.06),("neuron","sp_live","oracle_mean",.87,.04)]:
    r = one(T1, prog="POOLED_nodyck", beta="5.0", gran=g, metric=met, predictor=p)
    chk(f"tracr s1 {g} {met} {p}", r and r["mean"], m, 6e-3); chk(f"tracr s1 sd {g} {met} {p}", r and (r["std"] or 0), s, 6e-3)
for g, met, p, m, s in [("neuron","neg_on_important","books_signed",.57,.02),("head","neg_on_important","books_signed",.60,.20),
                        ("neuron","neg_on_important","taylor",.54,.07),("neuron","sp_live","books_abs",.81,.03),
                        ("neuron","sp_live","books_signed",.05,.02),("neuron","sp_live","movement",.05,.02),
                        ("neuron","auroc_used","books_signed",.99,.01)]:
    r = one(T2, prog="POOLED", gran=g, metric=met, predictor=p)
    chk(f"tracr s2 {g} {met} {p}", r and r["mean"], m, 6e-3); chk(f"tracr s2 sd {g} {met} {p}", r and r["std"], s, 6e-3)
print(f"{n - bad}/{n} quoted numbers match their files"); sys.exit(1 if bad else 0)
