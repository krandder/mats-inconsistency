"""M9(a) w1: K/Q the soup would need for CA110 and LFSR, from lottery_fit.json + streams_check.json.
(i) empirical bound: one-sided 95% bound on the Brier gain at K/Q=64 -> upper bound on pi -> lower
    bound on K/Q for half of the achievable gain Delta* (from the stream simulation).
(ii) read-ladder model (an assumption): pi_f = pi_periodic * r**(extra reads without partial credit),
    r = success factor per extra history read, r in {2^-7 (the CA110 bound), 2^-10, 2^-20 (4 random
    bytes at 1/32)}.
Run: python3 predict.py
"""
import json, math, os, csv
HERE = os.path.dirname(os.path.abspath(__file__))
fit = json.load(open(os.path.join(HERE, 'lottery_fit.json')))
st = json.load(open(os.path.join(HERE, 'streams_check.json')))
pi_per = fit['periodic']['L2']['pi']
# achievable gain vs the 0.25 prior with the minimal number of reads, and reads needed
target = {
    'ca110': dict(reads=2, extra_reads=1, dstar=0.25 - st['ca110']['2']['brier']),
    'lfsr': dict(reads=4, extra_reads=3, dstar=0.25 - st['lfsr']['4']['brier']),
    'pi': dict(reads=4, extra_reads=3, dstar=0.25 - st['pi']['4']['brier']),
}
rows, out = [], {'pi_periodic': pi_per}
for f, t in target.items():
    mean64, sd64, n64 = fit[f]['cells']['64.0']
    gain_ub = (0.25 - mean64) + 1.645 * sd64 / math.sqrt(n64)
    if gain_ub <= 0:   # Brier got worse with K/Q (spill harm): the bound is not informative
        pi_ub = xhalf_lb = r_emp = float('nan')
    else:
        frac_ub = min(gain_ub / t['dstar'], 0.999)
        pi_ub = -math.log1p(-frac_ub) / 64
        xhalf_lb = math.log(2) / pi_ub
        r_emp = pi_ub / pi_per
    rec = dict(family=f, reads_needed=t['reads'], dstar=t['dstar'], brier64_mean=mean64, brier64_sd=sd64,
               gain64_ub95=gain_ub, pi_ub95=pi_ub, kq_half_lb=xhalf_lb, r_upper_from_bound=r_emp)
    for lab, r in (('r2m7', 2 ** -7), ('r2m10', 2 ** -10), ('r2m20', 2 ** -20)):
        pi_m = pi_per * r ** t['extra_reads']
        rec['kq_half_' + lab] = math.log(2) / pi_m
        g1024 = t['dstar'] * (1 - (1 - pi_m) ** 1024)
        rec['brier_at_kq1024_' + lab] = 0.25 - g1024
    rows.append(rec); out[f] = rec
json.dump(out, open(os.path.join(HERE, 'predictions.json'), 'w'), indent=1)
with open(os.path.join(HERE, 'predictions.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
    for r in rows:
        w.writerow({k: (f'{v:.4g}' if isinstance(v, float) else v) for k, v in r.items()})
for r in rows:
    print({k: (f'{v:.3g}' if isinstance(v, float) else v) for k, v in r.items()})
