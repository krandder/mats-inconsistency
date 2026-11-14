"""Mean-field predictions written into PREREG.md (computed before any main run)."""
import json, math, sys
sys.path.insert(0, '.')
import mf, pnets
SIG = (0.001, 0.003, 0.01, 0.03, 0.1)
out = {}
for reg, (kind, sw2, sb2) in pnets.REGIMES.items():
    row = {}
    for s in SIG:
        m = mf.MF(kind, sw2, sb2, s); o = m.run(400, m_list=(0, 32))
        R = o['R'][0]
        row[str(s)] = dict(Lstar=mf.crossing(R), R8=R[7], R16=R[15], R32=R[31], R64=R[63],
                           Rback32_to_64=o['R'][32][63])
    fp = mf.fixed_point(sw2, sb2) if kind == 'tanh' else (
        dict(chi1=1.0, lam=0.0) if kind == 'relu' else dict(chi1=math.pi / (math.pi - 1), lam=0.5 * math.log(math.pi / (math.pi - 1))))
    out[reg] = dict(fp=fp, by_sigma=row)
json.dump(out, open('mf_predictions.json', 'w'), indent=1)
for reg, v in out.items():
    fp = v['fp']
    print(f"{reg:12s} chi1={fp['chi1']:.3f} lam={fp['lam']:+.3f}" + (f" chi_c={fp['chi_c']:.3f} c*={fp['c_star']:.3f} q*={fp['q']:.3f}" if 'chi_c' in fp else ''))
    print('   L*: ' + ' '.join(f"{s}:{v['by_sigma'][str(s)]['Lstar']:.1f}" for s in SIG))
    print('   R32: ' + ' '.join(f"{s}:{v['by_sigma'][str(s)]['R32']:.3f}" for s in SIG) + ' | R64: ' + ' '.join(f"{v['by_sigma'][str(s)]['R64']:.3f}" for s in SIG))
