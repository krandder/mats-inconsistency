"""P4 stages 1, 2 and the noise check: one job per net, 6 niced single-thread workers.

  python run.py pilot     # 5 random circuits + H1 + H3 (seed 0)
  python run.py full      # everything not yet done (resumes: skips existing JSON)
Per-net JSON goes to results/nets/<job>.json.
"""
import json, os, sys, time, traceback
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'nets')
EPS_NOISE = [0.25] * 4 + [0.02] * 4


def rnd(x, d=6):
    if isinstance(x, float):
        return float(f'{x:.{d}g}')
    if isinstance(x, (list, tuple)):
        return [rnd(v, d) for v in x]
    if isinstance(x, dict):
        return {str(k): rnd(v, d) for k, v in x.items()}
    if isinstance(x, (np.floating,)):
        return float(f'{float(x):.{d}g}')
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return rnd(x.tolist(), d)
    return x


def prog_for(kind, cs):
    import gen
    if kind == 'h1':
        return gen.h1()
    if kind == 'h3':
        return gen.h3()
    if kind == 'noise':
        return gen.random_circuit(cs, split_cones=True)
    return gen.random_circuit(cs)


def job_name(j):
    kind, cs, s = j
    return f'{kind}_c{cs}_s{s}'


def run_job(j):
    os.nice(10)
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, HERE)
    import gen, nets, crit
    kind, cs, s = j
    name = job_name(j)
    path = os.path.join(OUT, name + '.json')
    if os.path.exists(path):
        return name, 'skip'
    t0 = time.time()
    try:
        prog = prog_for(kind, cs)
        X = gen.allx(prog['nin'])
        V = gen.values(prog, X)
        Y = V[-1].T
        rec = dict(job=name, kind=kind, circuit=prog['name'], circuit_seed=cs, seed=s)
        if kind in ('r2', 'null'):
            if kind == 'r2':
                net, hist = nets.train_natural(prog, s, verbose=False)
                with torch.no_grad():
                    rec['train_acc'] = float(((net(nets.xin(X)) > 0).numpy() == Y).mean())
                rec['hist'] = hist
            else:
                torch.manual_seed(1000 + s)
                net = nets.MLP(nin=prog['nin'], nout=Y.shape[1], sign_layer=2, group_readout=False).eval()
            rec['align'] = crit.code_alignment(net, prog, X)
            if kind == 'r2':
                rec['r2'] = crit.r2_analysis(net, prog, X, Y, seed=s)
        else:
            eps = EPS_NOISE if kind == 'noise' else None
            # H1's 8-bit parity plateaus at IIA 0.68 under plain SIIT; a per-node probe loss (0.2) breaks it (stated deviation)
            net, hist = nets.train_siit(prog, s, eps=eps, log_every=250, min_steps=1000, verbose=False,
                                        all_levels=True, accept=0.99, node_sup=0.2 if kind == 'h1' else 0.0)
            rec['hist'] = hist
            rec['iia'] = hist[-1]
            rec['accepted'] = bool(hist[-1]['iia'] >= 0.98 and hist[-1]['strict_iia'] >= 0.98)
            if kind == 'noise':
                with torch.no_grad():
                    p = torch.sigmoid(net(nets.xin(X))).numpy()
                pk = np.where(Y == 1, p, 1 - p).mean(0)          # mean prob on the clean label
                rec['calib_mean_p'] = pk.tolist()
                rec['calibrated'] = bool(np.all(np.abs(pk - (1 - np.array(EPS_NOISE))) <= 0.02))
            levels = []
            nlev = len(prog['levels']) - 2
            for l in range(1, nlev + 1):
                groups = nets.groups_for(prog, l)
                la = crit.level_analysis(net, l, X, Y, groups, seed=1000 * s + l, eps=eps)
                xi, xp = la.pop('_pairs')
                T = la['_T']
                pt = gen.program_tables(prog, l, X, xi, xp)
                la['program'] = dict(r=pt['r'], marg=pt['flip']['marg'], shap=pt['flip']['shap'], m=pt['m'], u=pt['u'],
                                     pay_shap=pt['pay']['shap'], mi_shap=pt['mi']['shap'])
                la['roles'], la['names'], la['types'] = pt['roles'], pt['names'], pt['types']
                pl = prog.get('plants', {}).get(l, {})
                la['plants'] = {k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in pl.items()
                                if k != 'epi_src_name'}
                la['plant_scores'] = crit.plant_analysis(la, pl) if pl else None
                la.pop('_T')
                la['level'] = l
                levels.append(la)
            rec['levels'] = levels
        rec['sec'] = round(time.time() - t0, 1)
        json.dump(rnd(rec), open(path + '.tmp', 'w'))
        os.replace(path + '.tmp', path)
        return name, f'ok {rec["sec"]}s'
    except Exception:
        return name, 'ERR ' + traceback.format_exc()[-1500:]


def jobs(mode):
    J = []
    if mode in ('pilot', 'full'):
        J += [('r1', c, 0) for c in range(5)] + [('h1', 0, 0), ('h3', 0, 0)]
    if mode == 'full':
        J += [('r1', c, 0) for c in range(5, 30)] + [('r1', c, 1) for c in range(5)]
        J += [('h1', 0, s) for s in (1, 2)] + [('h3', 0, s) for s in (1, 2)]
        st0 = json.load(open(os.path.join(HERE, 'results', 'stage0.json')))
        sep = [d['seed'] for d in st0['separated_cones']['chosen']][:6]
        J += [('noise', c, 0) for c in sep]
        J += [('r2', 100 + c, 0) for c in range(10)] + [('null', 100 + c, c) for c in range(10)]
    if mode == 'r2':
        J += [('r2', 100 + c, 0) for c in range(10)] + [('null', 100 + c, c) for c in range(10)]
    if mode.startswith('one:'):
        k, c, s = mode[4:].split(',')
        J = [(k, int(c), int(s))]
    return J


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    os.environ['OMP_NUM_THREADS'] = '1'
    J = [j for j in jobs(sys.argv[1]) if not os.path.exists(os.path.join(OUT, job_name(j) + '.json'))]
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    print('jobs', len(J), flush=True)
    t0 = time.time()
    with mp.get_context('spawn').Pool(nw, maxtasksperchild=1) as pool:
        for name, st in pool.imap_unordered(run_job, J):
            print(round(time.time() - t0), name, st, flush=True)
    print('ALL DONE', round(time.time() - t0), flush=True)
