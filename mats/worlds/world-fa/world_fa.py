"""World FA for the shared harness (work/worlds/harness.py). Core maths: fa_core.py; spec as built: PREREG-FA.md.

usage (from work/worlds, with this directory on the path):
    PYTHONPATH=../world-fa python harness.py train --world world_fa --world-cfg '{"prior": "a"}' ... --nmin 3 --nmax 8
    PYTHONPATH=../world-fa python harness.py evalset --world world_fa --world-cfg '{"prior": "a"}' --Ns 4,8,12 \
        --policies T0,T8,T32,shufT32,dedup3T8,gate3T16 --out runs/es_fa

Policies (the harness's per-cell `policy` string):
  mix          training mixture of T (cfg T_train, p_T0)
  T<t>         fixed history length t
  shufT<t>     shuffled-history control: history columns permuted among traders relative to the query round.
               'bayes' = Bayes on what the model sees (shuffled); 'bayes_true' = Bayes on the true episode.
  dedup<k>T<t> one-hop dedup, graded: groups of 6 arms (f = 0, .25, .5, .75, .9, .97) of k special traders with one
               marginal level, background N - k traders from the prior, common random numbers across arms.
  freshT<t>    structures held out of training ((f) templates with train=False; (a) block sizes with a_train_alpha 0);
               'bayes' uses the full prior, 'bayes_train' the training part (memorisation check).
  gate<k>T<t>  history gate: same arms, but every arm's query round equals arm 0's (f = .97), so the arms differ
               only in history; arm order f = (.97, .9, .75, .5, .25, 0).
Training batches are generated on the GPU when available (sample_train returns X, mask as device tensors); training
episodes exclude held-out structures; cfg T_curr gives a T curriculum by training step (sample_train call count).
"""
import json
import math
import re

import numpy as np
import torch

import fa_core as C

DEDUP_FS = (0.0, 0.25, 0.5, 0.75, 0.9, 0.97)
GATE_FS = (0.97, 0.9, 0.75, 0.5, 0.25, 0.0)


def _seed(rng):
    return int(rng.integers(0, 2 ** 62))


class FAWorld:
    target_kind = 'real'
    name = 'world_fa'

    def __init__(self, cfg=None):
        c = C.full_cfg(cfg)
        c.setdefault('train_policies', ['mix'])
        c.setdefault('heldout', ['T0', 'T8', 'T32', 'shufT32'])
        self.cfg = json.loads(json.dumps(c))                     # JSON-stable (the harness compares saved cfgs)
        self.c = c
        self.feat_dim = C.feat_dim(c)
        self.train_policies = list(c['train_policies']); self.heldout = list(c['heldout'])
        self.test_policies = self.train_policies + [p for p in self.heldout if p not in self.train_policies]
        self.train_dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.calls = 0                                             # sample_train calls (= training steps)
        # training generator config: held-out structures removed ((a) a_train_alpha; (f) templates with train=False)
        self.c_train = dict(c, _part='train')
        if c.get('a_train_alpha') is not None:
            self.c_train['a_alpha'] = tuple(float(x) for x in c['a_train_alpha'])

    # ------------------------------------------------------------------ sampling
    def _batch(self, E, N, policy, dev, n=None):
        c = self.c
        Ns = N if n is None else torch.as_tensor(n, device=dev).long()
        if policy == 'mix':
            return C.gen(E, Ns, C.sample_T(E, c, dev), c, dev)
        mt = re.fullmatch(r'(shuf|fresh)?T(\d+)', policy)
        if mt:
            if mt.group(1) == 'fresh':
                return self._fresh(E, Ns, int(mt.group(2)), dev)
            b = C.gen(E, Ns, int(mt.group(2)), c, dev)
            return C.shuffle_history(b) if mt.group(1) else b
        md = re.fullmatch(r'(dedup|gate)(\d+)T(\d+)', policy)
        if md:
            gate = md.group(1) == 'gate'
            fs = GATE_FS if gate else DEDUP_FS
            return C.block_sets(max(1, E // len(fs)), N, int(md.group(3)), int(md.group(2)), fs, c, dev, gate=gate)
        raise ValueError(policy)

    def _fresh(self, E, N, T, dev):
        """Structures held out of training: (f) templates with train=False; (a) episodes with a block size whose
        a_train_alpha is 0 (rejection sampling from the full prior)."""
        c = self.c
        if c['prior'] == 'f':
            return C.gen(E, N, T, dict(c, _part='fresh'), dev)
        assert c['prior'] == 'a' and c.get('a_train_alpha') is not None, 'fresh cells need held-out structures'
        held = [s + 1 for s, a in enumerate(c['a_train_alpha']) if a == 0]
        out = []
        while sum(len(x['v']) for x in out) < E:
            b = C.gen(4 * E, N, T, c, dev)
            keep = torch.nonzero(torch.isin(b['size'], torch.as_tensor(held, device=dev)).any(1))[:, 0]
            if len(keep):
                out.append(C.take(b, keep))
        cat = {k: (torch.cat([x[k] for x in out])[:E] if torch.is_tensor(out[0][k]) and out[0][k].dim() > 0 else out[0][k])
               for k in out[0]}
        return cat

    def _wrap(self, b, policy, numpy_x=True):
        X, M = C.tokens(b, self.c)
        if numpy_x:
            X, M = X.cpu().numpy(), M.cpu().numpy()
        return dict(X=X, mask=M, target=b['v'].cpu().numpy().astype(np.float64), b=b,
                    policy=np.array([policy] * len(b['v'])))

    def sample(self, rng, E, N, policy, n=None):
        """Evaluation cells: CPU, deterministic given rng (dedup/gate round E down to whole groups of 6)."""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(_seed(rng))
            b = self._batch(E, N, policy, 'cpu', n)
        return self._wrap(b, policy)

    def sample_train(self, rng, E, nmin, nmax):
        """Training batch on the train device: N log-uniform in [nmin, nmax] per episode, T from the training mix.
        X and mask stay device tensors (the harness's torch.as_tensor is then a no-op); target is numpy."""
        dev = self.train_dev
        n = np.exp(rng.uniform(np.log(nmin), np.log(nmax + 1), E)).astype(np.int64).clip(nmin, nmax)
        torch.manual_seed(_seed(rng))
        ct = dict(self.c_train)
        if self.c.get('T_curr'):                                  # curriculum over T (large -> small)
            # the harness draws its validation set with the FIRST call: give it the final-stage distribution
            step = self.calls - 1 if self.calls > 0 else 10 ** 12
            st = [x for x in self.c['T_curr'] if step >= x[0]][-1]
            ct.update(T_train=(int(st[1]), int(st[2])), p_T0=float(st[3]))
        self.calls += 1
        b = C.gen(E, torch.as_tensor(n, device=dev), C.sample_T(E, ct, dev), ct, dev)
        return self._wrap(b, 'mix', numpy_x=(dev == 'cpu'))

    # ------------------------------------------------------------------ Bayes and references
    @staticmethod
    def _np(p):
        return tuple(x.detach().cpu().numpy().astype(np.float64) for x in p)

    def exact_bayes(self, ep):
        return self._np(C.bayes(ep['b'], self.c, True))

    def references(self, ep):
        b = ep['b']
        refs = C.references(b, self.c)
        if 'true' in b:                      # shuffled control: also Bayes on the true (unshuffled) episode
            refs['bayes_true'] = C.bayes(b['true'], self.c, True)
        return {k: self._np(v) for k, v in refs.items()}

    def meta(self, ep):
        b = ep['b']
        E = len(b['v'])
        g = lambda k, d=-1.0: b[k].detach().cpu().numpy().astype(float) if k in b and torch.is_tensor(b[k]) \
            and b[k].shape[:1] == (E,) else np.full(E, d)
        d = dict(N=g('N'), T=g('T'), arm=g('arm'), group=g('group'), f=g('fk', np.nan), s2k=g('s2k', np.nan),
                 h=g('h'), template=g('template'))
        if 'size' in b:
            size = b['size'].cpu().numpy()
            d['n_shared'] = (size >= 2).sum(1).astype(float); d['max_block'] = size.max(1).astype(float)
        else:
            d['n_shared'] = np.full(E, np.nan); d['max_block'] = np.full(E, np.nan)
        Sig = b['Sigma'].cpu().numpy(); vt = b['vt'].cpu().numpy()
        cor = []
        for i in range(E):
            n = int(vt[i].sum()); S = Sig[i, :n, :n]; s = np.sqrt(np.diag(S))
            R = S / s[:, None] / s[None]
            cor.append(np.abs(R[np.triu_indices(n, 1)]).mean() if n > 1 else 0.0)
        d['mean_abs_corr'] = np.array(cor)
        d['policy'] = ep['policy']
        return d

    splits = {'all': lambda m: np.ones(len(m['N']), bool),
              'T0': lambda m: m['T'] == 0,
              'Tpos': lambda m: m['T'] > 0,
              'shared': lambda m: m['n_shared'] > 0}
    abilities = [('history use (bayes_T0->bayes)', 'bayes_T0', 'bayes', 'Tpos'),
                 ('structure beyond precision weights (prec->bayes)', 'prec', 'bayes', 'Tpos'),
                 ('beyond plug-in GLS (gls_plugin->bayes)', 'gls_plugin', 'bayes', 'Tpos'),
                 ('any (prior->bayes)', 'prior', 'bayes', 'all')]


def make_world(cfg=None):
    return FAWorld(cfg)
