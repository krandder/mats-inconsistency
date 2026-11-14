"""One CSV per run for V1 (live wealth view) and for every analysis (S0, worker 4).

Long format, fixed schema regardless of which metrics a run logs:
    run_id,step,wall_s,metric,group,value
`group` is a family name, unit id or 'all'; scalar metrics use group 'all'.
A sidecar <run_id>.meta.json holds the config (arms, seed, git-free code hash).
Rows are flushed on every write so a tailing reader (V1) sees them at once.

Also: PrequentialGuard, the runtime assert that train and eval never overlap.
"""
import csv
import hashlib
import json
import os
import time

FIELDS = ["run_id", "step", "wall_s", "metric", "group", "value"]


def code_hash(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:12]


class RunLog:
    def __init__(self, out_dir, run_id, meta=None):
        os.makedirs(out_dir, exist_ok=True)
        self.run_id = run_id
        self.path = os.path.join(out_dir, f"{run_id}.csv")
        self.t0 = time.time()
        new = not os.path.exists(self.path)
        self._f = open(self.path, "a", newline="")
        self._w = csv.writer(self._f)
        if new:
            self._w.writerow(FIELDS)
            self._f.flush()
        m = dict(meta or {})
        m.setdefault("run_id", run_id)
        m.setdefault("started_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        with open(os.path.join(out_dir, f"{run_id}.meta.json"), "w") as f:
            json.dump(m, f, indent=1, sort_keys=True, default=str)

    def log(self, step, row):
        """row: {metric: scalar} or {metric: {group: scalar}} (e.g. panel.panel_row output)."""
        wall = round(time.time() - self.t0, 3)
        for metric, v in row.items():
            items = v.items() if isinstance(v, dict) else (("all", v),)
            for g, x in items:
                self._w.writerow([self.run_id, int(step), wall, metric, g, repr(float(x))])
        self._f.flush()

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def read_run(path):
    """-> {(metric, group): [(step, value), ...]} in file order."""
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out.setdefault((r["metric"], r["group"]), []).append((int(r["step"]), float(r["value"])))
    return out


class PrequentialGuard:
    """Asserts the prequential contract on (question, time) keys:
      - a key is evaluated before it is ever trained on (test, then train);
      - a key is never evaluated after it was trained on (no eval on train data).
    In multi-epoch stress mode pass stress=True: re-training is allowed, and every eval
    after the first pass is flagged in `in_sample_evals` (so the panel reports it apart).
    Keys may be any hashable, e.g. (q, t); use eval_many/train_many for arrays of keys.
    """

    def __init__(self, stress=False):
        self.stress = stress
        self.evaluated, self.trained = set(), set()
        self.in_sample_evals = 0

    def eval(self, key):
        if key in self.trained:
            if not self.stress:
                raise AssertionError(f"eval on already-trained key {key!r}")
            self.in_sample_evals += 1
        self.evaluated.add(key)

    def train(self, key):
        if key not in self.evaluated:
            raise AssertionError(f"train on key {key!r} before it was evaluated")
        if key in self.trained and not self.stress:
            raise AssertionError(f"key {key!r} trained twice in single-pass mode")
        self.trained.add(key)

    def eval_many(self, keys):
        for k in keys:
            self.eval(k)

    def train_many(self, keys):
        for k in keys:
            self.train(k)
