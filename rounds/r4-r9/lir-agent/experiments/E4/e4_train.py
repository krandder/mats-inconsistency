"""E4 driver: `python3 e4_train.py [--e4-gating role|all] [--e4-fixed-n N] <protocol.train flags>`.
Wraps protocol.train (no harness code changed): adds the syco16 preset, the M1 model (gating all),
and, at every built-in eval point, the fixed teacher-forced set + KL-to-init + reliance + on-policy
extras, appended to <out>/e4_fixed.jsonl."""
import json, os, pickle, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e4lib as E
import protocol.train as PT

argv = list(sys.argv[1:])
def pop(flag, default):
    if flag in argv:
        i = argv.index(flag); v = argv[i + 1]; del argv[i:i + 2]; return v
    return default
gating = pop("--e4-gating", "role")
fixed_n = int(pop("--e4-fixed-n", "500"))
if gating == "all":
    PT.RoleModel = E.AllGatesRoleModel


class E4Trainer(PT.Trainer):
    _fx = None
    _init = None

    def evaluate(self, tag):
        out = super().evaluate(tag)
        t0 = time.time()
        if self._fx is None:
            self._fx = E.build_fixed_set(self.rc.preset, fixed_n)
        self.model.eval()
        fe = E.fixed_eval(self.model, self.V, self._fx, self._init)
        if self._init is None:
            self._init = {"first": fe["_first"], "commit": fe["_commit"]}
            fe.update(kl_init_first=0.0, kl_init_commit=0.0, kl_init_total=0.0)
        fe = {k: v for k, v in fe.items() if not k.startswith("_")}
        with open(os.path.join(self.rc.out, "evalrecs", f"{tag}_main.pkl"), "rb") as f:
            recs = pickle.load(f)
        fe.update(E.onpolicy_extra(recs, self.ecfg.shot_cost))
        m = out["main"]
        for k in ("reward", "env_reward", "success", "fc_committed_brier", "fc_committed_ece", "fc_committed_signed_vs_realised",
                  "fc_committed_signed_vs_bayes", "fc_all_brier", "fc_all_ece", "fc_all_signed_vs_realised", "fc_all_signed_vs_bayes",
                  "fc_all_brier_vs_bayes", "belief_brier", "belief_ece", "syco_note_bias", "syco_agree_rate",
                  "reliance_swap_argmax_change", "reliance_swap_tv", "reliance_raise_worst_dp", "bias_dp_hinted",
                  "commit_consistent_with_stated", "regret_vs_best_candidate_bayes"):
            if k in m:
                fe["op_" + k] = m[k]
        fe.update(tag=tag, step=self.step, gating=gating, arm=self.rc.arm, seed=self.rc.seed, preset=self.rc.preset,
                  lr_action=self.rc.lr_action, t_fixed_eval=time.time() - t0)
        with open(os.path.join(self.rc.out, "e4_fixed.jsonl"), "a") as f:
            f.write(json.dumps(fe, default=float) + "\n")
        return out


PT.Trainer = E4Trainer
PT.main(argv)
