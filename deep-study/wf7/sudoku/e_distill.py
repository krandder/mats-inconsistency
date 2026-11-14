"""e_distill.py -- amplify, then distil into a proposal. Runs search S1 (M_A in its own-confidence order, corrected
against T_A) on the training puzzles with 12-16 clues and stores the ACCEPTED answers. D6 is then trained on
(puzzle, accepted answer) pairs: it never sees a real solution.
Usage: python -u e_distill.py"""
import time, json
import numpy as np, torch
import core

torch.set_num_threads(1); K, N = 6, 36; T0 = time.time()
meta = json.load(open("out/trust_meta_mA6.json")); TAU = meta["tau"]
MA = core.Net(core.load_model("ckpt/mA6.pt")[0], K, "mA6")
ck = torch.load("ckpt/trust_mA6.pt", map_location="cpu"); m = core.RelFormer(**ck["cfg"]); m.load_state_dict(ck["state"])
tA = core.TrustPolicy(m, K, "trustA")
d = np.load("data/k6_train_12p.npz"); NP_ = 2500; puz = torch.from_numpy(d["puz"][:NP_]); sol = torch.from_numpy(d["sol"][:NP_])
checker = core.make_checker(MA, tA, TAU); gen = torch.Generator().manual_seed(4242)
ans, idx = [], []
for a in range(0, len(puz), 250):
    r = core.search_first_accept(MA, core.pol_margin, puz[a:a + 250], None, checker, 8, gen, rounds=(1, 7))
    ans.append(r["answer"]); idx.append(r["acc_idx"])
    print(f"  {a+250}/{len(puz)}: accepted so far {(torch.cat(idx) >= 0).float().mean():.4f} ({time.time()-T0:.0f}s)", flush=True)
ans = torch.cat(ans); idx = torch.cat(idx); solved = idx >= 0
right = (ans == sol).all(1)                                              # SCORING only (not stored in the labels)
print(f"accepted {solved.float().mean():.4f} of {len(puz)} training puzzles within 8 candidates; accepted at candidate 0: "
      f"{(idx == 0).float().mean():.4f}; accepted answers equal to the real solution: {right[solved].float().mean():.6f} "
      f"({int((solved & ~right).sum())} wrong)")
full = np.zeros(len(d["puz"]), dtype=bool); full[:NP_] = solved.numpy()           # puzzles beyond NP_ are not used
ansf = -np.ones_like(d["sol"]); ansf[:NP_] = ans.numpy()
np.savez("out/distill_labels.npz", answer=ansf, solved=full)
print(f"calls of M_A {MA.calls}; done ({time.time()-T0:.0f}s)")
