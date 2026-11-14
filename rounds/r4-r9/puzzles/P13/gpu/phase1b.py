"""P13 Phase 1b (parts 3-4 of phase1.py, copied) on one clean L4 (run from ~/p13 on the VM): writes out/phase1b.json.
 1. clean (solo) 300-step timing at 24 px of every arm's training step (after 20 warm-up steps);
 2. packed timing: 2 and 3 ECON jobs at once, DTR + ECON, and BPTT + ECON (sets the packing plan);
 3. eval throughput at 204 px: 50 steps of M-ref and of a Trader2D on 8 d99 mazes;
 4. GPU kill-and-resume test: ECON killed after it 20 and resumed to 40 vs two uninterrupted runs to 40
    (the pair of uninterrupted runs measures GPU nondeterminism); state checksum at kill vs after reload.
"""
import hashlib, json, os, subprocess, sys, time
import torch

OUT = "out/p1"
os.makedirs(OUT, exist_ok=True)
PY = sys.executable
R = {"gpu": torch.cuda.get_device_name(0), "start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
ARMS = {
    "econ": "econ_maze.py --mode econ",
    "econ_nocap": "econ_maze.py --mode econ --wealth_cap 0",
    "econ_insq": "econ_maze.py --mode econ --insq 1",
    "econ_liqsqrt": "econ_maze.py --mode econ --liq sqrt",
    "bptt_dscov": "econ_maze.py --mode bptt --ds 1 --covsup 1 --vocsup 1 --ptopt 1 --gradckpt 1",
    "bptt_progcov": "econ_maze.py --mode bptt --prog 1 --alpha 0.01 --covsup 1 --vocsup 1 --ptopt 1 --gradckpt 1",
    "dtr": "dtr_maze.py --steps 40000",
}


def cmd(arm, tag, n=300):
    c = ARMS[arm].split()
    if arm == "dtr":
        return [PY] + c + ["--name", tag, "--out", OUT, "--time_steps", str(n)]
    return [PY] + c + ["--out", f"{OUT}/{tag}", "--time_steps", str(n)]


def timing_line(txt):
    for l in txt.splitlines():
        if l.startswith("{") and '"timing"' in l:
            return json.loads(l)
    return {"error": txt[-800:]}


def run_group(jobs):
    t0 = time.time()
    ps = [(tag, subprocess.Popen(cmd(arm, tag), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)) for arm, tag in jobs]
    res = {tag: timing_line(p.communicate()[0]) for tag, p in ps}
    res["_wall"] = round(time.time() - t0, 1)
    return res


def save():
    json.dump(R, open("out/phase1b.json.tmp", "w"), indent=1)
    os.replace("out/phase1b.json.tmp", "out/phase1b.json")


# Phase 1b (after the spot preemption of trader-b3-p13-1 at 22:03Z during dtr_x2): only parts 3 and 4.
# Parts 1-2 results: out/phase1.json (solo, econ_x2, econ_x3), dtr+econ recovered from the live log (gpu/phase1_recovered.json).
# eval throughput at 204 px, 50 steps, 8 mazes
subprocess.run([PY, "econ_maze.py", "--mode", "econ", "--iters", "3", "--out", f"{OUT}/tiny_econ", "--log_every", "1"],
               stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
R["eval204"] = {}
for kind, ck in [("mref", "mref/dt_net.pth"), ("trader", f"{OUT}/tiny_econ.pt")]:
    t0 = time.time()
    p = subprocess.run([PY, "eval_maze.py", "--kind", kind, "--ckpt", ck, "--sets", "d99", "--limit", "8",
                        "--mult", str(50 / 99), "--out", f"{OUT}/ev_{kind}"], capture_output=True, text=True)
    R["eval204"][kind] = {"wall_sec": round(time.time() - t0, 1), "rc": p.returncode, "tail": p.stdout[-600:] + p.stderr[-600:]}
    print(json.dumps({kind: R["eval204"][kind]}), flush=True); save()


# GPU kill-and-resume
def econ_run(tag, extra):
    base = [PY, "econ_maze.py", "--mode", "econ", "--iters", "40", "--log_every", "1", "--diag_every", "10", "--out", f"{OUT}/{tag}"]
    return subprocess.run(base + extra, capture_output=True, text=True)


def logs(tag):
    return json.load(open(f"{OUT}/{tag}.json"))["log"]


def sha_state(sd_list):
    h = hashlib.sha256()
    for sd in sd_list:
        for k in sorted(sd):
            h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


rs = {}
for tag in ["full_a", "full_b"]:
    rs[tag] = econ_run(tag, []).returncode
k = econ_run("killres", ["--kill_at", "20"])
rs["kill_rc"] = k.returncode
ck = torch.load(f"{OUT}/killres.ckpt.pt", map_location="cpu", weights_only=False)
rs["ckpt_state_sha"] = sha_state(ck["traders"])
# reload through the model code (as the resume does) and checksum again
sys.path.insert(0, os.getcwd())
import econ_maze as E  # noqa: E402
a = E.get_args(["--out", "/tmp/x"])
M = E.Model(a, "cuda")
for t, sd in zip(M.traders, ck["traders"]):
    t.load_state_dict(sd)
rs["reloaded_state_sha"] = sha_state([t.state_dict() for t in M.traders])
rs["checksum_match"] = rs["ckpt_state_sha"] == rs["reloaded_state_sha"]
r2 = econ_run("killres", [])
rs["resume_printed"] = "resumed_at_it" in r2.stdout
A, Bq, K = logs("full_a"), logs("full_b"), logs("killres")


def maxdiff(L1, L2, lo=20):
    d = 0.0
    for x, y in zip(L1, L2):
        if x["it"] < lo:
            continue
        for key, v in x.items():
            if isinstance(v, float) and key != "sec" and isinstance(y.get(key), float):
                d = max(d, abs(v - y[key]))
    return d


rs["resume_vs_full_a_maxdiff_it20_39"] = maxdiff(K, A)
rs["full_b_vs_full_a_maxdiff_it20_39 (GPU nondeterminism floor)"] = maxdiff(Bq, A)
rs["n_logged"] = [len(A), len(Bq), len(K)]
rs["pass_loss_1e-4"] = rs["resume_vs_full_a_maxdiff_it20_39"] <= max(1e-4, 2 * rs["full_b_vs_full_a_maxdiff_it20_39 (GPU nondeterminism floor)"])
R["gpu_resume"] = rs
R["end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
save()
print(json.dumps(rs), flush=True)
