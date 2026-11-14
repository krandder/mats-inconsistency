#!/bin/bash
# CPU smoke of the p3 options: tbptt/random pick, prog+covsup, econ rr, checkpoint+resume, eval reuse, dtrecall, reeval
cd ~; PY=python3
O=~/sm; mkdir -p $O; rm -f $O/*
S="--iters 6 --bs 4 --D 4 --width 16 --evalL 32 --device cpu"
$PY econ2.py --mode bptt --stream cov --ds 1 --covsup 1 --tbptt 1 --train_pick random $S --out $O/tb1r > $O/tb1r.stdout 2>&1 || echo FAIL tb1r
$PY econ2.py --mode bptt --stream cov --prog 1 --covsup 1 $S --out $O/pc > $O/pc.stdout 2>&1 || echo FAIL pc
$PY econ2.py --mode econ --stream cov --voc 1 --train_pick roundrobin --auctionL none $S --out $O/err > $O/err.stdout 2>&1 || echo FAIL err
# resume equivalence: A uninterrupted, B checkpointed every iteration then "crashed" before the final save
$PY econ2.py --mode econ --stream cov --voc 1 $S --evalL 32 --ckpt_every 0 --out $O/A > /dev/null 2>&1 || echo FAIL A
$PY econ2.py --mode econ --stream cov --voc 1 $S --evalL 32 --ckpt_every 0.000001 --out $O/B > /dev/null 2>&1 || echo FAIL B
rm $O/B.pt $O/B.json
$PY econ2.py --mode econ --stream cov --voc 1 $S --evalL 32 --ckpt_every 0.000001 --out $O/B > $O/B.stdout 2>&1 || echo FAIL B2
$PY econ2.py --mode bptt --stream cov --ds 1 --covsup 1 --tbptt 2 $S --ckpt_every 0 --out $O/C > /dev/null 2>&1 || echo FAIL C
$PY econ2.py --mode bptt --stream cov --ds 1 --covsup 1 --tbptt 2 $S --ckpt_every 0.000001 --out $O/Dd > /dev/null 2>&1 || echo FAIL D
rm $O/Dd.pt $O/Dd.json
$PY econ2.py --mode bptt --stream cov --ds 1 --covsup 1 --tbptt 2 $S --ckpt_every 0.000001 --out $O/Dd > $O/Dd.stdout 2>&1 || echo FAIL D2
# eval reuse on --load (same out): should skip evals and halting
$PY econ2.py --mode econ --stream cov --voc 1 $S --load $O/A --out $O/A > $O/Aload.stdout 2>&1 || echo FAIL Aload
$PY dtrecall.py --width 16 --iters 12 --bs 4 --evalL 32 --device cpu --ckpt_every 0 --out $O/dtA > /dev/null 2>&1 || echo FAIL dtA
$PY dtrecall.py --width 16 --iters 12 --bs 4 --evalL 32 --device cpu --ckpt_every 0.000001 --out $O/dtB > /dev/null 2>&1 || echo FAIL dtB
rm $O/dtB.pt $O/dtB.json
$PY dtrecall.py --width 16 --iters 12 --bs 4 --evalL 32 --device cpu --ckpt_every 0.000001 --out $O/dtB > $O/dtB.stdout 2>&1 || echo FAIL dtB2
$PY dtrecall.py --width 16 --time_budget 3 --bs 4 --evalL 32 --device cpu --out $O/dtT > $O/dtT.stdout 2>&1 || echo FAIL dtT
$PY reeval.py --ckpt $O/A.pt --out $O/rA.json --L 64 --n 4 --chunk 2 --device cpu > $O/rA.stdout 2>&1 || echo FAIL rA
$PY reeval.py --ckpt $O/C.pt --out $O/rC.json --L 64 --n 4 --chunk 2 --device cpu > $O/rC.stdout 2>&1 || echo FAIL rC
$PY reeval.py --ckpt $O/dtA.pt --out $O/rdt.json --L 64 --n 4 --chunk 2 --device cpu > $O/rdt.stdout 2>&1 || echo FAIL rdt
$PY - <<'PY'
import torch
O = __import__("os").path.expanduser("~/sm/")
def w(p, k):
    ck = torch.load(O + p, map_location="cpu", weights_only=False)[k]
    return ck if isinstance(ck, list) else [ck]
for a, b, k in [("A.pt", "B.pt", "traders"), ("C.pt", "Dd.pt", "traders"), ("dtA.pt", "dtB.pt", "net")]:
    d = max((x[n] - y[n]).abs().max().item() for x, y in zip(w(a, k), w(b, k)) for n in x)
    print("RESUME", a, b, "maxdiff", d)
PY
grep -h resumed_at_it $O/B.stdout $O/Dd.stdout $O/dtB.stdout | head -3
echo "Aload lines: $(wc -l < $O/Aload.stdout)"; head -c 300 $O/Aload.stdout
# the same resume on the GPU (the RNG-state bug of phase 1): must resume, not crash
G="--iters 6 --bs 4 --D 4 --width 16 --evalL 32 --device cuda --ckpt_every 0.000001"
$PY econ2.py --mode econ --stream cov --voc 1 $G --out $O/Bg > /dev/null 2>&1 || echo FAIL Bg
rm $O/Bg.pt $O/Bg.json; $PY econ2.py --mode econ --stream cov --voc 1 $G --out $O/Bg > $O/Bg.stdout 2>&1 || echo FAIL Bg2
$PY dtrecall.py --width 16 --iters 12 --bs 4 --evalL 32 --device cuda --ckpt_every 0.000001 --out $O/dtg > /dev/null 2>&1 || echo FAIL dtg
rm $O/dtg.pt $O/dtg.json; $PY dtrecall.py --width 16 --iters 12 --bs 4 --evalL 32 --device cuda --ckpt_every 0.000001 --out $O/dtg > $O/dtg.stdout 2>&1 || echo FAIL dtg2
grep -h resumed_at_it $O/Bg.stdout $O/dtg.stdout
echo SMOKE_DONE
