2026-09-28T08:54:53Z worker2 claims M9(b): confirm wealth-net prices LFSR/CA110 at 60k; FLOPs-to-Brier net vs soup VI. Files: study/M9/b/ , deliverable part study/M9/M9-b.md
2026-09-28T08:54:55Z w3: claim item (c) K/Q sweep {1,4,16,64} in M6's layer, fixed total params and fixed trader size. Output under M9/w3/.
2026-09-28T08:54:55Z w1 CLAIM item (a): lottery fit 1-(1-pi_f)^{K/Q} per family on logical-soup runs/phase3, K/Q prediction for LFSR/CA, lottery-ticket link. Outputs under M9/w1/.
2026-09-28T08:55:45Z w3: PREREG written (M9/w3/PREREG.md); 42 CPU runs (7 K configs x arms A,B x seeds 1-3), 3 concurrent.
2026-09-28T08:57:44Z w1: PREREG written (M9/w1/PREREG.md); CPU-only fits on logical-soup runs/phase3, no GPU.
2026-09-28T09:00:16Z worker2 DONE M9(b): M9/M9-b.md; data b/{final_brier,compute_to_brier}_summary.csv, b/flops_to_brier.png. Net prices LFSR/CA110 (WN 3/3 at 31.1k; 60k never reached). Net needs 10^9.0 (CA110) / 10^10.0-10.5 (LFSR) FLOPs/q vs soup max 10^7.54 VI/q. 1 CPU job, no GPU.
2026-09-28T09:09:45Z w3: local box load 45-55 (>40 gate), moved the 42 CPU runs to G1 job m9-w3-kq-sweep (CPU-only on the VM, --hours 1, reserve ~$1.16 of M9's $4). Results -> M9/w3/results/vm/.
2026-09-28T09:14:23Z w1: fits done (lottery_fit.csv/json/png, streams_check.csv/json, predictions.csv/json in M9/w1/); writing M9/w1/M9-a.md
2026-09-28T09:15:05Z w1 DONE M9(a): M9/M9-a.md; lottery R2 0.94/0.92/0.94/0.79 (periodic/prime/and/or), shuffle p=0.001; CA30 unlearnable (needs age 32), LFSR 4-sparse parity, CA110 needs 2 reads -> K/Q>=1.4k. CPU only, no GPU.
2026-09-28T09:32:46Z w3 DONE M9(c): M9/M9-c.md. K/Q law absent in the net (trend=false all 4 series; priced share flat in K; FT K has no effect excl. LFSR16); K=64 raises noise excess 0.002->0.007 in 3/3 seeds, all series. GPU $0.30 audited (job m9-w3-kq-sweep, VM deleted).
2026-09-28T09:36:00Z lead DONE M9: merged M9/M9.md; reran summarize.py, analyze_b.py, fit_lottery.py, predict.py in a clean /tmp copy, all outputs byte-identical; fixed M9-a item 5 + MAE range, M9-c LFSR priced wording. Stop rule: K/Q law absent in the net. GPU $0.30.
