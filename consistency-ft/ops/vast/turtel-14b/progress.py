#!/usr/bin/env python3
"""Print new training/validation metrics as the sync loop pulls them (mirrors data/vast-wave1/progress.py).

Usage: progress.py <lane_dir>   (exits when a done marker is synced)
"""
import json
import pathlib
import sys
import time


def main(argv):
    lane = pathlib.Path(argv[1])
    run = lane / 'results' / 'turtel-14b-grpo'
    seen_steps = set()
    seen_val = set()
    seen_status = set()
    while True:
        metrics = run / 'metrics.jsonl'
        if metrics.exists():
            for line in metrics.read_text().splitlines():
                try:
                    m = json.loads(line)
                except json.JSONDecodeError:
                    continue
                step = m.get('step')
                if step in seen_steps or step is None or step % 10:
                    continue
                seen_steps.add(step)
                reward = m.get('rewards/brier_with_guardrails/mean', m.get('reward'))
                print(f"step {step}: loss {m.get('loss')} reward {reward} kl {m.get('kl')} "
                      f"completion_len {m.get('completions/mean_length')} clipped {m.get('completions/clipped_ratio')}", flush=True)
        val = run / 'validation' / 'metrics.jsonl'
        if val.exists():
            for line in val.read_text().splitlines():
                try:
                    v = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if v.get('step') in seen_val:
                    continue
                seen_val.add(v.get('step'))
                print(f"VALIDATION step {v.get('step')}: soft_brier {v.get('soft_brier'):.4f} parse_rate {v.get('parse_rate'):.3f} "
                      f"extreme {v.get('extremization_rate')} n {v.get('n')}", flush=True)
        status = run / 'status.txt'
        if status.exists():
            for line in status.read_text().splitlines():
                if line not in seen_status:
                    seen_status.add(line)
                    print('STATUS ' + line, flush=True)
                    if any(m in line for m in ('RUN_DONE', 'RUN_FAILED', 'TRAINER EXIT=')):
                        test = run / 'test' / 'metrics.json'
                        if test.exists():
                            t = json.loads(test.read_text())
                            print(f"TEST soft_brier {t.get('soft_brier')} parse_rate {t.get('parse_rate')} n {t.get('n')} "
                                  f"comparators {t.get('comparators_same_rows')}", flush=True)
                        return 0
        time.sleep(60)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
