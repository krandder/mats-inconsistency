"""E18b runner (PREREG-B amendment 32): single-pass LLM answers (no reasoning) on the zero-mean matched pairs.
Arms (model, framing, samples):
  claude-haiku-4-5, claude-sonnet-4-6, claude-opus-4-8 through `claude -p` with thinking off (MAX_THINKING_TOKENS=0 and
  alwaysThinkingEnabled false), output capped at 16 tokens, and SYS appended to the system prompt;
  gpt-5.6-sol through `codex-any exec` at reasoning effort "none", tools disabled, SYS prepended to the prompt;
  Qwen/Qwen3-1.7B locally (CPU, non-thinking chat template, greedy decoding of up to 8 tokens, SYS as the system message).
Each call is appended to results/e18b_calls.jsonl; resumable; 1 s between API calls; an arm stops after 10 consecutive failures.
The answer is the number the response begins with (E18's addendum-3 rule); anything else is non-compliant and retried up to 3 times.
usage: python e18b_call.py ARM
"""
import json, os, re, subprocess, sys, time
import numpy as np

SYS = "Your entire reply must be one number with two decimals, written immediately, with no words, no working and no explanation."
ARMS = {f'{f}_{m}': (m, fr, 2 if fr == 'stated' and m != 'qwen17' else 1)
        for m in ('haiku45', 'sonnet46', 'opus48', 'gpt56', 'qwen17') for f, fr in (('S', 'stated'), ('U', 'unstated'))}
CLAUDE = {'haiku45': 'claude-haiku-4-5', 'sonnet46': 'claude-sonnet-4-6', 'opus48': 'claude-opus-4-8'}
LEAD = re.compile(r'^\s*(?:\*\*)?\s*([-+]?\d+(?:\.\d+)?)')
LOG = 'results/e18b_calls.jsonl'
_qwen = None


def parse(text):
    m = LEAD.match(text or ''); return float(m.group(1)) if m else None


def ask_claude(model, prompt):
    env = dict(os.environ, MAX_THINKING_TOKENS='0', CLAUDE_CODE_MAX_OUTPUT_TOKENS='16'); t0 = time.time()
    p = subprocess.run(['claude', '-p', '--model', model, '--effort', 'low', '--tools', '', '--bare', '--no-session-persistence',
                        '--setting-sources', '', '--settings', '{"alwaysThinkingEnabled": false}', '--append-system-prompt', SYS,
                        '--output-format', 'json', prompt], cwd='/tmp/e18_claude', env=env, stdin=subprocess.DEVNULL,
                       capture_output=True, text=True, timeout=600)
    try:
        j = json.loads(p.stdout)
    except Exception:
        return dict(text='', ok=False, err=(p.stdout + p.stderr)[-300:], sec=time.time() - t0)
    u = j.get('usage') or {}
    return dict(text=j.get('result') or '', ok=(not j.get('is_error')) and j.get('stop_reason') == 'end_turn', cost=j.get('total_cost_usd'),
                in_tok=u.get('input_tokens'), out_tok=u.get('output_tokens'), think_tok=(u.get('output_tokens_details') or {}).get('thinking_tokens'),
                sec=time.time() - t0)


def ask_gpt(prompt):
    t0 = time.time()
    p = subprocess.run(['codex-any', 'exec', '-m', 'gpt-5.6-sol', '-c', 'model_reasoning_effort="none"', '--skip-git-repo-check',
                        '--disable', 'shell_tool', '--disable', 'code_mode_host', '--disable', 'browser_use', '--disable', 'computer_use',
                        '-C', '/tmp/e18_codex', SYS + '\n\n' + prompt], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=600)
    m = re.search(r'^model: (\S+)', p.stderr, re.M); tok = re.search(r'tokens used\s*\n?\s*([\d,]+)', p.stderr)
    used = bool(re.search(r'^(exec|tool|apply_patch)\b', p.stderr, re.M))
    return dict(text=p.stdout.strip(), ok=p.returncode == 0 and bool(m) and m.group(1) == 'gpt-5.6-sol' and not used,
                tokens=int(tok.group(1).replace(',', '')) if tok else None, used_tools=used, sec=time.time() - t0)


def ask_qwen(prompt):
    global _qwen
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if _qwen is None:
        torch.set_num_threads(2)
        tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B'); m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-1.7B', dtype=torch.bfloat16)
        m.eval(); _qwen = (tok, m)
    tok, m = _qwen; t0 = time.time()
    enc = tok.apply_chat_template([{'role': 'system', 'content': SYS}, {'role': 'user', 'content': prompt}], add_generation_prompt=True,
                                  enable_thinking=False, return_tensors='pt', return_dict=True)
    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=8, do_sample=False)
    return dict(text=tok.decode(out[0, enc['input_ids'].shape[1]:], skip_special_tokens=True), ok=True, sec=time.time() - t0)


def ask(model, prompt):
    if model in CLAUDE:
        return ask_claude(CLAUDE[model], prompt)
    return ask_gpt(prompt) if model == 'gpt56' else ask_qwen(prompt)


def done_keys():
    out = set()
    if os.path.exists(LOG):
        for line in open(LOG):
            r = json.loads(line)
            if r['ok'] and parse(r['text']) is not None:
                out.add((r['arm'], r['pair'], r['version'], r['sample']))
    return out


def main():
    arm = sys.argv[1]; model, framing, nsamp = ARMS[arm]
    items = json.load(open('results/e18b_items.json')); done = done_keys(); fails = 0
    for smp in range(nsamp):
        for i in np.random.default_rng(17 + smp).permutation(len(items)):
            it = items[i]; k = (arm, it['pair'], it['version'], smp)
            if k in done:
                continue
            for attempt in range(4):
                try:
                    r = ask(model, it[framing])
                except Exception as e:
                    r = dict(text='', ok=False, err=f'{type(e).__name__}: {str(e)[:200]}')
                ans = parse(r.get('text')); ok = bool(r.get('ok')) and ans is not None
                rec = dict(arm=arm, model=model, framing=framing, pair=it['pair'], version=it['version'], sample=smp, attempt=attempt,
                           answer=ans, ok=ok, text=r.get('text') or '', t=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                           **{kk: v for kk, v in r.items() if kk not in ('text', 'ok')}, harness_ok=bool(r.get('ok')))
                with open(LOG, 'a') as fh:
                    fh.write(json.dumps(rec) + '\n')
                if ok:
                    fails = 0; break
                fails += 1
                if fails >= 10:
                    print(arm, 'stopping after 10 consecutive failures', flush=True); return
                time.sleep(5)
            if model != 'qwen17':
                time.sleep(1)
    print(arm, 'done', flush=True)


if __name__ == '__main__':
    main()
