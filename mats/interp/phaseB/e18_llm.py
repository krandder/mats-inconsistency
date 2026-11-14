"""E18 model access (PREREG-B amendment 25). Every call goes through a sanctioned fleet harness and is logged by the caller.
- Claude models: the Claude Code CLI in print mode (`claude -p`, bare, no tools, no session persistence, no setting sources).
  Direct Messages API calls through the proxy were refused upstream (429) in the plumbing check, while the CLI works.
- GPT-6: `codex-any exec` (the fleet's Codex path), with the reasoning effort set per arm.
Sampling parameters are not set (current models do not accept them); effort is the only knob.
"""
import json, os, re, subprocess, time

CLAUDE = {'haiku': ('claude-haiku-5-5', 'low'), 'sonnet': ('claude-sonnet-5-5', 'low'), 'opus': ('claude-opus-5-5', 'low'),
          'opus_high': ('claude-opus-5-5', 'high')}
NUM = re.compile(r'[-+]?\d+(?:\.\d+)?')


def ask_claude(key, prompt, workdir='/tmp/e18_claude'):
    os.makedirs(workdir, exist_ok=True)
    model, effort = CLAUDE[key]; t0 = time.time()
    p = subprocess.run(['claude', '-p', '--model', model, '--effort', effort, '--tools', '', '--bare', '--no-session-persistence',
                        '--setting-sources', '', '--output-format', 'json', prompt], cwd=workdir, stdin=subprocess.DEVNULL,
                       capture_output=True, text=True, timeout=1800)
    try:
        j = json.loads(p.stdout)
    except Exception:
        return dict(text='', ok=False, err=(p.stdout + p.stderr)[-400:], sec=time.time() - t0, model=model, effort=effort)
    u = j.get('usage', {}) or {}
    mu = (j.get('modelUsage') or {}).get(model, {})
    return dict(text=j.get('result', '') or '', ok=not j.get('is_error', False) and j.get('stop_reason') == 'end_turn',
                stop=j.get('stop_reason'), cost=j.get('total_cost_usd'), in_tok=u.get('input_tokens'), out_tok=u.get('output_tokens'),
                think_tok=(u.get('output_tokens_details') or {}).get('thinking_tokens'), model=mu.get('canonicalModel', model),
                effort=effort, sec=time.time() - t0)


def ask_gpt(prompt, effort='low', workdir='/tmp/e18_codex'):
    os.makedirs(workdir, exist_ok=True); t0 = time.time()
    p = subprocess.run(['codex-any', 'exec', '-m', 'gpt-6-astra', '-c', f'model_reasoning_effort="{effort}"', '--skip-git-repo-check',
                        '--disable', 'shell_tool', '--disable', 'code_mode_host', '--disable', 'browser_use', '--disable', 'computer_use',
                        '-C', workdir, prompt], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=1800)
    m = re.search(r'^model: (\S+)', p.stderr, re.M)
    tok = re.search(r'tokens used\s*\n?\s*([\d,]+)', p.stderr)
    used_tools = bool(re.search(r'^(exec|tool|apply_patch)\b', p.stderr, re.M))
    return dict(text=p.stdout.strip(), ok=p.returncode == 0 and bool(m) and m.group(1) == 'gpt-6-astra', model=m.group(1) if m else None,
                tokens=int(tok.group(1).replace(',', '')) if tok else None, used_tools=used_tools, effort=effort, sec=time.time() - t0)


LEAD = re.compile(r'^\s*(?:\*\*)?\s*([-+]?\d+(?:\.\d+)?)')


LAST = re.compile(r'^\s*(?:\*\*)?\s*([-+]?\d+(?:\.\d+)?)\s*(?:\*\*)?\s*\.?\s*$')


def parse(text):
    """Addenda 3-4: the number the response begins with (optionally bold); otherwise the number that alone makes up its last
    non-empty line (optionally bold); otherwise None (non-compliant)."""
    m = LEAD.match(text or '')
    if m:
        return float(m.group(1))
    lines = [l for l in (text or '').splitlines() if l.strip()]
    m = LAST.match(lines[-1]) if lines else None
    return float(m.group(1)) if m else None


def ask(key, prompt):
    return ask_gpt(prompt) if key == 'gpt' else ask_claude(key, prompt)
