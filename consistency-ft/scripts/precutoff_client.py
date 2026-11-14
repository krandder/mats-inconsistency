"""Chat-completion client for the pre-cutoff baseline (OpenAI direct and OpenRouter).

Keys are read at run time from deploy-host's [removed] over ssh and kept in memory;
they are never printed or written to disk.  Every call returns text, usage and
dollar cost (OpenRouter reports cost; OpenAI cost is computed from list prices).
A model that the provider rejects raises instead of falling back.
"""

from __future__ import annotations

import json
import random
import subprocess
import threading
import time
import urllib.error
import urllib.request

# Model registry: name -> (provider, provider model id, extra request fields, $/M in, $/M out)
MODELS = {
    "o3": ("openai", "o3", {"reasoning_effort": "medium"}, 2.0, 8.0),
    # Google AI Studio free-tier key (no dollar spend); price = paid list price, for the record only.
    "gemini-2.5-pro": ("google", "gemini-2.5-pro", {"reasoning_effort": "medium"}, 1.25, 10.0),
    # OpenRouter is unusable while its account credits are exhausted (402); kept for the record.
    "deepseek-r1": ("openrouter", "deepseek/deepseek-r1", {}, 0.7, 2.5),
    # Alibaba Model Studio (intl), original April-2025 checkpoint; thinking mode needs streaming.
    "qwen3-235b-a22b": ("dashscope", "qwen3-235b-a22b", {"enable_thinking": True}, 0.7, 8.4),
}
URLS = {"openai": "https://api.openai.com/v1/chat/completions",
        "openrouter": "https://openrouter.ai/api/v1/chat/completions",
        "google": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "dashscope": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"}
ENV_FILES = {"openai": ("[removed]", "OPENAI_API_KEY"),
             "openrouter": ("[removed]", "OPENROUTER_API_KEY"),
             "google": ("[removed]", "GEMINI_API_KEY_FREE"),
             "dashscope": ("[removed]", "DASHSCOPE_API_KEY")}
_KEYS: dict[str, str] = {}
_LOCK = threading.Lock()


def key(provider: str) -> str:
    with _LOCK:
        for attempt in range(5):
            if provider in _KEYS:
                break
            path, var = ENV_FILES[provider]
            out = subprocess.run(["ssh", "deploy-host", f"set -a; . {path}; set +a; printf %s \"${var}\""],
                                 capture_output=True, text=True, timeout=60)
            if out.returncode == 0 and out.stdout.strip():
                _KEYS[provider] = out.stdout.strip()
            else:
                time.sleep(3 * (attempt + 1))
        if provider not in _KEYS:
            raise RuntimeError(f"could not load {var} from deploy-host {path}")
        return _KEYS[provider]


def _read_stream(r) -> dict:
    """Assemble an OpenAI-style SSE stream into one response dict."""
    text, reasoning, usage, model, finish, lps = [], [], {}, None, None, []
    for raw in r:
        line = raw.decode(errors="replace").strip()
        if not line.startswith("data:"):
            continue
        chunk = line[5:].strip()
        if chunk == "[DONE]":
            break
        ev = json.loads(chunk)
        model = ev.get("model") or model
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("content"):
                text.append(d["content"])
            if d.get("reasoning_content"):
                reasoning.append(d["reasoning_content"])
            lp = d.get("logprobs") or ch.get("logprobs")  # only when the request asks for logprobs
            if lp and lp.get("content"):
                lps.extend(lp["content"])
            finish = ch.get("finish_reason") or finish
    return {"choices": [{"message": {"content": "".join(text), "reasoning": "".join(reasoning), "logprobs": lps or None},
                         "finish_reason": finish}], "usage": usage, "model": model}


def chat(model: str, messages: list[dict], *, max_tokens: int = 16000, max_wait: float = 1800,
         extra: dict | None = None) -> dict:
    """extra: request fields that override the registry's (e.g. a per-call seed or temperature)."""
    provider, mid, reg_extra, pin, pout = MODELS[model]
    body = {"model": mid, "messages": messages, **reg_extra, **(extra or {})}
    stream = provider == "dashscope"
    if provider == "openai":
        body["max_completion_tokens"] = max_tokens
    elif provider in ("google", "dashscope"):
        body["max_tokens"] = max_tokens
        if stream:
            body.update(stream=True, stream_options={"include_usage": True})
    else:
        body.update(max_tokens=max_tokens, usage={"include": True},
                    provider={"allow_fallbacks": True, "require_parameters": False})
    data = json.dumps(body).encode()
    start, delay, errors = time.time(), 5.0, []
    while True:
        req = urllib.request.Request(URLS[provider], data=data, headers={
            "Authorization": "Bearer " + key(provider), "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                resp = _read_stream(r) if stream else json.loads(r.read())
            if "choices" not in resp:
                raise RuntimeError("no choices: " + json.dumps(resp)[:300])
            msg = resp["choices"][0]["message"]
            usage = resp.get("usage") or {}
            served = resp.get("model")
            cost = usage.get("cost")
            if cost is None:
                cost = usage.get("prompt_tokens", 0) * pin / 1e6 + usage.get("completion_tokens", 0) * pout / 1e6
            return {"text": msg.get("content"), "reasoning": msg.get("reasoning"), "usage": usage, "cost": cost,
                    "served_model": served, "provider": resp.get("provider", provider),
                    "finish_reason": resp["choices"][0].get("finish_reason"), "errors": errors,
                    "logprobs": msg.get("logprobs")}
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:500]
            errors.append({"status": e.code, "err": msg})
            if e.code == 402 or "insufficient" in msg.lower() or "quota" in msg.lower() and e.code != 429:
                raise RuntimeError(f"provider refused on credit/quota ({model}, {e.code}): {msg[:300]}")
            if e.code in (400, 401, 403, 404) and ("model" in msg.lower() and ("not" in msg.lower() or "invalid" in msg.lower())):
                raise RuntimeError(f"model rejected ({model}): {msg[:300]}")  # never substitute
            if e.code not in (408, 409, 429, 500, 502, 503, 504, 520, 524, 529):
                return {"text": None, "reasoning": None, "usage": {}, "cost": 0.0, "served_model": None,
                        "provider": provider, "finish_reason": None, "errors": errors}
        except Exception as e:  # network, timeout, bad JSON
            errors.append({"status": None, "err": repr(e)[:300]})
        if time.time() - start > max_wait:
            return {"text": None, "reasoning": None, "usage": {}, "cost": 0.0, "served_model": None,
                    "provider": provider, "finish_reason": None, "errors": errors}
        time.sleep(delay + random.uniform(0, delay / 2))
        delay = min(delay * 2, 120)
