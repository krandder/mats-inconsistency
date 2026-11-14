# lm.py - one frozen language model, optional LoRA writers, batched sampling and scoring.
# sample(): K rollouts per prompt at a temperature, full distribution (no top-k/top-p), returns the tokens and the
#           WRITER's own log-probability of each rollout (under the distribution it was sampled from).
# score():  teacher-forced log-probability of given demonstrations under (adapter, prompt, temperature).
# A rollout ends when a stop token is drawn (any token containing a newline, or a special token). The stop token is
# not part of the demonstration; ending is charged the total probability of the stop set.
import torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache

class LM:
    def __init__(self, path, dtype=torch.bfloat16):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).cuda().eval()
        self.base = self.model
        V = self.model.get_input_embeddings().weight.shape[0]
        n = len(self.tok)
        strs = self.tok.batch_decode([[i] for i in range(n)])
        stop = torch.zeros(V, dtype=torch.bool)
        for i, s in enumerate(strs):
            if "\n" in s:
                stop[i] = True
        for i in self.tok.all_special_ids:
            stop[i] = True
        stop[n:] = True
        self.stop = stop.cuda()
        self.stop_idx = torch.nonzero(self.stop).squeeze(1)
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        self.peft = None
        self.calls = 0  # number of token positions run through the network (compute accounting)

    def load_adapter(self, path, name):
        from peft import PeftModel
        if self.peft is None:
            self.peft = PeftModel.from_pretrained(self.base, path, adapter_name=name)
            self.peft.eval()
        else:
            self.peft.load_adapter(path, adapter_name=name)

    def enc(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def dec(self, ids):
        return self.tok.decode(ids)

    def _fwd(self, adapter, **kw):
        self.calls += kw["input_ids"].numel()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self.peft is None:
                assert adapter is None
                return self.base(**kw)
            if adapter is None:
                with self.peft.disable_adapter():
                    return self.peft(**kw)
            self.peft.set_adapter(adapter)
            return self.peft(**kw)

    def _prefix(self, adapter, prompts, K, repeat=True):
        P = len(prompts); L = max(len(p) for p in prompts)
        ids = torch.full((P, L), self.pad, dtype=torch.long); att = torch.zeros((P, L), dtype=torch.long)
        for i, p in enumerate(prompts):
            ids[i, L - len(p):] = torch.tensor(p); att[i, L - len(p):] = 1
        ids = ids.cuda(); att = att.cuda()
        out = self._fwd(adapter, input_ids=ids, attention_mask=att, use_cache=True)
        past = out.past_key_values
        if repeat:
            past.batch_repeat_interleave(K)
        logits = out.logits[:, -1].repeat_interleave(K, 0)
        att = att.repeat_interleave(K, 0)
        return past, logits, att

    @torch.no_grad()
    def sample(self, prompts, K, temp, max_new, gen, adapter=None, greedy=False):
        """K rollouts for each prompt. prompts: list of P token-id lists.
        Returns a list (per prompt) of K dicts {ids (without the stop token), logq, done}.
        The K rollouts of one prompt are packed into ONE sequence after the prompt (the prompt's cache is stored once):
        at step t the K new tokens are fed together, all at position L + t, and a 4-D mask lets the token of rollout k
        see the prompt and rollout k's own earlier tokens only. score() uses the ordinary path, so the equality of
        logq here with score() there (checked in prof.py and in analyze.py) tests this packing."""
        P = len(prompts); L = max(len(p) for p in prompts)
        ids = torch.full((P, L), self.pad, dtype=torch.long); att = torch.zeros((P, L), dtype=torch.long)
        for i, p in enumerate(prompts):
            ids[i, L - len(p):] = torch.tensor(p); att[i, L - len(p):] = 1
        ids = ids.cuda(); att = att.cuda()
        out = self._fwd(adapter, input_ids=ids, attention_mask=att, use_cache=True)
        past = out.past_key_values
        V = out.logits.shape[-1]
        lp = torch.log_softmax(out.logits[:, -1].float() / temp, -1)[:, None, :].expand(P, K, V).reshape(P * K, V)
        del out
        dt = self.base.dtype
        allow = torch.zeros((P, K, L + K * max_new), dtype=torch.bool, device="cuda")
        allow[:, :, :L] = att.bool()[:, None, :]
        allow[:, :, L:] = torch.eye(K, dtype=torch.bool, device="cuda").repeat(1, max_new)[None]
        amask = torch.zeros((P, 1, K, L + K * max_new), dtype=dt, device="cuda").masked_fill(~allow[:, None], torch.finfo(dt).min)
        del allow
        B = P * K
        toks = torch.full((B, max_new), -1, dtype=torch.long, device="cuda")
        logq = torch.zeros(B, device="cuda"); done = torch.zeros(B, dtype=torch.bool, device="cuda")
        for t in range(max_new):
            if greedy:
                nxt = lp.argmax(-1)
            else:
                nxt = torch.multinomial(lp.exp(), 1, generator=gen).squeeze(1)
            lq = lp.gather(1, nxt[:, None]).squeeze(1)
            isstop = self.stop[nxt]
            lq = torch.where(isstop, torch.logsumexp(lp[:, self.stop_idx], -1), lq)
            act = ~done
            logq[act] += lq[act]
            rec = act & ~isstop
            toks[rec, t] = nxt[rec]
            done |= isstop
            if bool(done.all()) or t == max_new - 1:
                break
            inp = torch.where(done, torch.full_like(nxt, self.pad), nxt).view(P, K)
            pos = torch.full((P, K), L + t, dtype=torch.long, device="cuda")
            out = self._fwd(adapter, input_ids=inp, attention_mask=amask[:, :, :, :L + K * (t + 1)], position_ids=pos,
                            past_key_values=past, use_cache=True)
            past = out.past_key_values
            lp = torch.log_softmax(out.logits.float().view(B, V) / temp, -1)
        toks = toks.cpu().tolist(); logq = logq.cpu().tolist(); done = done.cpu().tolist()
        res = []
        for i in range(len(prompts)):
            row = []
            for k in range(K):
                b = i * K + k
                row.append({"ids": [x for x in toks[b] if x >= 0], "logq": logq[b], "done": done[b]})
            res.append(row)
        return res

    @torch.no_grad()
    def score(self, prompts, demos, temp=1.0, adapter=None):
        """prompts: list of P token-id lists; demos: list of P lists, each with the same number K of token-id lists
        (without the stop token). Returns P lists of K floats: the log-probability, under softmax(logits / temp), of
        writing exactly these tokens and then stopping (total probability of the stop set)."""
        K = len(demos[0])
        assert all(len(d) == K for d in demos)
        past, logits, att = self._prefix(adapter, prompts, K)
        flat = [d for row in demos for d in row]
        B = len(flat); T = max(1, max(len(d) for d in flat))
        x = torch.full((B, T), self.pad, dtype=torch.long); m = torch.zeros((B, T), dtype=torch.bool)
        n = torch.tensor([len(d) for d in flat])
        for b, d in enumerate(flat):
            x[b, :len(d)] = torch.tensor(d); m[b, :len(d)] = True
        x = x.cuda(); m = m.cuda(); n = n.cuda()
        lp0 = torch.log_softmax(logits.float() / temp, -1)
        tot = torch.where(n > 0, lp0.gather(1, x[:, :1]).squeeze(1), torch.logsumexp(lp0[:, self.stop_idx], -1))
        del lp0
        att2 = torch.cat([att, torch.ones((B, T), dtype=torch.long, device="cuda")], 1)
        out = self._fwd(adapter, input_ids=x, attention_mask=att2, past_key_values=past, use_cache=True)
        for a in range(0, B, 16):  # chunked: the logits are large (rows x tokens x vocabulary)
            lp = torch.log_softmax(out.logits[a:a + 16].float() / temp, -1)
            xa, ma, na = x[a:a + 16], m[a:a + 16], n[a:a + 16]
            if T > 1:
                tot[a:a + 16] += (lp[:, :-1].gather(2, xa[:, 1:, None]).squeeze(2) * ma[:, 1:]).sum(1)
            last = lp[torch.arange(lp.shape[0], device="cuda"), (na - 1).clamp(min=0)]
            tot[a:a + 16] += torch.where(na > 0, torch.logsumexp(last[:, self.stop_idx], -1), torch.zeros_like(tot[a:a + 16]))
        tot = tot.cpu().tolist()
        return [tot[i * K:(i + 1) * K] for i in range(len(prompts))]
