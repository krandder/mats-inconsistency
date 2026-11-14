s = open('prop.py').read()
# 1. selector matrices: built once per (T, H, w) on the CPU, cached on device
old = """            sel = torch.zeros(T, T, H, T, T, H, dtype=self.dt, device=self.dev)        # x_a = (t, w, h)
            for t_ in range(T):
                for h_ in range(H):
                    sel[t_, :, h_, t_, w, h_] = 1.0
            sel = sel.reshape(n_s, n_s)
"""
assert old in s
s = s.replace(old, "            sel = self._sel(T, H, w)                                                   # x_a = (t, w, h)\n")
old = "    # ------------------------------------------------------------ source registry\n"
assert old in s
s = s.replace(old, '''    def _sel(self, T, H, w):
        key = (T, H, w, self.dt, str(self.dev))
        if not hasattr(self, '_selc'):
            self._selc = {}
        if key not in self._selc:
            sel = torch.zeros(T, T, H, T, T, H, dtype=self.dt)
            for t_ in range(T):
                for h_ in range(H):
                    sel[t_, :, h_, t_, w, h_] = 1.0
            self._selc[key] = sel.reshape(T * T * H, T * T * H).to(self.dev)
        return self._selc[key]

''' + old)
# 2. debug dict only when asked
s = s.replace("        self.dbg = dict(ms=ms, Pss=Pss)\n", "        self.dbg = dict(ms=ms, Pss=Pss) if self.debug else {}\n")
s = s.replace("        self.dbg.update(Ea=Ea, Paa=Paa, ml=ml, Pll=Pll)\n", "        if self.debug: self.dbg.update(Ea=Ea, Paa=Paa, ml=ml, Pll=Pll)\n")
s = s.replace('                self.dbg["Tk"] = Tk + Tk.permute(0, 4, 5, 6, 1, 2, 3)\n', '                if self.debug: self.dbg["Tk"] = Tk + Tk.permute(0, 4, 5, 6, 1, 2, 3)\n')
s = s.replace("        self.dbg.update(mo=mo, Po=Po)\n", "        if self.debug: self.dbg.update(mo=mo, Po=Po)\n")
s = s.replace("        self.psd = True\n", "        self.psd = True\n        self.debug = False\n        self.checkpoint = True\n", 1)
# 3. per-block gradient checkpointing (registry state passed in/out so recomputation is idempotent)
old = """        for blk in blocks:
            p = lambda t: t.to(dt)
            y = self.layernorm(self.res_block(st, T, d), p(blk.ln1.weight), p(blk.ln1.bias))
            att = self.attention(y, p(blk.qkv.weight), p(blk.qkv.bias), p(blk.o.weight), p(blk.o.bias), H, allow)
            st = self.add_branch(st, att)
            if want_points: points.append(self.point_stats(st, N))
            y2 = self.layernorm(self.res_block(st, T, d), p(blk.ln2.weight), p(blk.ln2.bias))
            h = self.linear_tok(y2, p(blk.fc1.weight), p(blk.fc1.bias))
            g = self.gelu_op(h)
            mm = self.linear_tok(g, p(blk.fc2.weight), p(blk.fc2.bias))
            st = self.add_branch(st, mm)
            if want_points: points.append(self.point_stats(st, N))
"""
assert old in s
new = """        def block_fn(blk, m_, P_, A_, c3_, c4_):
            self.c3 = [c3_] if c3_.shape[1] else []
            self.c4 = [c4_] if c4_.shape[1] else []
            self.J = c3_.shape[1]
            st_ = (m_, P_, A_)
            p = lambda t: t.to(dt)
            y = self.layernorm(self.res_block(st_, T, d), p(blk.ln1.weight), p(blk.ln1.bias))
            att = self.attention(y, p(blk.qkv.weight), p(blk.qkv.bias), p(blk.o.weight), p(blk.o.bias), H, allow)
            st1 = self.add_branch(st_, att)
            y2 = self.layernorm(self.res_block(st1, T, d), p(blk.ln2.weight), p(blk.ln2.bias))
            h = self.linear_tok(y2, p(blk.fc1.weight), p(blk.fc1.bias))
            g = self.gelu_op(h)
            mm = self.linear_tok(g, p(blk.fc2.weight), p(blk.fc2.bias))
            st2 = self.add_branch(st1, mm)
            c3n, c4n = self.cs()
            return st1[0], st1[1], self.pad(st1[2]), st2[0], st2[1], self.pad(st2[2]), c3n, c4n

        for blk in blocks:
            c3_, c4_ = self.cs()
            args = (st[0], st[1], self.pad(st[2]), c3_, c4_)
            if self.checkpoint and torch.is_grad_enabled():
                out_ = torch.utils.checkpoint.checkpoint(block_fn, blk, *args, use_reentrant=False)
            else:
                out_ = block_fn(blk, *args)
            m1, P1, A1, m2, P2, A2, c3n, c4n = out_
            self.c3 = [c3n] if c3n.shape[1] else []
            self.c4 = [c4n] if c4n.shape[1] else []
            self.J = c3n.shape[1]
            st1, st = (m1, P1, A1), (m2, P2, A2)
            if want_points:
                points.append(self.point_stats(st1, N)); points.append(self.point_stats(st, N))
"""
s = s.replace(old, new)
if 'import torch.utils.checkpoint' not in s:
    s = s.replace("import torch.nn.functional as Fn\n", "import torch.nn.functional as Fn\nimport torch.utils.checkpoint\n", 1)
open('prop.py', 'w').write(s)
print('ok')
