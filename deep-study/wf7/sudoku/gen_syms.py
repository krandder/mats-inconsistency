"""gen_syms.py -- world side: store the generator's cell symmetries (used only to augment real puzzle data)."""
import numpy as np, oracle as O
for K in (4, 6):
    s = O.cell_symmetries(K); np.save(f"data/k{K}_syms.npy", s); print(K, s.shape)
