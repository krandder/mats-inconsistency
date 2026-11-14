| arm | params | seeds kept (excluded: gap) | test NLL gap | KL raw | KL_sys | nuisance share | sys: mean / var / shape (nats) | corr of per-episode KL_sys with main seeds | R vs base |
|---|---|---|---|---|---|---|---|---|---|
| base | 18732 | 4 (s11: 0.142) | 0.0087 | 0.0087 ± 0.0013 | 0.0056 ± 0.0005 | 35% | 0.0020 / 0.0015 / 0.0021 | 0.93 | +0.00 |
|  | per seed KL_sys | 0.0068, 0.0047, 0.0048, 0.0064 | | | | | | | |
| h3 | 20828 | 4 (s10: 0.142) | 0.0173 | 0.0177 ± 0.0051 | 0.0081 ± 0.0017 | 55% | 0.0029 / 0.0029 / 0.0022 | 0.75 | -0.43 |
|  | per seed KL_sys | 0.0118, 0.0045, 0.0099, 0.0061 | | | | | | | |
| h4 | 22924 | 5 (none) | 0.0095 | 0.0100 ± 0.0020 | 0.0050 ± 0.0006 | 49% | 0.0017 / 0.0015 / 0.0018 | 0.89 | +0.11 |
|  | per seed KL_sys | 0.0037, 0.0035, 0.0063, 0.0057, 0.0060 | | | | | | | |
| w64 | 22924 | 5 (none) | 0.0080 | 0.0086 ± 0.0011 | 0.0050 ± 0.0002 | 42% | 0.0016 / 0.0013 / 0.0020 | 0.93 | +0.12 |
|  | per seed KL_sys | 0.0045, 0.0045, 0.0048, 0.0057, 0.0053 | | | | | | | |
| L3 | 27276 | 5 (none) | 0.0047 | 0.0051 ± 0.0006 | 0.0031 ± 0.0003 | 41% | 0.0009 / 0.0007 / 0.0015 | 0.87 | +0.46 |
|  | per seed KL_sys | 0.0030, 0.0039, 0.0032, 0.0021, 0.0031 | | | | | | | |

R: base +0.00, h3 -0.43, h4 +0.11, w64 +0.12, L3 +0.46; Welch one-sided p (h4 < base) = 0.242
(i) two-head bottleneck supported: False
(ii) head count, not capacity: False
(iii) capacity, not head count: False
(iv) head count not the limit (R_h4 < 0.15): True ; then capacity reductions w64 / L3: +0.12 / +0.46
Reference: CPU-trained main seeds 0-4 KL_sys mean 0.0058
