## 4. Registered predictions (`PREREG.md` sections 6 and 9; all CHECKED; sections 2e, 2f and the mechanism are post hoc)

| # | Registered item | Outcome | My prediction |
|---|---|---|---|
| H1 (i) | pairs term improves accuracy at n = 64, 256 | no: +5 % / -7 %*, -3 % / +4 % at the validation-chosen weight | NULL or MINOR: right |
| H1 (ii-o) | order-variance term, same | no gain in both worlds: +15 % / -2 %, +4 % / +6 % | NULL or MINOR: right |
| H1 (ii-s) | score-anchored term, same | MINOR: +17 %* / +6 %, +15 %* / +3 %; largest at fewest data in direction | MAJOR: wrong |
| H1 (iii) | over-compute control | NULL: +2 % / +9 %*, -8 % / +3 %* | NULL: right |
| H1 n = 0 | the term alone, no real outcome | (ii-s) 0.048 / 0.055 against 0.031 / 0.032 for 20,480 real vectors; (i), (ii-o) = the constant; (iii) diverges | (ii-s) within 0.02: right in world 101, wrong in 102 (0.023); the others: right |
| H2 | D and V_ord fall from K = 1 to K = 4 (n = 20,480) | no: D rises 0.043 / 0.063 to 0.058 / 0.111; V_ord 0.87 / 0.75 to 0.74 / 0.89 | falls: wrong; "mostly noise floor": right |
| H3 | (i) cuts D, (ii-o) cuts V_ord by 30 % or more at every n | yes: D to 2-4 % of its value; V_ord to 12-55 % | yes: right |
| H4 | M's D below the amplified predictor's at K = 1, above it at the top K | below at both K (0.005 / 0.007 against 0.04-0.19) | half wrong |
| H5 | drag at n = 64 and 1,024 in both worlds for (i) or (ii-o), weight 10 | 3 of 4 cells (not world 101 at n = 64); every cell at n >= 1,024 | "under half of the cells": wrong |
| H5 anchor, built | (ii-s) shows no drag in those cells; (iii-sym) 5 % worse than (iii) | holds at weights 0.1, 1, fails at 10; (iii-sym) 14-24 % worse from n = 1,024 | first wrong at weight 10; second right |
| wrong score | (ii-s), M's samples worse than data-only on checks 12, 25 at n >= 4,096 | by 0.000-0.007 at weights 1, 10 and for M's samples; not at 0.1 | right in direction, small |
