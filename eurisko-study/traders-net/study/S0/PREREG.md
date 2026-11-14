# PREREG, S0 (index)

S0 is substrate and measurement code, not an experiment. Each worker pre-registered its own
validation runs before running them; the lead ran no new experiment (only reran the V1 demo after
a leak fix, see SUBSTRATE.md section 3).
- `PREREG-w1.md`: prequential P4/P16 vs multi-epoch ME, seeds 1-3 (amendment 1 before results).
- `PREREG-w2.md`: stream audits A0/A1/A2, seeds 1-3.
- `PREREG-w3.md`: LI family checks + logistic/net baseline, seeds 0-2.
- `PREREG-w4.md`: panel toy values, Gini-null calibration, seeds 0-2.
Stop rule for the merge: `check_substrate.py` PASS (tests green and every quoted number re-read
from its file).
