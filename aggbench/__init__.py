"""aggbench: shared stage-1 harness for learning to aggregate forecasts from overlapping sources.

Environments with exact Bayes ceilings, non-learned baselines, metrics with bootstrap CIs,
controls for learned aggregators and a results schema. See aggbench/README.md.
"""
__version__ = "0.1.0"

from .envs import PRESETS, VISIBILITY, Batch, View, make_env  # noqa: E402,F401
from .splits import episode_seeds, verify_disjoint  # noqa: E402,F401
