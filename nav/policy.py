"""Policies that pick (stride, turn) for each gait cycle.

- `TabPFNNavPolicy`: two TabPFN regressors (stride, turn) that map the goal to the command.
- `BaselinePolicy`: a hand-written steer-to-bearing controller, for comparison.
- `dynamics.DynamicsPlanner` (in dynamics.py): TabPFN predicts the motion of candidate commands.

TabPFN is used only through its built-in `fit()` and `predict()` (rule in AGENTS.md): `fit()` stores
the examples as context and the pretrained weights are never changed.

All policies take a list of observed rows (one per robot, so many episodes run in lockstep) and
return an (n, 2) array.
"""

import math
import warnings

import numpy as np

from common import MAX_STEP, STRIDE_RANGE, TURN_RANGE
from hindsight import DEFAULT_GROUPS, LABELS, feature_columns

MODEL_VERSION = "v2"  # TabPFN v2 weights: not licence-gated, so no login is needed


def sample_context(table, rows, seed=0):
    """At most `rows` examples, spread evenly over goal distance and bearing."""
    if len(table) <= rows:
        return table
    bearing = np.arctan2(table["goal_sin"], table["goal_cos"])
    bins = (np.digitize(table["goal_dist"].to_numpy(), np.linspace(0, 1.1, 6)) * 10
            + np.digitize(bearing.to_numpy(), np.linspace(-math.pi, math.pi, 9)))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(table))
    per_bin = {}
    keep = []
    quota = max(1, rows // len(set(bins.tolist())))
    for i in order:
        if per_bin.get(bins[i], 0) < quota:
            per_bin[bins[i]] = per_bin.get(bins[i], 0) + 1
            keep.append(i)
    if len(keep) < rows:
        kept = set(keep)
        keep += [i for i in order if i not in kept][: rows - len(keep)]
    return table.iloc[sorted(keep[:rows])]


def fit_regressor(X, y, device="cpu", n_estimators=4, seed=0):
    """A pretrained TabPFN regressor with (X, y) as its context: TabPFN's own `fit()`, nothing else."""
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    model = TabPFNRegressor.create_default_for_version(
        ModelVersion(MODEL_VERSION), device=device, n_estimators=n_estimators, random_state=seed,
        fit_mode="fit_with_cache", ignore_pretraining_limits=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(np.asarray(X, np.float32), np.asarray(y, np.float32))
    return model


def predict(model, X):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.predict(np.asarray(X, np.float32))


class TabPFNNavPolicy:
    """goal (distance, bearing, horizon) -> (stride, turn), with `context` as TabPFN's examples."""

    kind = "goal"
    name = "tabpfn"
    labels = LABELS

    def __init__(self, context, groups=DEFAULT_GROUPS, device="cpu", n_estimators=4, seed=0):
        self.groups = tuple(groups)
        self.columns = feature_columns(self.groups, context)
        X = context[self.columns].to_numpy(np.float32)
        self.models = {label: fit_regressor(X, context[label], device, n_estimators, seed) for label in LABELS}

    def predict_labels(self, table):
        """{label: predictions} for the rows of an example table (for validation)."""
        X = table[self.columns].to_numpy(np.float32)
        return {label: predict(self.models[label], X) for label in LABELS}

    def act(self, rows):
        X = np.array([[row[c] for c in self.columns] for row in rows], dtype=np.float32)
        out = np.stack([predict(self.models[label], X) for label in LABELS], axis=1)
        out[:, 0] = np.clip(out[:, 0], *STRIDE_RANGE)
        out[:, 1] = np.clip(out[:, 1], *TURN_RANGE)
        return out


class BaselinePolicy:
    """Turn toward b in proportion to the bearing; walk fast when facing b, slow down near it."""

    name = "baseline"

    def act(self, rows):
        out = []
        for row in rows:
            bearing = math.atan2(row["target_sin"], row["target_cos"])
            turn = float(np.clip(0.5 * bearing, *TURN_RANGE))
            facing = max(0.0, math.cos(bearing))
            speed = min(1.0, row["target_dist"] / (2 * MAX_STEP))
            out.append((STRIDE_RANGE[1] * facing ** 2 * speed, turn))
        return np.array(out)
