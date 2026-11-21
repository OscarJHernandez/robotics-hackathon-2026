"""Policies that pick (stride, turn) for each gait cycle.

- `TabPFNNavPolicy`: TabPFN regressors (one for stride, one for turn) with a context of training
  examples. `checkpoint=None` uses the pretrained TabPFN v2 weights; a path uses fine-tuned weights.
- `BaselinePolicy`: a hand-written steer-to-bearing controller, for comparison.

All policies take a list of observed rows (one per robot, so many episodes run in lockstep) and
return an (n, 2) array.
"""

import math
import warnings
from pathlib import Path

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


def make_regressor(checkpoint=None, device="cpu", n_estimators=4, seed=0):
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    kwargs = dict(device=device, n_estimators=n_estimators, random_state=seed, fit_mode="fit_with_cache",
                  ignore_pretraining_limits=True)
    if checkpoint is not None:
        kwargs["model_path"] = str(checkpoint)
    return TabPFNRegressor.create_default_for_version(ModelVersion(MODEL_VERSION), **kwargs)


class TabPFNNavPolicy:
    def __init__(self, context, checkpoints=None, groups=DEFAULT_GROUPS, device="cpu", n_estimators=4,
                 context_rows=3000, seed=0, noise=None, name=None):
        """`checkpoints` = {"stride": path, "turn": path} for fine-tuned weights, or None."""
        self.groups = tuple(groups)
        self.columns = feature_columns(self.groups, context)
        self.context = sample_context(context, context_rows, seed)
        self.noise = noise  # (stride sd, turn sd) for exploration, or None
        self.rng = np.random.default_rng(seed)
        self.name = name or ("tabpfn-finetuned" if checkpoints else "tabpfn-pretrained")
        X = self.context[self.columns].to_numpy(np.float32)
        self.models = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for label in LABELS:
                model = make_regressor((checkpoints or {}).get(label), device, n_estimators, seed)
                model.fit(X, self.context[label].to_numpy(np.float32))
                self.models[label] = model

    def act(self, rows):
        X = np.array([[row[c] for c in self.columns] for row in rows], dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = np.stack([self.models[label].predict(X) for label in LABELS], axis=1)
        if self.noise is not None:
            out = out + self.rng.normal(0.0, self.noise, size=out.shape)
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


def checkpoint_paths(folder, round_):
    folder = Path(folder)
    return {label: folder / "round_{}_{}.ckpt".format(round_, label) for label in LABELS}
