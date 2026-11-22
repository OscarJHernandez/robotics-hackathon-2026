"""The student: two TabPFN regressors (stride, turn), built with TabPFN's own `fit()` once per round.

No fine-tuning (AGENTS.md): `fit()` stores the rows as context and the pretrained weights stay as they are.
To make the student better, change the rows that go into `fit()`.
"""

import time
import warnings

import numpy as np

from nav.common import STRIDE_RANGE, TURN_RANGE

from .envfile import require_token

LABELS = ("y_stride", "y_turn")


class TabPFNPolicy:
    name = "tabpfn-dagger"

    def __init__(self, columns, categorical, device="auto", n_estimators=4, model_version="v3.5", seed=0):
        self.columns = list(columns)
        self.categorical_idx = [self.columns.index(c) for c in categorical if c in self.columns]
        self.device, self.n_estimators, self.model_version, self.seed = device, n_estimators, model_version, seed
        self.models = {}
        self.fit_seconds = 0.0
        self.context_rows = 0

    @classmethod
    def from_config(cls, student_cfg, columns, categorical, seed=0):
        return cls(columns, categorical, student_cfg["device"], student_cfg["n_estimators"],
                   student_cfg["model_version"], seed)

    def _make(self):
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        return TabPFNRegressor.create_default_for_version(
            ModelVersion(self.model_version), device=self.device, n_estimators=self.n_estimators,
            random_state=self.seed, categorical_features_indices=self.categorical_idx or None,
            fit_mode="fit_with_cache", ignore_pretraining_limits=True)

    def fit(self, context):
        """TabPFN's fit() on the context rows, once per label. Returns the seconds it took."""
        require_token(self.model_version)
        X = np.asarray(context[self.columns], np.float32)
        start = time.perf_counter()
        for label in LABELS:
            model = self._make()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X, np.asarray(context[label], np.float32))
            self.models[label] = model
        self.fit_seconds = time.perf_counter() - start
        self.context_rows = len(X)
        return self.fit_seconds

    def act_batch(self, rows):
        """Commands for a batch of feature rows (DataFrame or array in `columns` order): array (n, 2)."""
        X = np.asarray(rows[self.columns] if hasattr(rows, "columns") else rows, np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stride = self.models["y_stride"].predict(X)
            turn = self.models["y_turn"].predict(X)
        return np.stack([np.clip(stride, *STRIDE_RANGE), np.clip(turn, *TURN_RANGE)], axis=1)


def choose_context(table, max_rows=None, seed=0, bins=6):
    """All rows, or at most `max_rows` spread evenly over bins of the expert's turn and stride."""
    if not max_rows or len(table) <= max_rows:
        return table
    turn_bin = np.digitize(table["y_turn"], np.linspace(*TURN_RANGE, bins + 1)[1:-1])
    stride_bin = np.digitize(table["y_stride"], np.linspace(*STRIDE_RANGE, 4)[1:-1])
    key = turn_bin * 10 + stride_bin
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(key == k) for k in np.unique(key)]
    quota = max(1, max_rows // len(groups))
    keep = np.concatenate([rng.choice(g, min(len(g), quota), replace=False) for g in groups])
    rest = np.setdiff1d(np.arange(len(table)), keep)
    if len(keep) < max_rows and len(rest):
        keep = np.concatenate([keep, rng.choice(rest, min(len(rest), max_rows - len(keep)), replace=False)])
    return table.iloc[np.sort(keep)]
