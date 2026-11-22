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


# ---------------------------------------------------------------------------------------------
# DAgger v2: gated student, shield, mode-aware context, baselines. Still fit()/predict() only.
# ---------------------------------------------------------------------------------------------

STOP_STRIDE = 0.02
TURN_MAGNITUDE = TURN_RANGE[1]
FREE_BINS = (0.5, 1.0)  # metres: < 0.5, 0.5 to 1, > 1
MODE_NAMES = ("walk", "stop_left", "stop_right")


def mode_labels(y_stride, y_turn):
    """0 = walk, 1 = stop and turn left, 2 = stop and turn right (stop when stride < 0.02; left when turn >= 0)."""
    stride, turn = np.asarray(y_stride, float), np.asarray(y_turn, float)
    return np.where(stride >= STOP_STRIDE, 0, np.where(turn >= 0, 1, 2)).astype(int)


def _matrix(rows, columns):
    return np.asarray(rows[columns] if hasattr(rows, "columns") else rows, np.float32)


class GatedPolicy:
    """TabPFN classifier picks the mode (walk / stop-left / stop-right); two regressors, fitted on walk rows only,
    give stride and turn when walking. Stop modes use fixed commands. All fit()/predict(): no weights change."""
    name = "tabpfn-gated"

    def __init__(self, columns, categorical, device="auto", n_estimators=4, model_version="v3.5", seed=0):
        self.columns = list(columns)
        self.categorical_idx = [self.columns.index(c) for c in categorical if c in self.columns]
        self.device, self.n_estimators, self.model_version, self.seed = device, n_estimators, model_version, seed
        self.classifier = None
        self.models = {}
        self.fit_seconds = 0.0
        self.context_rows = 0
        self.last_proba = None  # (n, 3) mode probabilities of the last act_batch

    def _kwargs(self):
        from tabpfn.constants import ModelVersion
        return dict(device=self.device, n_estimators=self.n_estimators, random_state=self.seed,
                    categorical_features_indices=self.categorical_idx or None, fit_mode="fit_with_cache",
                    ignore_pretraining_limits=True), ModelVersion(self.model_version)

    def _make_classifier(self):
        from tabpfn import TabPFNClassifier
        kw, version = self._kwargs()
        return TabPFNClassifier.create_default_for_version(version, **kw)

    def _make_regressor(self):
        from tabpfn import TabPFNRegressor
        kw, version = self._kwargs()
        return TabPFNRegressor.create_default_for_version(version, **kw)

    def fit(self, context):
        require_token(self.model_version)
        X = np.asarray(context[self.columns], np.float32)
        modes = mode_labels(context["y_stride"], context["y_turn"])
        walk = modes == 0
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.classifier = self._make_classifier()
            self.classifier.fit(X, modes)
            for label in LABELS:
                model = self._make_regressor()
                model.fit(X[walk], np.asarray(context[label], np.float32)[walk])
                self.models[label] = model
        self.fit_seconds = time.perf_counter() - start
        self.context_rows = len(X)
        return self.fit_seconds

    def act_batch(self, rows):
        X = _matrix(rows, self.columns)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = np.asarray(self.classifier.predict_proba(X))
            stride = self.models["y_stride"].predict(X)
            turn = self.models["y_turn"].predict(X)
        proba = np.zeros((len(X), 3))
        for j, cls in enumerate(self.classifier.classes_):
            proba[:, int(cls)] = raw[:, j]
        self.last_proba = proba
        mode = proba.argmax(axis=1)
        out = np.stack([np.clip(stride, *STRIDE_RANGE), np.clip(turn, *TURN_RANGE)], axis=1)
        out[mode == 1] = (0.0, TURN_MAGNITUDE)
        out[mode == 2] = (0.0, -TURN_MAGNITUDE)
        return out


class Shield:
    """Wraps a policy: too close to an obstacle straight ahead means stride 0 and a turn toward the freer side.

    `rows` should be a DataFrame holding free_ahead and free_left/free_right (or free_s0..free_s6). A plain array
    is read in the wrapped policy's `columns` order, which then must contain those columns."""
    name = "student + shield"

    def __init__(self, policy, free_ahead_min=0.35):
        self.policy, self.free_ahead_min = policy, free_ahead_min
        self.columns = getattr(policy, "columns", None)
        self.shielded = 0  # rows overridden by the last act_batch

    def __getattr__(self, name):  # fit, fit_seconds, last_proba ... come from the wrapped policy
        return getattr(self.__dict__["policy"], name)

    @staticmethod
    def _sides(rows):
        have = set(rows.columns)
        if {"free_left", "free_right"} <= have:
            return rows["free_left"].to_numpy(float), rows["free_right"].to_numpy(float)
        if all(f"free_s{i}" in have for i in range(7)):
            left = rows[["free_s0", "free_s1", "free_s2"]].to_numpy(float).mean(axis=1)
            right = rows[["free_s4", "free_s5", "free_s6"]].to_numpy(float).mean(axis=1)
            return left, right
        raise KeyError("Shield needs free_left/free_right or free_s0..free_s6 columns")

    def act_batch(self, rows):
        if not hasattr(rows, "columns"):
            import pandas as pd
            rows = pd.DataFrame(np.asarray(rows), columns=self.columns)
        out = np.array(self.policy.act_batch(rows), float)
        close = rows["free_ahead"].to_numpy(float) < self.free_ahead_min
        left, right = self._sides(rows)
        out[close, 0] = 0.0
        weak = close & (np.abs(out[:, 1]) < 0.1)
        out[weak, 1] = np.where(left[weak] >= right[weak], TURN_MAGNITUDE, -TURN_MAGNITUDE)
        self.shielded = int(close.sum())
        return out


LIDAR_BINS = (0.27, 0.45)  # metres: nearest lidar range; under 0.27 m is almost absent from expert data (18 of 41k rows) yet every crash happens there


def lidar_min(table):
    cols = [c for c in table.columns if c.startswith("lr_")]
    if not cols:
        raise ValueError("lidar quotas need lr_ columns in the rows")
    return table[cols].to_numpy(float).min(axis=1)


def _group_keys(table, by="free"):
    mode = mode_labels(table["y_stride"], table["y_turn"])
    if by == "lidar":
        near = np.digitize(lidar_min(table), LIDAR_BINS)
    else:
        near = np.digitize(np.asarray(table["free_ahead"], float), FREE_BINS)
    return mode * 3 + near


def context_summary(table):
    """Rows per group (mode x free-ahead bin) as {"walk/<0.5m": n, ...}, plus "student_rows"."""
    labels = ["<0.5m", "0.5-1m", ">1m"]
    key = _group_keys(table)
    out = {f"{MODE_NAMES[k // 3]}/{labels[k % 3]}": int((key == k).sum()) for k in np.unique(key)}
    if "used_expert" in table.columns:
        out["student_rows"] = int((~table["used_expert"].astype(bool)).sum())
    return out


def _fill(table, keep, max_rows, rng):
    rest = np.setdiff1d(np.arange(len(table)), keep)
    if len(keep) < max_rows and len(rest):
        keep = np.concatenate([keep, rng.choice(rest, min(len(rest), max_rows - len(keep)), replace=False)])
    return table.iloc[np.sort(keep)]


def choose_context(table, max_rows=None, seed=0, bins=6, quotas="turn_stride"):
    """All rows, or at most `max_rows` chosen by a quota scheme.

    "turn_stride" (default): even over bins of the expert's turn and stride.
    "mode_x_free": rows the student drove (used_expert False) are always kept; the rest get an equal quota per
    non-empty group of mode (walk / stop-left / stop-right) x free-ahead bin (<0.5, 0.5-1, >1 m); the remainder
    is filled at random."""
    if quotas not in ("turn_stride", "mode_x_free", "mode_x_lidar", "uniform"):
        raise ValueError(f"unknown quotas {quotas!r}")
    if not max_rows or len(table) <= max_rows:
        return table
    rng = np.random.default_rng(seed)
    if quotas == "uniform":  # a plain random sample
        return table.iloc[np.sort(rng.choice(len(table), max_rows, replace=False))]
    if quotas in ("mode_x_free", "mode_x_lidar"):
        if "used_expert" in table.columns:
            student = (~table["used_expert"].astype(bool)).to_numpy()
        else:
            student = np.zeros(len(table), bool)
        keep = np.flatnonzero(student)
        if len(keep) > max_rows:
            keep = rng.choice(keep, max_rows, replace=False)
        room = max_rows - len(keep)
        pool = np.flatnonzero(~student)
        key = _group_keys(table, "lidar" if quotas == "mode_x_lidar" else "free")[pool]
        groups = [pool[key == k] for k in np.unique(key)]
        if room > 0 and groups:
            quota = max(1, room // len(groups))
            keep = np.concatenate([keep] + [rng.choice(g, min(len(g), quota), replace=False) for g in groups])
        return _fill(table, keep, max_rows, rng)
    turn_bin = np.digitize(table["y_turn"], np.linspace(*TURN_RANGE, bins + 1)[1:-1])
    stride_bin = np.digitize(table["y_stride"], np.linspace(*STRIDE_RANGE, 4)[1:-1])
    key = turn_bin * 10 + stride_bin
    groups = [np.flatnonzero(key == k) for k in np.unique(key)]
    quota = max(1, max_rows // len(groups))
    keep = np.concatenate([rng.choice(g, min(len(g), quota), replace=False) for g in groups])
    return _fill(table, keep, max_rows, rng)


def summary_columns(columns):
    """The non-patch columns (no pc_/pd_ prefix): for the summary-only TabPFN baseline."""
    return [c for c in columns if not c.startswith(("pc_", "pd_"))]


class KNNPolicy:
    """k-nearest-neighbours baseline on standardized columns (multi-output stride and turn)."""
    name = "knn"

    def __init__(self, columns, k=10):
        self.columns, self.k = list(columns), k
        self.model = None
        self.mean = self.std = None
        self.fit_seconds = 0.0
        self.context_rows = 0

    def fit(self, context):
        from sklearn.neighbors import KNeighborsRegressor
        X = np.asarray(context[self.columns], float)
        start = time.perf_counter()
        self.mean = X.mean(axis=0)
        std = X.std(axis=0)
        self.std = np.where(std > 1e-9, std, 1.0)
        self.model = KNeighborsRegressor(n_neighbors=min(self.k, len(X)))
        self.model.fit((X - self.mean) / self.std, np.asarray(context[list(LABELS)], float))
        self.fit_seconds = time.perf_counter() - start
        self.context_rows = len(X)
        return self.fit_seconds

    def act_batch(self, rows):
        X = (np.asarray(_matrix(rows, self.columns), float) - self.mean) / self.std
        pred = self.model.predict(X)
        return np.stack([np.clip(pred[:, 0], *STRIDE_RANGE), np.clip(pred[:, 1], *TURN_RANGE)], axis=1)


class RulePolicy:
    """No model: stop and turn on the spot when the goal is more than `threshold` rad off, else walk toward it
    (the hand-written controller the expert uses, aimed at the goal itself). Shows how far heading alone goes."""
    name = "rule"

    def __init__(self, columns, threshold=0.8, gain=0.6):
        self.columns = list(columns)
        self.threshold, self.gain = threshold, gain
        self._idx = [self.columns.index(c) for c in ("goal_dist", "goal_sin", "goal_cos")]
        self.fit_seconds = 0.0
        self.context_rows = 0

    def fit(self, context):
        self.context_rows = len(context)
        return 0.0

    def act_batch(self, rows):
        X = _matrix(rows, self.columns)
        dist, bearing = X[:, self._idx[0]], np.arctan2(X[:, self._idx[1]], X[:, self._idx[2]])
        turn = np.clip(self.gain * bearing, *TURN_RANGE)
        stride = STRIDE_RANGE[1] * np.maximum(0.0, np.cos(bearing)) ** 2 * np.clip(dist / 0.5, 0.4, 1.0)
        stop = np.abs(bearing) > self.threshold
        stride = np.where(stop, 0.0, stride)
        turn = np.where(stop, np.sign(bearing) * TURN_MAGNITUDE, turn)
        return np.stack([np.clip(stride, *STRIDE_RANGE), turn], axis=1)


class LidarShield:
    """Wraps a student: if any lidar ray within +-`fov_deg` of straight ahead reads under `min_range`, the stride is
    set to 0 and the robot turns away from that obstacle. A hand-written rule on top of the student, always reported
    separately ("student + lidar shield"). Expert data shows the robot is almost never this close (18 of 41k rows
    under 0.26 m), while every student crash happened at 0.15 to 0.31 m."""
    name = "student + lidar shield"

    def __init__(self, policy, columns, min_range=0.27, fov_deg=90, front_range=None, front_deg=45,
                 turn="away"):
        """`front_range` (optional): a larger threshold for the cone within +-`front_deg` of straight ahead, since
        a step walks up to 0.13 m forward. `turn`: "away" from the nearest ray, or "freer" = toward the side
        (left or right half) with the larger mean clearance."""
        self.policy, self.min_range, self.fov = policy, min_range, fov_deg
        self.front_range, self.front_deg, self.turn = front_range, front_deg, turn
        if turn not in ("away", "freer"):
            raise ValueError("turn must be 'away' or 'freer'")
        self.columns = list(columns)
        self._rays = [i for i, c in enumerate(self.columns) if c.startswith("lr_")]
        if not self._rays:
            raise ValueError("the lidar shield needs lr_ feature columns")
        n = len(self._rays)
        angles = np.array([int(self.columns[i][3:]) * 360.0 / n for i in self._rays])
        self._bearing = np.where(angles > 180, angles - 360, angles)  # degrees, + = left
        self._front = np.abs(self._bearing) <= fov_deg
        self.shielded = 0

    def __getattr__(self, name):  # fit, fit_seconds, context_rows ... come from the wrapped policy
        return getattr(self.__dict__["policy"], name)

    def act_batch(self, rows):
        X = _matrix(rows, self.columns)
        out = np.array(self.policy.act_batch(X), float)
        ranges = X[:, self._rays]
        front = np.where(self._front, ranges, np.inf)
        nearest = front.argmin(axis=1)
        close = front[np.arange(len(X)), nearest] < self.min_range
        if self.front_range is not None:
            cone = np.abs(self._bearing) <= self.front_deg
            close |= np.where(cone, ranges, np.inf).min(axis=1) < self.front_range
        out[close, 0] = 0.0
        if self.turn == "freer":
            left = (self._bearing > 10) & (self._bearing < 170)
            right = (self._bearing < -10) & (self._bearing > -170)
            sign = np.where(ranges[:, left].mean(axis=1) >= ranges[:, right].mean(axis=1), 1.0, -1.0)
            out[close, 1] = sign[close] * TURN_MAGNITUDE
        else:
            away = -np.sign(self._bearing[nearest[close]])  # obstacle on the left (+) -> turn right (-)
            away = np.where(away == 0, 1.0, away)
            out[close, 1] = away * TURN_MAGNITUDE
        self.shielded = int(close.sum())
        return out


# ---------------------------------------------------------------------------------------------
# The hosted TabPFN API (tabpfn-client): the same fit()/predict(), run by Prior Labs. Used by the app's student.
# ---------------------------------------------------------------------------------------------

# Measured 2026-11-22 with the v29 export (31,586 rows stacked to 63,172 x 117): with fit_mode "fit_with_cache" fit()
# is free (18 s) and a predict() costs 10,000 credits and takes about 0.6 s; with the server default
# ("fit_preprocessors") every predict() re-runs the whole context: about 13.4 s and 11,400 credits.
CREDITS_PER_PREDICT = 10_000
CLIENT_FIT_MODE = "fit_with_cache"


class OutOfCredits(RuntimeError):
    """The TabPFN API budget (a CreditBudget, or the account's credits) would be exceeded."""


class CreditBudget:
    """At most `max_calls` predict() calls; `spend()` raises OutOfCredits before the call that would exceed it."""

    def __init__(self, max_calls):
        self.max_calls = None if max_calls is None else int(max_calls)
        self.calls = 0

    def spend(self, n=1):
        if self.max_calls is not None and self.calls + n > self.max_calls:
            raise OutOfCredits("TabPFN API budget of {} predict calls used up".format(self.max_calls))
        self.calls += n

    @property
    def left(self):
        return None if self.max_calls is None else self.max_calls - self.calls

    def credits(self):
        return self.calls * CREDITS_PER_PREDICT


def client_login():
    """Give tabpfn_client the key from $TABPFN_TOKEN or the gitignored .env (never printed). False without a key."""
    import os

    from .envfile import load_token
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    if load_token() is None:
        return False
    import tabpfn_client
    tabpfn_client.set_access_token(os.environ["TABPFN_TOKEN"])
    return True


def parse_usage(usage):
    """{"used", "limit", "left", "reset"} from the API's usage dict (current_usage, usage_limit, reset_time) or
    from tabpfn_client.get_api_usage()'s sentence. limit and left are None when the limit is unlimited (-1)."""
    import re
    if isinstance(usage, str):
        m = re.search(r"used (\d+) of the allowed limit of (Unlimited|-?\d+) credits(?:\. The limit will reset at (.*?)\.?$)?",
                      usage.strip())
        if not m:
            raise ValueError("cannot read the TabPFN API usage")
        used, limit, reset = int(m.group(1)), (-1 if m.group(2) == "Unlimited" else int(m.group(2))), m.group(3)
    else:
        used, limit, reset = int(usage["current_usage"]), int(usage["usage_limit"]), usage.get("reset_time")
    if limit == -1:
        return {"used": used, "limit": None, "left": None, "reset": reset}
    return {"used": used, "limit": limit, "left": limit - used, "reset": reset}


def api_usage():
    """The account's usage through tabpfn_client (one small request, no credits). Needs client_login() first."""
    try:
        from tabpfn_client.client import ServiceClient
        from tabpfn_client.config import get_access_token
        return parse_usage(ServiceClient.get_api_usage(get_access_token()))
    except ImportError:
        import tabpfn_client
        return parse_usage(tabpfn_client.get_api_usage())


def credits_left():
    """Credits left on the TabPFN API account: an int, or None when the limit is unlimited."""
    return api_usage()["left"]


class ClientTabPFNPolicy:
    """The student through the hosted TabPFN API: ONE regressor for both commands, so one decision is one call.

    fit(): the context rows stacked twice with an `output` column (0 = stride, 1 = turn) and the matching label.
    act_batch(rows): 2n rows (each row once with output 0 and once with output 1) in ONE predict() call, clipped to
    the command ranges. Still TabPFN's fit()/predict() only: no weights change (AGENTS.md).
    `budget` (CreditBudget) is charged once per predict() call, before it is made. fit_mode "fit_with_cache" (default)
    keeps a cache of the context on the server, so a predict() is fast; if the server has dropped it, the policy
    calls fit() again (free) and retries once.
    """
    name = "tabpfn-client"

    def __init__(self, columns, categorical=(), n_estimators=4, model_version="v3.5", seed=0, budget=None,
                 fit_mode=CLIENT_FIT_MODE, estimator=None):
        self.columns = list(columns)
        self.categorical_idx = [self.columns.index(c) for c in categorical if c in self.columns]
        self.n_estimators, self.model_version, self.seed, self.fit_mode = n_estimators, model_version, seed, fit_mode
        self.budget = budget if budget is not None else CreditBudget(None)
        self._estimator = estimator  # a factory for tests (a fake); default: tabpfn_client.TabPFNRegressor
        self.model = None
        self.fit_seconds = 0.0
        self.context_rows = 0
        self.calls = 0
        self.last_seconds = None

    def _make(self):
        if self._estimator is not None:
            return self._estimator()
        from tabpfn_client import TabPFNRegressor
        kw = dict(n_estimators=self.n_estimators, random_state=self.seed, ignore_pretraining_limits=True,
                  categorical_features_indices=self.categorical_idx + [len(self.columns)])  # `output` is categorical
        if self.fit_mode:
            kw["fit_mode"] = self.fit_mode
        return TabPFNRegressor.create_default_for_version(self.model_version, **kw)

    @staticmethod
    def stack(X, outputs=(0.0, 1.0)):
        """(n, k) -> (2n, k + 1): the rows with output 0, then the same rows with output 1."""
        X = np.asarray(X, np.float32)
        return np.concatenate([np.column_stack([X, np.full(len(X), o, np.float32)]) for o in outputs])

    def fit(self, context):
        X = np.asarray(context[self.columns], np.float32)
        y = np.concatenate([np.asarray(context[label], np.float32) for label in LABELS])
        start = time.perf_counter()
        self._context = (self.stack(X), y)
        self._fit()
        self.fit_seconds = time.perf_counter() - start
        self.context_rows = len(X)
        return self.fit_seconds

    def _fit(self):
        self.model = self._make()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(*self._context)

    def _predict(self, Z):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.asarray(self.model.predict(Z), float).reshape(-1)

    def act_batch(self, rows):
        X = _matrix(rows, self.columns)
        self.budget.spend(1)
        start = time.perf_counter()
        try:
            pred = self._predict(self.stack(X))
        except Exception as error:  # the server dropped its cache of the context: fit() again (free), once
            if type(error).__name__ not in ("RefitRequiredError", "FittedModelNotFoundError"):
                raise
            self._fit()
            self.budget.spend(1)
            pred = self._predict(self.stack(X))
        self.last_seconds = time.perf_counter() - start
        self.calls += 1
        n = len(X)
        return np.stack([np.clip(pred[:n], *STRIDE_RANGE), np.clip(pred[n:2 * n], *TURN_RANGE)], axis=1)


STUDENT_KINDS = ("tabpfn", "gated", "knn", "rule")


BACKENDS = ("local", "client")


def make_student(student_cfg, columns, categorical, seed=0, budget=None):
    """Build the student from config: kind tabpfn | gated | knn | rule, optional shield {free_ahead_min}.

    `backend` (kind tabpfn only): "local" (default: the tabpfn package on this machine, two regressors) or
    "client" (the hosted TabPFN API, one stacked regressor, one call per decision; `budget` caps the calls)."""
    kind = student_cfg.get("kind", "tabpfn")
    backend = student_cfg.get("backend", "local")
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}")
    if backend == "client" and kind != "tabpfn":
        raise ValueError("the client backend is for kind tabpfn only")
    if kind == "rule":
        return RulePolicy(columns, student_cfg.get("threshold", 0.8), student_cfg.get("gain", 0.6))
    if kind == "tabpfn" and backend == "client":
        policy = ClientTabPFNPolicy(columns, categorical, student_cfg.get("n_estimators", 4),
                                    student_cfg.get("model_version", "v3.5"), seed, budget,
                                    student_cfg.get("fit_mode", CLIENT_FIT_MODE))
    elif kind == "tabpfn":
        policy = TabPFNPolicy.from_config(student_cfg, columns, categorical, seed)
    elif kind == "gated":
        policy = GatedPolicy(columns, categorical, student_cfg["device"], student_cfg["n_estimators"],
                             student_cfg["model_version"], seed)
    elif kind == "knn":
        policy = KNNPolicy(columns, student_cfg.get("k", 10))
    else:
        raise ValueError(f"unknown student kind {kind!r}")
    shield = student_cfg.get("shield")
    if shield is not None:
        policy = Shield(policy, **shield)
    lidar_shield = student_cfg.get("lidar_shield")
    if lidar_shield is not None:
        policy = LidarShield(policy, columns, **lidar_shield)
    return policy
