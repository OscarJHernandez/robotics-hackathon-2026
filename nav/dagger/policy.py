"""The student: two TabPFN regressors (stride, turn), built with TabPFN's own `fit()` once per round.

No fine-tuning (AGENTS.md): `fit()` stores the rows as context and the pretrained weights stay as they are.
To make the student better, change the rows that go into `fit()`.
"""

import re
import threading
import time
import warnings

import numpy as np

from nav.common import TURN_RANGE

from . import STRIDE

from .envfile import require_token

LABELS = ("y_stride", "y_turn")
API_MIN_INTERVAL = 1.05  # seconds between predict() calls: the API allows at most 60 per minute
API_RETRIES = 8
API_TIMEOUT = 120.0  # seconds: a predict() that has not answered by then is given up and tried again


def run_with_timeout(fn, args, timeout):
    """fn(*args) in a daemon thread; raises TimeoutError after `timeout` s (the thread is left behind and
    cannot keep the process from exiting)."""
    result = {}

    def target():
        try:
            result["value"] = fn(*args)
        except BaseException as error:  # handed to the caller
            result["error"] = error

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError("API call did not answer within {:.0f} s".format(timeout))
    if "error" in result:
        raise result["error"]
    return result["value"]


def call_api(fn, *args, state={"last": 0.0}, timeout=API_TIMEOUT):
    """One API call, spaced API_MIN_INTERVAL from the last, retried on HTTP 429 (rate limit) after the wait
    the server asks for, and retried when it does not answer within `timeout` s."""
    for attempt in range(API_RETRIES + 1):
        wait = state["last"] + API_MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        state["last"] = time.monotonic()
        try:
            return run_with_timeout(fn, args, timeout)
        except TimeoutError:
            if attempt == API_RETRIES:
                raise
        except RuntimeError as error:
            if "429" not in str(error) or attempt == API_RETRIES:
                raise
            match = re.search(r"Retry in (\d+(?:\.\d+)?)s", str(error))
            time.sleep(float(match.group(1)) + 0.5 if match else 5.0 * (attempt + 1))


class TabPFNPolicy:
    name = "tabpfn-dagger"

    def __init__(self, columns, categorical, device="auto", n_estimators=4, model_version="v3.5", seed=0,
                 backend="local"):
        if backend not in ("local", "api"):
            raise ValueError("student backend must be 'local' or 'api'")
        self.columns = list(columns)
        self.categorical_idx = [self.columns.index(c) for c in categorical if c in self.columns]
        self.device, self.n_estimators, self.model_version, self.seed = device, n_estimators, model_version, seed
        self.backend = backend
        self.source_index = None  # set with use_source_columns() when act_batch() gets wider matrices
        self.models = {}
        self.fit_seconds = 0.0
        self.context_rows = 0
        if backend == "api":
            self.device = "api"

    @classmethod
    def from_config(cls, student_cfg, columns, categorical, seed=0):
        return cls(columns, categorical, student_cfg["device"], student_cfg["n_estimators"],
                   student_cfg["model_version"], seed, student_cfg.get("backend", "local"))

    def use_source_columns(self, source_columns):
        """act_batch() will get arrays in `source_columns` order and pick this student's columns from them."""
        source = list(source_columns)
        self.source_index = None if source == self.columns else [source.index(c) for c in self.columns]
        return self

    def _make(self):
        if self.backend == "api":
            # Prior Labs' hosted TabPFN (tabpfn-client): the rows go to their servers and every predict() uses
            # the account's quota. fit_with_cache lets later predict() calls reuse the server-side fit.
            from tabpfn_client import TabPFNRegressor as ApiRegressor
            return ApiRegressor(model_path="{}_default".format(self.model_version), n_estimators=self.n_estimators,
                                random_state=self.seed, categorical_features_indices=self.categorical_idx or None,
                                fit_mode="fit_with_cache", ignore_pretraining_limits=True)
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
        if hasattr(rows, "columns"):
            X = np.asarray(rows[self.columns], np.float32)
        else:
            X = np.asarray(rows, np.float32)
            if self.source_index is not None:
                X = X[:, self.source_index]  # a featurizer matrix with more columns than the student uses
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.backend == "api":
                stride = call_api(self.models["y_stride"].predict, X)
                turn = call_api(self.models["y_turn"].predict, X)
            else:
                stride = self.models["y_stride"].predict(X)
                turn = self.models["y_turn"].predict(X)
        return np.stack([np.clip(stride, *STRIDE), np.clip(turn, *TURN_RANGE)], axis=1)


def choose_context(table, max_rows=None, seed=0, bins=6):
    """All rows, or at most `max_rows` spread evenly over bins of the expert's turn and stride."""
    if not max_rows or len(table) <= max_rows:
        return table
    turn_bin = np.digitize(table["y_turn"], np.linspace(*TURN_RANGE, bins + 1)[1:-1])
    stride_bin = np.digitize(table["y_stride"], [-1e-6, 0.1, 0.2])  # backing up / stop or slow / medium / fast
    key = turn_bin * 10 + stride_bin
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(key == k) for k in np.unique(key)]
    quota = max(1, max_rows // len(groups))
    keep = np.concatenate([rng.choice(g, min(len(g), quota), replace=False) for g in groups])
    rest = np.setdiff1d(np.arange(len(table)), keep)
    if len(keep) < max_rows and len(rest):
        keep = np.concatenate([keep, rng.choice(rest, min(len(rest), max_rows - len(keep)), replace=False)])
    return table.iloc[np.sort(keep)]
