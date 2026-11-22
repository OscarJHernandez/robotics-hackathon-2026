"""Settings (dagger.yaml), the episode plan of every round, and where runs are stored."""

import copy
import json
from pathlib import Path

import numpy as np
import yaml

from nav.common import data_dir

from . import REPO

DEFAULT_CONFIG = Path(__file__).resolve().parent / "dagger.yaml"
HELDOUT_START = 100_000  # held-out scene ids start here, far from the training ids
EPISODE_STRIDE = 1_000_000  # episode id = round * EPISODE_STRIDE + index

SMOKE = {"rounds": [{"episodes": 8, "beta": 1.0}, {"episodes": 4, "beta": 0.5}],
         "scenes": {"train": 8, "heldout": 2, "heldout_episodes": 2},
         "env": {"n_envs": 4, "max_steps": 40}}


def _merge(base, extra):
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_config(path=None, smoke=False):
    config = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text())
    if smoke:
        _merge(config, SMOKE)
    return config


def train_ids(config):
    return list(range(config["scenes"]["train"]))


def heldout_ids(config):
    return list(range(HELDOUT_START, HELDOUT_START + config["scenes"]["heldout"]))


def episode_plan(config, round_):
    """Episodes of one round: [{"episode", "scene_id", "seed"}]. Round "heldout" is the final test.

    Round 0 visits every training scene once (in order, then again when there are more episodes than scenes).
    Later rounds pick training scenes at random with a new start pose (seed = round number).
    """
    if round_ == "heldout":
        ids = heldout_ids(config)
        n = config["scenes"]["heldout_episodes"]
        return [{"episode": 99 * EPISODE_STRIDE + i, "scene_id": ids[i % len(ids)], "seed": 99 + i // len(ids)}
                for i in range(n)]
    ids = train_ids(config)
    n = config["rounds"][round_]["episodes"]
    if round_ == 0:
        picks = [ids[i % len(ids)] for i in range(n)]
        seeds = [i // len(ids) for i in range(n)]
    else:
        rng = np.random.default_rng([config["seed"], round_])
        picks = [int(s) for s in rng.choice(ids, size=n, replace=n > len(ids))]
        seeds = [round_] * n
    return [{"episode": round_ * EPISODE_STRIDE + i, "scene_id": s, "seed": seed}
            for i, (s, seed) in enumerate(zip(picks, seeds))]


def split_plan(plan, weights):
    """Split an episode plan between machines in proportion to `weights` {host: float}. Contiguous chunks."""
    hosts = [h for h, w in weights.items() if w > 0]
    total = sum(weights[h] for h in hosts)
    counts = [int(len(plan) * weights[h] / total) for h in hosts]
    for i in range(len(plan) - sum(counts)):
        counts[i % len(hosts)] += 1
    out, start = {}, 0
    for host, count in zip(hosts, counts):
        out[host] = plan[start:start + count]
        start += count
    return out


def runs_dir():
    path = data_dir() / "dagger"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_dir(name):
    path = runs_dir() / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "logs").mkdir(exist_ok=True)
    return path


def vision_dir():
    """data/vision in the main checkout (same rule as nav's data_dir)."""
    return data_dir().parent / "vision"


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=1, default=str))


def repo():
    return REPO


# ---- v2: data sets by role (disjoint scene ids), and where things live ---------------------------------------------

SETS = {  # name: (first scene id, layouts, start seed, first episode id)
    "round0": (0, 1000, 0, 0),                      # training: expert, one episode per layout
    "perturb": (2000, 300, 0, 2_000_000),            # training: expert with heading perturbations, new layouts
    "offline_expert": (100_000, 200, 99, 100_000),   # held out, offline metrics only
    "offline_perturb": (100_000, 100, 98, 110_000),  # held out, perturbed expert, offline metrics only
    "screen": (101_000, 200, 99, 101_000),           # held out, closed loop, student alone
    "confirm": (103_000, 200, 99, 103_000),          # held out, once per promotion
    "final": (102_000, 200, 99, 102_000),            # held out, robust check (fresh block per check: 104000, ...)
    "final_b": (102_000, 200, 98, 105_000),          # the same final layouts with a second start seed
}


def set_spec(name):
    """(first scene id, layouts, seed, first episode id) of a fixed set, or of a DAgger round "dagger_r<k>"
    (k = 1..60: student-driven rounds on their own training layouts 10000 + 1000 k, never the held-out ones)."""
    if name in SETS:
        return SETS[name]
    if name.startswith("dagger_r") and name[8:].isdigit() and 1 <= int(name[8:]) <= 60:
        k = int(name[8:])
        return (10_000 + 1000 * k, 1000, 10 + k, 20_000_000 + 100_000 * k)
    raise KeyError("unknown data set {!r}".format(name))


def set_plan(name, first=0, count=None):
    """Episodes of a set: [{"episode", "scene_id", "seed"}], optionally only `count` of them from index `first`."""
    scene0, layouts, seed, episode0 = set_spec(name)
    count = layouts - first if count is None else count
    return [{"episode": episode0 + i, "scene_id": scene0 + i, "seed": seed} for i in range(first, first + count)]


def v2_dir():
    path = runs_dir() / "v2"
    for sub in ("data", "experiments"):
        (path / sub).mkdir(parents=True, exist_ok=True)
    return path


def dataset_dir(name):
    path = v2_dir() / "data" / name
    for sub in ("parts", "episodes", "logs"):
        (path / sub).mkdir(parents=True, exist_ok=True)
    return path


def experiment_dir(exp_id):
    path = v2_dir() / "experiments" / exp_id
    for sub in ("parts", "episodes", "logs"):
        (path / sub).mkdir(parents=True, exist_ok=True)
    return path
