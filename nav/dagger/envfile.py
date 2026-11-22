"""The TabPFN API key: from $TABPFN_TOKEN, else from a gitignored `.env` file. Never printed or logged."""

import os
import subprocess
from pathlib import Path

from . import REPO

GATED = ("v2.5", "v2.6", "v3", "v3.5", "v3.5-fast")  # model versions whose weights need an accepted licence


def _candidates():
    yield REPO / ".env"
    try:
        common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=REPO,
                                capture_output=True, text=True, check=True).stdout.strip()
        yield Path(common).parent / ".env"  # the main checkout, when REPO is a git worktree
    except (OSError, subprocess.CalledProcessError):
        pass


def read_env_file(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_token():
    """Put TABPFN_TOKEN into the environment if a .env has it. Returns where it came from (never the value)."""
    if os.environ.get("TABPFN_TOKEN"):
        return "environment"
    for path in _candidates():
        if path.is_file():
            token = read_env_file(path).get("TABPFN_TOKEN")
            if token:
                os.environ["TABPFN_TOKEN"] = token
                return str(path)
    return None


def require_token(model_version):
    """Fail early with a clear message when a licence-gated model has no key."""
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")  # never open a login page on a headless worker
    if model_version not in GATED:
        return None
    source = load_token()
    if source is None:
        raise SystemExit("TabPFN {} needs an API key: set TABPFN_TOKEN, or put TABPFN_TOKEN=... in the "
                         "repo's .env (gitignored). Or set model_version: v2 in dagger.yaml.".format(model_version))
    return source
