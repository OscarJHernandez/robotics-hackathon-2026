"""Verified copies between machines: scp, then check size and SHA-256 on the other side.

Remote paths are relative to ~/Projects/robotics-hackathon-2026 on the host. `-o BatchMode=yes` makes ssh fail
instead of asking for a password.
"""

import hashlib
import shlex
import subprocess
from pathlib import Path

REMOTE_ROOT = "~/Projects/robotics-hackathon-2026"
SSH_OPTS = ["-o", "BatchMode=yes"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _remote(rel):
    return "{}/{}".format(REMOTE_ROOT, str(rel).lstrip("/"))


def _shell_path(rel):
    """Remote path for use inside double quotes in a remote shell ($HOME expands, ~ would not)."""
    return _remote(rel).replace("~", "$HOME", 1)


def scp_command(src, dst):
    return ["scp", "-p", *SSH_OPTS, src, dst]


def remote_hash_command(host, rel):
    """ssh command that prints the file's SHA-256 line and then its size in bytes."""
    script = 'f="{}"; (shasum -a 256 "$f" || sha256sum "$f") && wc -c < "$f"'.format(_shell_path(rel))
    return ["ssh", *SSH_OPTS, host, script]


def remote_mkdir_command(host, rel):
    parent = _shell_path(rel).rsplit("/", 1)[0]
    return ["ssh", *SSH_OPTS, host, 'mkdir -p "{}"'.format(parent)]


def remote_list_command(host, rel):
    return ["ssh", *SSH_OPTS, host, 'cd "{}" 2>/dev/null && find . -type f | sort'.format(_shell_path(rel))]


def _run(cmd, allow_fail=False):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 and not allow_fail:
        raise RuntimeError("{} failed ({}): {}".format(" ".join(shlex.quote(c) for c in cmd[:4]), result.returncode,
                                                       result.stderr.strip()))
    return result.stdout if result.returncode == 0 else ""


def remote_info(host, rel):
    """(sha256, size) of a remote file."""
    parts = _run(remote_hash_command(host, rel)).split()
    return parts[0], int(parts[2])


def _verify(local_sha, local_size, remote_sha, remote_size, what):
    if remote_size != local_size or remote_sha != local_sha:
        raise RuntimeError("{}: size/SHA-256 mismatch (this side {} bytes {}, other side {} bytes {})".format(
            what, local_size, local_sha, remote_size, remote_sha))


def push(host, local, remote_rel):
    """Copy a local file or directory to host:remote_rel (a directory file by file); verify every file."""
    local = Path(local)
    if local.is_dir():
        for path in sorted(p for p in local.rglob("*") if p.is_file()):
            push(host, path, "{}/{}".format(str(remote_rel).rstrip("/"), path.relative_to(local).as_posix()))
        return
    _run(remote_mkdir_command(host, remote_rel))
    _run(scp_command(str(local), "{}:{}".format(host, _remote(remote_rel))))
    remote_sha, remote_size = remote_info(host, remote_rel)
    _verify(sha256(local), local.stat().st_size, remote_sha, remote_size,
            "push {} -> {}:{}".format(local, host, remote_rel))


def list_remote(host, remote_rel):
    """Files under a remote directory, relative to it; empty when it is a file or missing."""
    out = _run(remote_list_command(host, remote_rel), allow_fail=True)
    return [line[2:] for line in out.splitlines() if line.startswith("./")]


def pull(host, remote_rel, local):
    """Copy host:remote_rel (file or directory, directories file by file) to local; verify every file."""
    local = Path(local)
    files = list_remote(host, remote_rel)
    if files:
        for rel in files:
            pull(host, "{}/{}".format(str(remote_rel).rstrip("/"), rel), local / rel)
        return
    local.parent.mkdir(parents=True, exist_ok=True)
    _run(scp_command("{}:{}".format(host, _remote(remote_rel)), str(local)))
    remote_sha, remote_size = remote_info(host, remote_rel)
    _verify(sha256(local), local.stat().st_size, remote_sha, remote_size,
            "pull {}:{} -> {}".format(host, remote_rel, local))
