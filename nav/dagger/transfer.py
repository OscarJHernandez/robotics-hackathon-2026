"""Verified copies between machines: scp, then check size and SHA-256 on the other side.

Remote paths are relative to ~/Projects/robotics-hackathon-2026 on the host. `-o BatchMode=yes` makes ssh fail
instead of asking for a password.
"""

import fnmatch
import hashlib
import os
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




def remote_mv_command(host, src_rel, dst_rel):
    return ["ssh", *SSH_OPTS, host, 'mv -f "{}" "{}"'.format(_shell_path(src_rel), _shell_path(dst_rel))]


def remote_rm_command(host, rel):
    return ["ssh", *SSH_OPTS, host, 'rm -f "{}"'.format(_shell_path(rel))]


def remote_spec(host, rel):
    """The scp remote argument: exactly host:path. Commands are argument lists (no local shell), so nothing like
    $h:path can be expanded by zsh (where ':P' is a modifier and the copy silently lands in a local folder)."""
    return host + ":" + _remote(rel)


def _push_file(host, local, remote_rel):
    part = "{}.part".format(remote_rel)
    _run(remote_mkdir_command(host, remote_rel))
    _run(scp_command(str(local), remote_spec(host, part)))
    try:
        remote_sha, remote_size = remote_info(host, part)
        _verify(sha256(local), local.stat().st_size, remote_sha, remote_size,
                "push {} -> {}:{}".format(local, host, remote_rel))
    except Exception:
        _run(remote_rm_command(host, part), allow_fail=True)
        raise
    _run(remote_mv_command(host, part, remote_rel))


def push(host, local, remote_rel):
    """Copy a local file or directory to host:remote_rel (a directory file by file). A file goes to
    <name>.part first, is checked (size, SHA-256) on the remote and only then renamed; a bad copy is removed."""
    local = Path(local)
    if local.is_dir():
        for path in sorted(p for p in local.rglob("*") if p.is_file()):
            push(host, path, "{}/{}".format(str(remote_rel).rstrip("/"), path.relative_to(local).as_posix()))
        return
    _push_file(host, local, str(remote_rel))


def list_remote(host, remote_rel):
    """Files under a remote directory, relative to it; empty when it is a file or missing."""
    out = _run(remote_list_command(host, remote_rel), allow_fail=True)
    return [line[2:] for line in out.splitlines() if line.startswith("./")]


def _pull_file(host, remote_rel, local):
    local.parent.mkdir(parents=True, exist_ok=True)
    part = local.with_name(local.name + ".part")
    try:
        _run(scp_command(remote_spec(host, remote_rel), str(part)))
        remote_sha, remote_size = remote_info(host, remote_rel)
        _verify(sha256(part), part.stat().st_size, remote_sha, remote_size,
                "pull {}:{} -> {}".format(host, remote_rel, local))
    except Exception:
        part.unlink(missing_ok=True)
        raise
    os.replace(part, local)


def pull(host, remote_rel, local):
    """Copy host:remote_rel (file or directory, directories file by file) to local. A file is downloaded to
    <local>.part, checked against the remote size and SHA-256 and only then moved into place."""
    local = Path(local)
    files = list_remote(host, remote_rel)
    if files:
        for rel in files:
            pull(host, "{}/{}".format(str(remote_rel).rstrip("/"), rel), local / rel)
        return
    _pull_file(host, str(remote_rel), local)


def _matches(rel, patterns, recursive):
    if not recursive and "/" in rel:
        return False
    if rel.endswith(".part"):
        return False
    return any(fnmatch.fnmatch(rel.rsplit("/", 1)[-1], pat) for pat in patterns)


def _remote_info_or_none(host, rel):
    try:
        return remote_info(host, rel)
    except (RuntimeError, IndexError, ValueError):
        return None


def push_dir(host, local_dir, remote_rel_dir, patterns=("*",), recursive=False):
    """Push every matching file of local_dir (one level of files; sub-folders too if recursive) through push().
    Files whose remote copy has the same size and hash are skipped. Returns [(name, bytes, "copied"|"skipped")]."""
    local_dir = Path(local_dir)
    paths = local_dir.rglob("*") if recursive else local_dir.glob("*")
    result = []
    for path in sorted(p for p in paths if p.is_file()):
        rel = path.relative_to(local_dir).as_posix()
        if not _matches(rel, patterns, recursive):
            continue
        remote_rel = "{}/{}".format(str(remote_rel_dir).rstrip("/"), rel)
        size = path.stat().st_size
        info = _remote_info_or_none(host, remote_rel)
        if info is not None and info == (sha256(path), size):
            result.append((rel, size, "skipped"))
            continue
        push(host, path, remote_rel)
        result.append((rel, size, "copied"))
    return result


def pull_dir(host, remote_rel_dir, local_dir, patterns=("*",), recursive=False):
    """Pull every matching file of the remote folder through pull(); files whose local copy has the same size and
    hash are skipped. Returns [(name, bytes, "copied"|"skipped")]."""
    local_dir = Path(local_dir)
    result = []
    for rel in sorted(list_remote(host, remote_rel_dir)):
        if not _matches(rel, patterns, recursive):
            continue
        remote_rel = "{}/{}".format(str(remote_rel_dir).rstrip("/"), rel)
        target = local_dir / rel
        info = _remote_info_or_none(host, remote_rel)
        if info is None:
            raise RuntimeError("cannot read {}:{}".format(host, remote_rel))
        if target.is_file() and (sha256(target), target.stat().st_size) == (info[0], info[1]):
            result.append((rel, info[1], "skipped"))
            continue
        pull(host, remote_rel, target)
        result.append((rel, info[1], "copied"))
    return result
