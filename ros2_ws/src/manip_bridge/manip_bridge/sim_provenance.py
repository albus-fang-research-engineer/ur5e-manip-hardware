"""Provenance of the sim code hardware imports through its *_compat.py modules.

Hardware runs against whatever the mounted sim checkout holds, including
uncommitted work, so every run records exactly which sim code it used:

    root           the sim checkout (the dir holding .git)
    commit         HEAD, or None if git or .git is unavailable
    dirty          any uncommitted change in the checkout (a recorded
                   warning, never a stop)
    dirty_sources  the imported files that differ from HEAD -- the part
                   of `dirty` that can actually change a result
    files          sha256 of each imported file, relative to root. This is
                   what makes a dirty run reproducible: it says what ran,
                   whether or not it was committed.

This module imports no sim code itself, so it is not a compat module.
"""
import hashlib
import os
import subprocess
from functools import lru_cache


def _find_root(path):
    d = os.path.dirname(os.path.abspath(path))
    while True:
        if os.path.exists(os.path.join(d, ".git")):       # dir, or file for worktrees
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def _git(root, *args):
    """git output, or None. safe.directory: the container runs as root on a
    host-owned mount, which plain git refuses as dubious ownership.
    --no-optional-locks: the mount is read-only, so status must not try to
    refresh the index."""
    try:
        r = subprocess.run(["git", "-c", f"safe.directory={root}", "--no-optional-locks",
                            "-C", root, *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout if r.returncode == 0 else None


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=None)
def provenance(sources):
    """`sources`: tuple of absolute paths of the sim files a compat module
    imports. Cached: the checkout can't change under a running process in
    any way worth re-hashing for."""
    sources = tuple(os.path.abspath(s) for s in sources)
    if not sources:
        raise ValueError("provenance() needs at least one source file")
    root = _find_root(sources[0])
    base = root if root is not None else os.path.dirname(sources[0])
    files = {os.path.relpath(s, base): _sha256(s) for s in sources}
    out = {"root": root, "commit": None, "dirty": None, "dirty_sources": None, "files": files}
    if root is None:
        return out
    head = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain")
    if head is not None:
        out["commit"] = head.strip()
    if status is not None:
        out["dirty"] = bool(status.strip())
        src_status = _git(root, "status", "--porcelain", "--", *files)
        out["dirty_sources"] = sorted(ln[3:] for ln in (src_status or "").splitlines() if ln.strip())
    return out
