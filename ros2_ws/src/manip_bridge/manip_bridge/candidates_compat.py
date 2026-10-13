"""The ONE place hardware imports the sim's call-#2 renderer
(scripts/render_candidates.py: run_vlm -- the marked canonical views and
their manifest).

Rule: every sim module hardware uses gets its own *_compat.py, and nothing
else in this repo imports manip_sim (test/test_sim_imports.py enforces it).

GL: render_candidates imports mujoco at MODULE level, so it is imported at
call time inside render_menu, never at import of this compat -- importing
this file stays GL-free (it only locates the module).

Output directory: run_vlm writes to its module global VLM_DIR (a path
relative to the process cwd, outputs/candidates/vlm), read INSIDE the
function body (`out_dir = VLM_DIR / name`), not bound as a default
argument -- so setting the attribute for the duration of the call redirects
it. Guards, because that override has a silent failure mode:
  - the module must HAVE VLM_DIR and run_vlm; otherwise refuse loudly (a
    sim rename would make setattr create a dead attribute and the renders
    would land in the old default)
  - the original value is restored in a finally (an exception mid-render
    must not leave the sim module pointing at a run dir)
  - a stale manifest is removed BEFORE the call and the new one must exist
    after it with source == <obj_dir>/candidates.json, so a render that
    went elsewhere cannot pass on a previous run's file
The clean fix is an out_dir parameter on run_vlm; it rides with the next
sim patch that touches render_candidates (call #3's framed arm), not a
patch of its own.

The sim checkout is bind-mounted read-only at /opt/manip-sim and put on
PYTHONPATH by docker-compose (SIM_DIR, default ../ur5e-manip-sim).
"""
import importlib
import importlib.util
import json
from pathlib import Path

MODULE = "scripts.render_candidates"

try:
    _spec = importlib.util.find_spec(MODULE)       # imports the namespace pkg only
except ImportError:
    _spec = None
if _spec is None or not _spec.origin:
    raise ImportError(
        "candidates_compat: cannot locate scripts/render_candidates.py. The sim checkout must "
        "be bind-mounted at /opt/manip-sim and on PYTHONPATH, and sim's scripts/ is a "
        "NAMESPACE package, so any other scripts/ dir earlier on PYTHONPATH shadows it "
        "silently -- check inside the container:\n"
        "    ls /opt/manip-sim/scripts/render_candidates.py\n"
        "    python3 -c 'import scripts; print(scripts.__path__)'")

from manip_bridge.sim_provenance import provenance as _provenance  # noqa: E402

SOURCES = (_spec.origin,)

__all__ = ["CandidatesCompatError", "render_menu", "provenance", "SOURCES"]


class CandidatesCompatError(RuntimeError):
    """The renderer cannot be redirected or did not produce this call's manifest."""


def _load():
    """The renderer module (imports mujoco). Indirected for tests."""
    return importlib.import_module(MODULE)


def render_menu(name: str, obj_dir, parts, out_dir) -> tuple[dict, Path]:
    """run_vlm(name, obj_dir, parts) with its output under out_dir ->
    (manifest, manifest path). out_dir/<name>/{<view>.png, manifest.json}."""
    obj_dir, out_dir = Path(obj_dir), Path(out_dir)
    rc = _load()
    missing = [a for a in ("VLM_DIR", "run_vlm") if not hasattr(rc, a)]
    if missing:
        raise CandidatesCompatError(
            f"{rc.__file__} has no {missing}: the output redirect would set a dead attribute "
            "and the renders would land elsewhere -- update candidates_compat for the sim's "
            "new interface (or give run_vlm an out_dir parameter)")
    mpath = out_dir / name / "manifest.json"
    if mpath.exists():
        mpath.unlink()
    old = rc.VLM_DIR
    rc.VLM_DIR = out_dir
    try:
        rc.run_vlm(name, obj_dir, list(parts))
    finally:
        rc.VLM_DIR = old
    if not mpath.is_file():
        raise CandidatesCompatError(f"run_vlm wrote no manifest at {mpath} -- the output "
                                    "redirect did not take; the renders went elsewhere")
    man = json.loads(mpath.read_text())
    if man.get("source") != str(obj_dir / "candidates.json"):
        raise CandidatesCompatError(f"{mpath} records source {man.get('source')!r}, not "
                                    f"{obj_dir / 'candidates.json'} -- not this call's render")
    return man, mpath


def provenance():
    return _provenance(SOURCES)
