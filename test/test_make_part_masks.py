"""Offline tests for outputs/runs/make_part_masks.py -- no MuJoCo, no zmq.

The compat and sidecar modules are stubbed in sys.modules before the driver
loads (the repo's stub-sidecar pattern). The stub renderer writes view PNGs
the way sim's render_depth_views does -- filenames with '+' -> 'p' and
'-' -> 'n' applied -- and returns raw view names, so these tests pin the
load-bearing contract: mask dirs keyed off the WRITTEN filenames, because
sim's read_mask_dir applies the same transform at lookup and silently skips
any view dir it doesn't find (a half-wrong naming scheme would degrade
grounding quietly, not loudly).

Run from the repo root:  python -m pytest test/test_make_part_masks.py -v
"""
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

PIL_Image = pytest.importorskip("PIL.Image")

PX = 64
VNAMES = ["x+", "x-", "top"]          # raw sim-style names containing +/-


_STUBBED = ("manip_bridge", "manip_bridge.render_compat", "manip_bridge.zmq_client")


@pytest.fixture(autouse=True)
def _restore_stubbed_modules():
    """_stub_modules replaces the manip_bridge package itself in sys.modules
    with a plain (non-package) module; left in place it breaks every later
    in-process `import manip_bridge.<x>` in the same pytest session, so the
    outcome of other suites depended on file order. Put back whatever was
    there before each test."""
    saved = {k: sys.modules.get(k) for k in _STUBBED}
    yield
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


def _stub_modules(sam_behavior):
    """Install manip_bridge.render_compat / zmq_client stubs; return the pkg."""
    pkg = types.ModuleType("manip_bridge")
    rc = types.ModuleType("manip_bridge.render_compat")
    rc.VIEW_PX = PX

    def load_obj(path):
        V = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
        return V, np.array([[0, 1, 2]])

    def render_depth_views(name, obj_dir, V, out_views, px=PX):
        out_views = Path(out_views)
        out_views.mkdir(parents=True, exist_ok=True)
        paths = {}
        for vname in VNAMES:
            p = out_views / f"{vname.replace('+', 'p').replace('-', 'n')}.png"
            PIL_Image.fromarray(np.zeros((px, px, 3), np.uint8)).save(p)
            paths[vname] = p
        return {v: {} for v in VNAMES}, (lambda P: ({}, {})), paths

    rc.load_obj = load_obj
    rc.render_depth_views = render_depth_views
    rc.provenance = lambda: {"root": "/opt/manip-sim", "commit": "stub", "dirty": False}

    zc = types.ModuleType("manip_bridge.zmq_client")

    class SidecarError(RuntimeError):
        pass

    class SidecarClient:
        def __init__(self, addr, timeout_ms, codec="msgpack"):
            self.addr = addr

        def call(self, payload, timeout_ms=None):
            assert payload["cmd"] == "segment" and "rgb" in payload
            return sam_behavior(payload["prompt"])

    zc.SidecarClient, zc.SidecarError = SidecarClient, SidecarError
    pkg.render_compat, pkg.zmq_client = rc, zc
    sys.modules["manip_bridge"] = pkg
    sys.modules["manip_bridge.render_compat"] = rc
    sys.modules["manip_bridge.zmq_client"] = zc


def _load_driver():
    mod_path = Path(__file__).resolve().parents[1] / "outputs" / "runs" / "make_part_masks.py"
    spec = importlib.util.spec_from_file_location("make_part_masks", mod_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _full_mask():
    m = np.zeros((PX, PX), bool)
    m[10:40, 10:40] = True
    return m


def _run(tmp_path, sam_behavior, parts="handle,body"):
    _stub_modules(sam_behavior)
    drv = _load_driver()
    asset = tmp_path / "assets"
    (asset / "meshes").mkdir(parents=True)
    (asset / "meshes" / "mug_visual.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    root = tmp_path / "sam"
    rc = drv.main([str(asset), "--name", "mug", "--parts", parts,
                   "--masks-root", str(root)])
    return rc, root / "mug"


def test_layout_and_pn_transform(tmp_path):
    rc, obj = _run(tmp_path, lambda prompt: {"masks": np.stack([_full_mask()])})
    assert rc == 0
    # dirs keyed by the TRANSFORMED written names; raw names must not exist
    assert (obj / "xp" / "handle.png").exists() and (obj / "xn" / "body.png").exists()
    assert not (obj / "x+").exists() and not (obj / "x-").exists()
    # sim's exact read pattern round-trips the mask
    read = np.asarray(PIL_Image.open(obj / "xp" / "handle.png").convert("L")) > 0
    assert read.dtype == bool and np.array_equal(read, _full_mask())
    # 0/255 L-mode on disk
    raw = np.asarray(PIL_Image.open(obj / "xp" / "handle.png"))
    assert set(np.unique(raw)) <= {0, 255}


def test_empty_view_recorded_not_written(tmp_path):
    # handle empty in every view named 'x+'; present elsewhere -> exit 0
    def sam(prompt, _v=iter(range(100))):
        return {"masks": np.zeros((0, PX, PX))} if prompt == "skipme" else \
               {"masks": np.stack([_full_mask()])}

    calls = []

    def sam2(prompt):
        calls.append(prompt)
        # first view's 'handle' call comes back empty, everything else full
        if prompt == "handle" and calls.count("handle") == 1:
            return {"masks": np.zeros((0, PX, PX))}
        return {"masks": np.stack([_full_mask()])}

    rc, obj = _run(tmp_path, sam2)
    assert rc == 0
    man = json.loads((obj / "manifest.json").read_text())
    first = man["views"]["x+"]["parts"]["handle"]
    assert first == {"written": False, "instances": 0, "px": 0}
    assert not (obj / "xp" / "handle.png").exists()          # absent file...
    assert (obj / "xn" / "handle.png").exists()              # ...but found elsewhere
    assert man["views"]["x-"]["parts"]["handle"]["written"] is True


def test_part_missing_everywhere_fails(tmp_path):
    def sam(prompt):
        if prompt == "spout":
            return {"masks": np.zeros((0, PX, PX))}
        return {"masks": np.stack([_full_mask()])}

    rc, obj = _run(tmp_path, sam, parts="handle,spout")
    assert rc == 1                                            # grounding can't fit spout
    man = json.loads((obj / "manifest.json").read_text())     # manifest still written
    assert all(not v["parts"]["spout"]["written"] for v in man["views"].values())


def test_manifest_provenance_and_counts(tmp_path):
    rc, obj = _run(tmp_path, lambda p: {"masks": np.stack([_full_mask(), _full_mask()])})
    man = json.loads((obj / "manifest.json").read_text())
    assert man["sim_provenance"]["commit"] == "stub"          # repo-structure condition 2
    p = man["views"]["top"]["parts"]["handle"]
    assert p["instances"] == 2 and p["px"] == int(_full_mask().sum())
    assert man["parts"] == ["handle", "body"] and man["name"] == "mug"
