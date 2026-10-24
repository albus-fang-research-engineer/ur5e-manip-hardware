"""outputs/runs/ground_asset.py -- the grounding seam -- and its map-back.

Three layers:
  * frames_to_body (canonical_frame.py): pure numpy, no sim.
  * the driver's gates and wiring, with ground_object STUBBED: every refusal,
    the sigma/front plumbing, exit codes. No sim, no GL.
  * end to end (sim checkout on the grounding-seam branch + a MuJoCo GL
    backend): the synthetic tilted mug canonicalized, its eight canonical
    views rendered, handle/rim masks rasterized from known geometry, the REAL
    ground_object run through grounding_compat, mapped back, and checked in
    the tracked body frame.
"""
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE = REPO_ROOT / "ros2_ws" / "src" / "manip_bridge"
SIM_DIR = Path(os.environ.get("SIM_DIR", REPO_ROOT.parent / "ur5e-manip-sim")).resolve()
_PG = SIM_DIR / "manip_sim" / "part_grounding.py"
HAVE_SEAM = _PG.is_file() and "def annotate_depth_checks" in _PG.read_text()
needs_seam = pytest.mark.skipif(not HAVE_SEAM, reason=f"no grounding-seam sim checkout at "
                                                      f"{SIM_DIR} (set SIM_DIR)")

sys.path.insert(0, str(BRIDGE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from manip_bridge.canonical_frame import BODY_COORDINATES, frames_to_body  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "outputs" / "runs" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


drv = _load("ground_asset")


def _rot(axis, deg):
    from scipy.spatial.transform import Rotation
    a = np.asarray(axis, float)
    return Rotation.from_rotvec(np.deg2rad(deg) * a / np.linalg.norm(a)).as_matrix()


R67 = _rot([0.3, -1.0, 0.2], 67.0)


# ------------------------------------------------------------------ map-back (pure)

CANON = {
    "object": "mug", "coordinates": "body frame of mjcf body 'object'",
    "provenance": {"producer": "manip_sim.part_grounding", "depth_semantics": {"inconsistent_edges": 0}},
    "points": {"handle_center": {"xyz": [-0.05, 0.002, 0.04], "status": "grounded", "comment": "line fit",
                                 "depth_check": "skip", "winding": 0.002},
               "rim_center": {"xyz": [0.0, 0.0, 0.09], "status": "grounded", "comment": "ring fit",
                              "depth_check": "skip", "winding": -0.0}},
    "axes": {"up_axis": {"xyz": [0.0, 0.0, 1.0], "status": "grounded", "comment": "canonical",
                         "sigma_deg": 2.07},
             "handle_axis": {"xyz": [-0.6, 0.0, 0.8], "status": "grounded", "comment": "line",
                             "sigma_deg": 1.4}},
    "quantities": {"rim_radius": {"value": 0.0443, "status": "grounded", "comment": "ring"}},
}


def test_frames_to_body_is_the_rotation_and_nothing_else():
    before = copy.deepcopy(CANON)
    body = frames_to_body(CANON, R67, {"tag": "x"})
    assert CANON == before                                       # input untouched
    for sec in ("points", "axes"):
        for n, e in CANON[sec].items():
            want = R67 @ np.asarray(e["xyz"])
            if sec == "axes":
                want /= np.linalg.norm(want)
            assert np.allclose(body[sec][n]["xyz"], want, atol=1e-6), n
            extra = {k: v for k, v in e.items() if k != "xyz"}
            assert {k: v for k, v in body[sec][n].items() if k != "xyz"} == extra, n
    assert np.allclose(body["axes"]["up_axis"]["xyz"], R67[:, 2], atol=1e-6)
    assert body["quantities"] == CANON["quantities"]
    assert body["coordinates"] == BODY_COORDINATES
    assert body["provenance"]["canonical"] == {"tag": "x"}
    assert body["provenance"]["depth_semantics"] == CANON["provenance"]["depth_semantics"]


def test_frames_to_body_round_trips_and_refuses_reflections():
    there = frames_to_body(CANON, R67)
    back = frames_to_body(there, R67.T)
    for n in CANON["points"]:
        assert np.allclose(back["points"][n]["xyz"], CANON["points"][n]["xyz"], atol=2e-6)
    with pytest.raises(ValueError, match="proper rotation"):
        frames_to_body(CANON, np.diag([1.0, 1.0, -1.0]))


# ------------------------------------------------------------------ the driver, ground_object stubbed

def _fake_run(tmp, *, up_source="refined", accepted=True, alpha=2, independent=True,
              parts=("handle", "rim"), written=None):
    """A run dir with a fake canonical asset + masks tree that pass every gate."""
    run = tmp / "run"
    asset = run / "assets_canon" / "mug"
    (asset / "meshes").mkdir(parents=True)
    obj = asset / "meshes" / "mug_visual.obj"
    obj.write_text("# canonical copy\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    sha = hashlib.sha256(obj.read_bytes()).hexdigest()
    (asset / "render_asset.json").write_text(json.dumps(
        {"name": "mug", "rotation_body_from_canon": R67.tolist(), "obj_sha256": sha}))
    summary = run / "summary.json"
    summary.write_text(json.dumps({"objects": {"m2": {"decision": {"front": {"independent": independent}}}}}))
    (asset / "canonical.json").write_text(json.dumps({
        "R_body_from_canon": R67.tolist(), "asset": {"obj_sha256": sha, "name": "mug"},
        "up": {"source": up_source, "fit_sigma_deg": 0.128, "angle_to_table_deg": 2.07,
               "fit_note": "" if up_source == "refined" else "snap 6.2 deg exceeds the 5 deg basin"},
        "azimuth": {"accepted": accepted, "sigma_deg": 20.03 if accepted else None,
                    "alpha_reported": alpha},
        "inputs": {"summary": str(summary), "key": "m2"}}))
    written = written or {p: True for p in parts}
    views = {v: {"parts": {p: {"written": written[p]} for p in parts}}
             for v in ("+x", "-x", "top")}
    (run / "masks_canon" / "mug").mkdir(parents=True)
    (run / "masks_canon" / "mug" / "manifest.json").write_text(json.dumps(
        {"name": "mug", "asset": {"obj_sha256": sha, "rotated": True},
         "parts": list(parts), "views": views}))
    return run


class FakeGroundingError(RuntimeError):
    pass


class StubBackend:
    """Stands in for (ground_object, GroundingError, provenance); writes a
    canonical frames.json shaped like the sim's and records the call."""

    def __init__(self, inconsistent=0, primitives=None, raise_msg=None, flipped=0, pinch=0,
                 legacy=False):
        self.inconsistent, self.raise_msg = inconsistent, raise_msg
        self.flipped, self.pinch, self.legacy = flipped, pinch, legacy
        self.primitives = primitives or {"handle": "line", "rim": "ring"}
        self.calls = []

    def ground_object(self, name, obj_dir, parts, out_dir, provider, masks_root, write, **kw):
        self.calls.append(dict(name=name, parts=parts, provider=provider, write=write, **kw))
        if self.raise_msg:
            raise FakeGroundingError(self.raise_msg)
        doc = copy.deepcopy(CANON)
        ds = {"inconsistent_edges": self.inconsistent, "boundary_edges": 3,
              "non_manifold_edges": self.pinch, "merged_vertices": 94198, "vertices": 106500}
        if not self.legacy:                          # the sim's counts since pinch/flip split
            ds.update(flipped_edges=self.flipped, pinch_edges=self.pinch)
        doc["provenance"]["depth_semantics"] = ds
        (out_dir / name).mkdir(parents=True, exist_ok=True)
        (out_dir / name / "frames.json").write_text(json.dumps(doc))
        return dict(self.primitives)

    def backend(self):
        return self.ground_object, FakeGroundingError, lambda: {"commit": "stub"}


def test_happy_path_wiring(tmp_path):
    run = _fake_run(tmp_path)
    st = StubBackend()
    code, body, rep = drv.ground(run, backend=st.backend())
    assert code == 0
    call = st.calls[0]
    assert call["provider"] == "masks" and call["write"] is True and call["parts"] == ("handle", "rim")
    assert call["up_sigma_deg"] == 2.07                          # max(0.128, 2.07), refined
    assert np.array_equal(call["front"], [1.0, 0.0, 0.0]) and call["front_sigma_deg"] == 20.03
    on_disk = json.loads((run / "grounding" / "m2" / "frames.json").read_text())
    assert on_disk == json.loads(json.dumps(body))
    assert np.allclose(body["axes"]["up_axis"]["xyz"], R67[:, 2], atol=1e-6)
    for n, p in CANON["points"].items():                         # depth_check survives the map
        assert body["points"][n]["depth_check"] == p["depth_check"]
        assert body["points"][n]["winding"] == p["winding"]
    c = body["provenance"]["canonical"]
    assert np.allclose(c["R_body_from_canon"], R67) and c["key"] == "m2"
    assert c["up_sigma"]["value_deg"] == 2.07 and "refined" in c["up_sigma"]["basis"]
    assert c["front"] == {"front_source": c["front"]["front_source"], "front_independent": True,
                          "alpha": 2, "front_sign_fixed": False, "sigma_deg": 20.03}
    assert c["front"]["front_source"].startswith("oriany_real_crop")
    assert c["sim"] == {"commit": "stub"} and c["primitives"] == {"handle": "line", "rim": "ring"}
    assert c["depth_marginal"] == {}                              # saturated windings
    # the masks used are pinned by content, not only by path: with the
    # manifest's own record of each part's prompt and its phrases file, the
    # chain from frames.json back to every SAM3 prompt is auditable
    man = Path(c["masks_manifest"])
    assert c["masks_manifest_sha256"] == hashlib.sha256(man.read_bytes()).hexdigest()


def test_marginal_winding_is_flagged_not_refused(tmp_path, capsys, monkeypatch):
    class Open(StubBackend):
        def ground_object(self, name, obj_dir, parts, out_dir, *a, **kw):
            r = super().ground_object(name, obj_dir, parts, out_dir, *a, **kw)
            p = out_dir / name / "frames.json"
            d = json.loads(p.read_text())
            d["points"]["rim_center"].update(winding=0.52, depth_check="interior")
            p.write_text(json.dumps(d))
            return r
    run = _fake_run(tmp_path)
    st = Open()
    monkeypatch.setattr(drv, "_backend", st.backend)
    assert drv.main(["--run", str(run)]) == 0
    assert "rim_center winding +0.520 is marginal" in capsys.readouterr().out
    body = json.loads((run / "grounding" / "m2" / "frames.json").read_text())
    assert body["provenance"]["canonical"]["depth_marginal"] == {"rim_center": 0.52}


@pytest.mark.parametrize("accepted, alpha, sign_fixed, front_passed", [
    (True, 1, True, True), (True, 2, False, True), (False, 0, False, False), (False, None, False, False)])
def test_front_block_is_schema_constant(tmp_path, accepted, alpha, sign_fixed, front_passed):
    run = _fake_run(tmp_path, accepted=accepted, alpha=alpha)
    st = StubBackend()
    _, body, _ = drv.ground(run, backend=st.backend())
    f = body["provenance"]["canonical"]["front"]
    assert set(f) == {"front_source", "front_independent", "alpha", "front_sign_fixed", "sigma_deg"}
    assert f["front_sign_fixed"] is sign_fixed and f["alpha"] == alpha
    assert (st.calls[0]["front"] is not None) is front_passed
    if not front_passed:
        assert f["front_source"] is None and st.calls[0]["front_sigma_deg"] == 0.0


def _corrupt(run, what):
    asset = run / "assets_canon" / "mug"
    man = run / "masks_canon" / "mug" / "manifest.json"
    edit = lambda p, fn: p.write_text(json.dumps(fn(json.loads(p.read_text()))))   # noqa: E731
    if what == "identity_asset":
        edit(asset / "render_asset.json", lambda d: {**d, "rotation_body_from_canon": None})
    elif what == "reflection":
        edit(asset / "render_asset.json",
             lambda d: {**d, "rotation_body_from_canon": np.diag([1., 1., -1.]).tolist()})
    elif what == "obj_modified":
        (asset / "meshes" / "mug_visual.obj").write_text("v 9 9 9\n")
    elif what == "canonical_R_differs":
        edit(asset / "canonical.json", lambda d: {**d, "R_body_from_canon": np.eye(3).tolist()})
    elif what == "canonical_sha_differs":
        edit(asset / "canonical.json", lambda d: {**d, "asset": {"obj_sha256": "0" * 64}})
    elif what == "no_manifest":
        man.unlink()
    elif what == "manifest_name":
        edit(man, lambda d: {**d, "name": "cup"})
    elif what == "manifest_no_asset":
        edit(man, lambda d: {**d, "asset": None})
    elif what == "manifest_sha":
        edit(man, lambda d: {**d, "asset": {"obj_sha256": "f" * 64, "rotated": True}})
    elif what == "manifest_not_rotated":
        edit(man, lambda d: {**d, "asset": {**d["asset"], "rotated": False}})
    elif what == "no_canonical_json":
        (asset / "canonical.json").unlink()


@pytest.mark.parametrize("what, match", [
    ("identity_asset", "identity"), ("reflection", "reflection"), ("obj_modified", "modified"),
    ("canonical_R_differs", "R differs"), ("canonical_sha_differs", "different geometry"),
    ("no_canonical_json", "canonical.json"), ("no_manifest", "aborted"),
    ("manifest_name", "'cup'"), ("manifest_no_asset", "no asset record"),
    ("manifest_sha", "rendered from geometry"), ("manifest_not_rotated", "not rendered from a rotated"),
])
def test_input_refusals_before_grounding(tmp_path, what, match):
    run = _fake_run(tmp_path)
    _corrupt(run, what)
    st = StubBackend()
    with pytest.raises(drv.SeamRefusal, match=match):
        drv.ground(run, backend=st.backend())
    assert st.calls == [], "refused before grounding anything"


def test_parts_gates(tmp_path):
    """A part SAM3 was never asked for is a command mismatch: refused."""
    run = _fake_run(tmp_path)
    st = StubBackend()
    with pytest.raises(drv.SeamRefusal, match=r"\['spout'\] are not in the masks manifest"):
        drv.ground(run, parts="handle,spout", backend=st.backend())
    assert st.calls == []
    run2 = _fake_run(tmp_path / "b", written={"handle": True, "rim": False})
    st = StubBackend(primitives={"handle": "line"})
    code, body, _ = drv.ground(run2, parts="handle", backend=st.backend())
    assert code == 0 and st.calls[0]["parts"] == ("handle",)
    assert body["provenance"]["canonical"]["parts_dropped"] == {}


def test_part_with_no_mask_is_dropped_not_refused(tmp_path, capsys, monkeypatch):
    """M2: the VLM named handle,body; SAM3 found body in no view. The rest is
    grounded, the drop recorded, exit 1 -- plan_scene's commands run unedited."""
    run = _fake_run(tmp_path, written={"handle": True, "rim": False})
    st = StubBackend(primitives={"handle": "line"})
    code, body, rep = drv.ground(run, parts="handle,rim", backend=st.backend())
    assert code == 1 and st.calls[0]["parts"] == ("handle",)
    c = body["provenance"]["canonical"]
    assert c["parts_requested"] == ["handle", "rim"] and c["parts_grounded"] == ["handle"]
    assert list(c["parts_dropped"]) == ["rim"]
    assert c["parts_dropped"]["rim"].startswith("no mask in any of 3 views of ")
    assert rep["dropped"] == c["parts_dropped"] and rep["ungrounded"] == []
    assert (run / "grounding" / "m2" / "frames.json").is_file()
    # the CLI says so and exits 1, still printing the check_symbols command
    monkeypatch.setattr(drv, "_backend", StubBackend(primitives={"handle": "line"}).backend)
    assert drv.main(["--run", str(run)]) == 1                # default parts = manifest's
    out = capsys.readouterr().out
    assert "DROPPED: rim (no mask in any of 3 views" in out and "check_symbols.py" in out


def test_no_requested_part_has_a_mask_is_refused(tmp_path):
    run = _fake_run(tmp_path, written={"handle": False, "rim": False})
    body = run / "grounding" / "m2" / "frames.json"
    body.parent.mkdir(parents=True)
    body.write_text("{}")                                     # a previous success
    st = StubBackend()
    with pytest.raises(drv.SeamRefusal, match=r"none of the requested parts \['handle', 'rim'\] "
                                              r"has a mask in any view"):
        drv.ground(run, backend=st.backend())
    assert st.calls == [] and not body.exists()


# run 20261023_032004, verbatim: the table normal (camera frame) and each object's
# two registrations of the same mesh, rotation part (Any6D, FoundationPose)
PLANE_N = [0.04103538267756178, -0.920205405989437, -0.38927895930337847]
REGS = {
    "mug": ([[-0.936359473802134, 0.05170173595197277, -0.3472144385254499],
             [0.10785534586549908, -0.8988822844870094, -0.42470915107123497],
             [-0.33406310809370215, -0.43512937057267, 0.8361006342993693]],
            [[-0.9263980250422371, 0.07068483626953172, -0.3698517988592083],
             [0.10754147573374444, -0.891645503914635, -0.4397762230337439],
             [-0.3608622038879263, -0.447182232731094, 0.8184170822598845]]),
    "teapot": ([[-0.5816583806894112, -0.22012436926116924, -0.7830828757105753],
                [0.48794751286645005, -0.8646709733611359, -0.11937894502026486],
                [-0.650830817387451, -0.45154110542018017, 0.6103522566966588]],
               [[-0.5897995934684152, -0.23911065175578045, -0.7713381461858463],
                [0.5173440820729975, -0.8452928389938449, -0.13354818265245327],
                [-0.6200738184188553, -0.47781388904326694, 0.6222558534468028]]),
}


def _with_registrations(run, which):
    """Put one object's real registrations + the real table plane into the
    fake run's summary under key m2 (the fixture's key)."""
    sp = run / "summary.json"
    s = json.loads(sp.read_text())
    T = [np.eye(4), np.eye(4)]
    for t, R in zip(T, REGS[which]):
        t[:3, :3] = R
    s["objects"]["m2"].update(any6d={"cam_T_obj": T[0].tolist()},
                              pose_on_final={"cam_T_obj": T[1].tolist()})
    s["packet"] = {"plane": {"n": PLANE_N}}
    sp.write_text(json.dumps(s))


def test_table_normal_up_sigma_is_the_registrations_disagreement(tmp_path):
    """The 20261023_032004 teapot: refinement rejected, sigma measured from
    Any6D vs FoundationPose = 2.36 deg, passed to ground_object and recorded."""
    run = _fake_run(tmp_path, up_source="table_normal")
    _with_registrations(run, "teapot")
    st = StubBackend()
    code, body, rep = drv.ground(run, backend=st.backend())
    assert code == 0
    assert st.calls[0]["up_sigma_deg"] == pytest.approx(2.36, abs=0.01)
    rec = body["provenance"]["canonical"]["up_sigma"]
    assert rec["value_deg"] == pytest.approx(2.36, abs=0.01)
    assert "table_normal" in rec["basis"] and "lower bound" in rec["basis"]
    # the mug's registrations, through the same rule: 1.30 deg (its refined fit gave 1.67)
    assert drv.up_disagreement(run / "summary.json", "m2")[0] == pytest.approx(2.36, abs=0.01)
    _with_registrations(run, "mug")
    assert drv.up_disagreement(run / "summary.json", "m2")[0] == pytest.approx(1.30, abs=0.01)


def test_table_normal_up_refused_without_two_registrations(tmp_path):
    run = _fake_run(tmp_path, up_source="table_normal")       # summary has no registrations
    with pytest.raises(drv.SeamRefusal, match="needs both registrations"):
        drv.ground(run, backend=StubBackend().backend())
    _with_registrations(run, "teapot")
    sp = run / "summary.json"
    s = json.loads(sp.read_text())
    s["objects"]["m2"]["pose_on_final"] = s["objects"]["m2"]["any6d"]   # identical: nothing measured
    sp.write_text(json.dumps(s))
    with pytest.raises(drv.SeamRefusal, match="not independent enough"):
        drv.ground(run, backend=StubBackend().backend())


def test_flipped_mesh_refused_and_no_stale_success(tmp_path):
    run = _fake_run(tmp_path)
    code, _, _ = drv.ground(run, backend=StubBackend().backend())
    body = run / "grounding" / "m2" / "frames.json"
    assert code == 0 and body.is_file()
    with pytest.raises(drv.SeamRefusal, match="3 flipped edges"):
        drv.ground(run, backend=StubBackend(inconsistent=3, flipped=3).backend())
    assert not body.exists(), "a refusal must not leave the previous success for check_symbols"


def test_pinch_edges_pass_and_an_old_sim_still_refuses(tmp_path):
    """The 20261023_032004 mug: 544 inconsistent directed edges, all from 272
    balanced four-face pinches, 0 flipped -- grounded, counts recorded. A sim
    without flipped_edges cannot tell the two apart: refused as before."""
    run = _fake_run(tmp_path)
    code, body, rep = drv.ground(run, backend=StubBackend(inconsistent=544, pinch=272).backend())
    assert code == 0
    ds = body["provenance"]["depth_semantics"]
    assert (ds["flipped_edges"], ds["pinch_edges"], ds["inconsistent_edges"]) == (0, 272, 544)
    with pytest.raises(drv.SeamRefusal, match="predates flipped_edges"):
        drv.ground(run, backend=StubBackend(inconsistent=3, legacy=True).backend())


def test_grounding_error_becomes_refusal(tmp_path):
    run = _fake_run(tmp_path)
    with pytest.raises(drv.SeamRefusal, match="ground_object: .*no masks"):
        drv.ground(run, backend=StubBackend(raise_msg="[ground] no masks under x").backend())


def test_ungrounded_part_exits_1_with_frames_written(tmp_path):
    run = _fake_run(tmp_path)
    code, body, rep = drv.ground(run, backend=StubBackend(
        primitives={"handle": "ungrounded", "rim": "ring"}).backend())
    assert code == 1 and rep["ungrounded"] == ["handle"]
    assert (run / "grounding" / "m2" / "frames.json").is_file()
    assert body["provenance"]["canonical"]["primitives"]["handle"] == "ungrounded"


def test_cli(tmp_path, capsys, monkeypatch):
    run = _fake_run(tmp_path)
    st = StubBackend()
    monkeypatch.setattr(drv, "_backend", st.backend)
    assert drv.main(["--run", str(run)]) == 0
    out = capsys.readouterr().out
    assert "check_symbols.py" in out and "--key m2" in out and "frames.json" in out
    # never the run root, where M1's symbols_<key> acceptance record lives
    assert f"--out-prefix {run / 'grounding'}/symbols_m2" in out
    assert "sign fixed False" in out
    _corrupt(run, "manifest_sha")
    assert drv.main(["--run", str(run)]) == 2
    assert "REFUSED" in capsys.readouterr().err


def test_asset_resolution(tmp_path):
    run = _fake_run(tmp_path)
    assert drv.resolve_asset(run, None, None) == run / "assets_canon" / "mug"
    (run / "assets_canon" / "cup").mkdir()
    (run / "assets_canon" / "cup" / "render_asset.json").write_text("{}")
    with pytest.raises(drv.SeamRefusal, match="2 canonical assets"):
        drv.resolve_asset(run, None, None)
    assert drv.resolve_asset(run, "mug", None) == run / "assets_canon" / "mug"


# ------------------------------------------------------------------ grounding_compat

@needs_seam
def test_grounding_compat_contract_and_stays_gl_free():
    code = ("import sys; import manip_bridge.grounding_compat as g; "
            "assert set(g.__all__) == {'ground_object', 'GroundingError', 'provenance', 'SOURCES'}; "
            "assert [s.rsplit('/', 1)[-1] for s in g.SOURCES] == "
            "['ground_parts.py', 'part_grounding.py', 'proposal.py', 'render_candidates.py']; "
            "assert issubclass(g.GroundingError, RuntimeError); "
            "assert 'mujoco' not in sys.modules, 'importing the compat loaded mujoco'")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(SIM_DIR), str(BRIDGE)]))
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]


# ------------------------------------------------------------------ end to end

def _gl_ok():
    try:
        import mujoco
    except ImportError:
        return False
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        m = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
        mujoco.Renderer(m, 16, 16).close()
        return True
    except Exception:
        return False


@needs_seam
def test_end_to_end_on_the_tilted_mug(tmp_path):
    """canonicalize -> render the canonical views -> handle/rim masks from known
    geometry -> REAL ground_object via grounding_compat -> map back. Checked in
    the tracked body frame against the truth the fixture was built from."""
    trimesh = pytest.importorskip("trimesh")
    if not _gl_ok():
        pytest.skip("no MuJoCo GL backend (MUJOCO_GL=egl|osmesa)")
    from PIL import Image
    from scipy.ndimage import binary_dilation
    import test_canonical_frame as tcf

    sys.path.insert(0, str(SIM_DIR))
    try:
        canon_drv = _load("canonicalize_asset")
        # --- the run: tilted textured mug -> Any6D-style final mesh -> summary
        run = tmp_path / "run"
        run.mkdir()
        Vu, F = tcf.upright_mug()
        V = Vu @ tcf.TILT.T + np.array([0.21, -0.13, 0.08])
        uv = np.stack([0.5 + 0.5 * np.tanh(5 * V[:, 0]), 0.5 + 0.5 * np.tanh(5 * V[:, 1])], 1)
        tex = Image.fromarray(np.tile(np.array([[200, 60, 60], [60, 60, 200]], np.uint8)[:, None],
                                      (16, 32, 1)).reshape(32, 32, 3))
        glb = run / "mug.glb"
        trimesh.Trimesh(V, F, process=False,
                        visual=trimesh.visual.TextureVisuals(uv=uv, image=tex)).export(glb)
        g = trimesh.load(str(glb), force="mesh")
        Vg = (np.asarray(g.vertices) - 0.5 * (np.asarray(g.vertices).min(0)
                                               + np.asarray(g.vertices).max(0))) * 0.93
        fm = g.copy()
        fm.vertices = Vg
        fm.visual = trimesh.visual.ColorVisuals(fm)
        final = run / "final_mesh_mug.obj"
        fm.export(final)
        up_true, front_true = tcf.TILT[:, 2], tcf.TILT[:, 0]       # handle is on -front
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = tcf.CAM, [0.02, -0.05, 0.48]
        n_body = tcf._rot(np.cross(up_true, front_true), 2.0) @ up_true
        oa = {"front_cam": (tcf.CAM @ tcf._rot(up_true, 15.0) @ front_true).tolist(),
              "up_cam": (tcf.CAM @ up_true).tolist(), "alpha": 2}
        (run / "summary.json").write_text(json.dumps({
            "packet": {"plane": {"n": (tcf.CAM @ n_body).tolist()}},
            "objects": {"m2": {"pose_on_final": {"cam_T_obj": T.tolist(), "mesh": str(final)},
                               "decision": {"oa_real": oa, "front": {"independent": True}}}}}))
        canon_drv.canonicalize(run, canonical=glb, provenance={"commit": "stub"})
        asset = run / "assets_canon" / "mug"
        ra = json.loads((asset / "render_asset.json").read_text())
        R = np.asarray(ra["rotation_body_from_canon"])

        # --- masks on the canonical views, from known geometry in the canonical frame
        from manip_bridge.canonical_frame import surface_samples
        from manip_bridge.render_compat import VIEW_PX, load_obj, render_depth_views
        Vc, Fc = load_obj(asset / "meshes" / "mug_visual.obj")
        cams, project, paths = render_depth_views("mug", asset, Vc, tmp_path / "views")
        P = surface_samples(Vc, Fc, 150000, np.random.default_rng(1))
        z = P[:, 2]
        bottom = P[z < z.min() + 0.004]
        axis_xy = bottom[:, :2].mean(0)
        r_xy = np.linalg.norm(P[:, :2] - axis_xy, axis=1)
        labels = {"handle": r_xy > 0.93 * 0.0505,                  # beyond the widest body wall
                  "rim": z > z.max() - 0.004}                       # the lip
        uv_by, vis_by = project(P)
        root = run / "masks_canon"
        views = {}
        for v in cams:
            d = root / "mug" / v.replace("+", "p").replace("-", "n")
            d.mkdir(parents=True)
            pix = np.round(uv_by[v]).astype(int)
            inb = (pix >= 0).all(1) & (pix < VIEW_PX).all(1) & vis_by[v]
            views[v] = {"parts": {}}
            for part, lab in labels.items():
                m = np.zeros((VIEW_PX, VIEW_PX), bool)
                sel = inb & lab
                m[pix[sel, 1], pix[sel, 0]] = True
                m = binary_dilation(m, iterations=2)
                written = bool(m.sum() > 50)
                if written:
                    Image.fromarray((m * 255).astype(np.uint8)).save(d / f"{part}.png")
                views[v]["parts"][part] = {"written": written}
        (root / "mug" / "manifest.json").write_text(json.dumps(
            {"name": "mug", "asset": {"obj_sha256": ra["obj_sha256"], "rotated": True},
             "parts": ["handle", "rim"], "views": views}))

        # --- the seam, for real
        code, body, rep = drv.ground(run)
    finally:
        sys.path.remove(str(SIM_DIR))

    assert code == 0, rep
    canon = json.loads(Path(rep["canon_path"]).read_text())
    ax, pt = body["axes"], body["points"]
    # map-back + sigma wiring
    assert np.allclose(ax["up_axis"]["xyz"], R[:, 2], atol=1e-5)
    assert np.degrees(np.arccos(np.clip(np.dot(ax["up_axis"]["xyz"], up_true), -1, 1))) < 1.0
    assert ax["up_axis"]["sigma_deg"] == pytest.approx(2.0, abs=0.1)        # max(fit, 2.0 table tilt)
    assert ax["lateral_axis"]["sigma_deg"] == pytest.approx(
        np.hypot(ax["up_axis"]["sigma_deg"], ax["front_axis"]["sigma_deg"]), abs=0.02)
    # depth_check (and winding) survive the map, symbol by symbol
    assert set(pt) == set(canon["points"]) and pt
    for n in pt:
        assert pt[n]["depth_check"] == canon["points"][n]["depth_check"], n
        assert pt[n]["winding"] == canon["points"][n]["winding"], n
        assert np.allclose(pt[n]["xyz"], R @ np.asarray(canon["points"][n]["xyz"]), atol=1e-5), n
    # grounding, judged in the TRACKED body frame against the fixture's truth
    s = 0.93
    rim_c, handle_c = np.asarray(pt["rim_center"]["xyz"]), np.asarray(pt["handle_center"]["xyz"])
    assert body["quantities"]["rim_radius"]["value"] == pytest.approx(s * 0.0476, abs=0.003)
    assert np.degrees(np.arccos(abs(np.dot(ax["rim_axis"]["xyz"], up_true)))) < 3.0
    reach = (handle_c - rim_c) - np.dot(handle_c - rim_c, up_true) * up_true
    assert np.dot(reach, -front_true) > s * 0.045, "handle_center is not on the handle side"
    # this fixture mug is a zero-thickness OPEN sheet (88 boundary edges), not a
    # closed shell like TRELLIS's: from the mouth, the opening covers about half
    # the sphere, so rim_center reads w ~ 0.5 -- and the seam flags it rather
    # than trusting the label. (On a closed rim the center reads ~0, skip: the
    # sim's closed-torus case.)
    assert 0.3 < abs(pt["rim_center"]["winding"]) < 0.7
    assert "rim_center" in body["provenance"]["canonical"]["depth_marginal"]
    assert body["provenance"]["depth_semantics"]["boundary_edges"] > 0
    assert body["provenance"]["canonical"]["front"]["front_sign_fixed"] is False   # alpha 2
    assert body["provenance"]["depth_semantics"]["inconsistent_edges"] == 0


def test_no_parts_is_refused(tmp_path):
    """The other end of the same quiet path: a manifest with no parts and no
    --parts would ground only the canonical axes and pass on up_axis alone."""
    run = _fake_run(tmp_path, parts=())
    st = StubBackend()
    with pytest.raises(drv.SeamRefusal, match="no parts to ground"):
        drv.ground(run, backend=st.backend())
    assert st.calls == []
