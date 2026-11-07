#!/usr/bin/env python3
"""Hand-authored arm, symbols: the sim's mesh calibration
(scripts/calibrate_frames_from_mesh.py, via pour_stages_compat) run on the
library meshes, written once per mesh as a library fixture.

    teapot  spout_tip, pour_axis, tilt_axis (= up x pour, positive tips the
            spout down), up_axis, front_axis (= pour_axis), lateral_axis
    mug     opening_center, rim_radius, up_axis

The sim's calibration assumes body +z = up. A tracked mesh's body frame is
arbitrary, so the mesh is moved into a canonical frame first (V_c = V @ R,
R = R_body_from_canon from a run's canonical.json: z = the table normal at
that registration), calibrated there, and the results are mapped back
(x_body = R x_c). The fixture is in the BODY frame of the tracked mesh, as
fixtures/README.md requires; its own up/front/lateral axes define the
canonical frame plan_stages_hw.py plans in (front = the calibrated pour
axis, so it differs from the input canonical frame by a yaw only).

The output carries the mesh sha256; plan_stages_hw.py refuses a fixture
whose sha does not match the run's tracked mesh.

In the ros2 container:

    export PYTHONPATH=/root/ros2_ws/src/manip_bridge:$PYTHONPATH
    python3 /data/runs/hand_symbols.py \\
        --canon-run /data/runs/20261031_174621 --out /data/runs/library
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

try:
    from manip_bridge import pour_stages_compat as psc
except ImportError as e:
    sys.exit(f"hand_symbols: cannot import manip_bridge ({e}). In the ros2 container:\n"
             "    export PYTHONPATH=/root/ros2_ws/src/manip_bridge:$PYTHONPATH")


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _vertices(mesh: Path) -> np.ndarray:
    import trimesh
    return np.asarray(trimesh.load(str(mesh), force="mesh", process=False).vertices, float)


def _canon_R(canonical_json: Path) -> np.ndarray:
    R = np.asarray(json.loads(canonical_json.read_text())["R_body_from_canon"], float)
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt


def _sym(v, comment):
    return {"xyz": np.round(np.asarray(v, float), 6).tolist(), "status": "calibrated",
            "comment": comment}


def teapot_symbols(V_body: np.ndarray, R: np.ndarray) -> tuple[dict, dict]:
    V_c = V_body @ R
    a_c = psc.mesh_pour_axis(V_c, np.array([1.0, 0.0, 0.0]))   # seed: canonical front
    tp = psc.calibrate_teapot(V_c, a_c)
    up_b, pour_b = R[:, 2], _unit(R @ tp["pour_axis"])
    tilt_b = _unit(R @ tp["tilt_axis"])
    note = "calibrate_frames_from_mesh (sim) in the canonical frame, mapped to body"
    sym = {
        "points": {"spout_tip": {**_sym(R @ tp["spout_tip"], note + "; centroid of the "
                                        "0.1% extremal vertices along pour_axis"),
                                 "depth_check": "surface"}},
        "axes": {"up_axis": _sym(up_b, "canonical up (the input canonical.json's table normal)"),
                 "pour_axis": _sym(pour_b, note + "; horizontal spout direction"),
                 "tilt_axis": _sym(tilt_b, "up_axis x pour_axis, positive tips the spout down"),
                 "front_axis": _sym(pour_b, "canonical frame: front = pour_axis"),
                 "lateral_axis": _sym(np.cross(up_b, pour_b), "canonical frame: up x front")},
        "quantities": {},
    }
    diag = {"pour_axis_vs_input_front_deg": float(np.degrees(np.arccos(np.clip(
                pour_b @ R[:, 0], -1, 1)))),
            "spout_tip_reach_m": float(np.linalg.norm((tp["spout_tip"] - V_c.mean(0))[:2]))}
    return sym, diag


def mug_symbols(V_body: np.ndarray, R: np.ndarray) -> tuple[dict, dict]:
    V_c = V_body @ R
    mg = psc.calibrate_mug(V_c)
    note = "calibrate_frames_from_mesh (sim) in the canonical frame, mapped to body"
    sym = {
        "points": {"opening_center": {**_sym(R @ mg["opening_center"], note + "; rim-band "
                                             "circle fit centre at rim height"),
                                      "depth_check": "skip"}},
        "axes": {"up_axis": _sym(R[:, 2], "canonical up (the input canonical.json's table normal)"),
                 "front_axis": _sym(R[:, 0], "canonical frame: input canonical.json front"),
                 "lateral_axis": _sym(R[:, 1], "canonical frame: up x front")},
        "quantities": {"rim_radius": {"value": round(float(mg["rim_radius"]), 5),
                                      "status": "calibrated", "comment": note}},
    }
    return sym, {"rim_radius_m": float(mg["rim_radius"])}


def write(obj: str, mesh: Path, canonical_json: Path, sym: dict, diag: dict, out: Path):
    R_new = np.column_stack([sym["axes"]["front_axis"]["xyz"], sym["axes"]["lateral_axis"]["xyz"],
                             sym["axes"]["up_axis"]["xyz"]])
    doc = {"object": obj, "units": "meters",
           "coordinates": f"body frame of the tracked mesh {mesh.name} (identity transform)",
           "schema": "sim frames.json (points/axes/quantities); up/front/lateral axes define "
                     "the canonical frame the hand arm plans in",
           **sym,
           "canonical": {"R_body_from_canon": np.round(R_new, 6).tolist(),
                         "convention": "columns [x=front, y=up x front, z=up] in the body frame"},
           "diagnostics": diag,
           "provenance": {"mesh": str(mesh),
                          "mesh_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest(),
                          "canonical_json": str(canonical_json),
                          "sim_module": psc.SIM_MODULE,
                          "producer": "outputs/runs/hand_symbols.py"}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"[hand_symbols] {obj}: wrote {out}")
    for k, v in sym["points"].items():
        print(f"    point {k:15s} {v['xyz']}")
    for k, v in sym["axes"].items():
        print(f"    axis  {k:15s} {v['xyz']}")
    for k, v in sym["quantities"].items():
        print(f"    qty   {k:15s} {v['value']}")
    print(f"    diagnostics {json.dumps({k: round(v, 4) for k, v in diag.items()})}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--canon-run", required=True, type=Path,
                    help="a run with assets_canon/<obj>/canonical.json for both objects")
    ap.add_argument("--teapot", default="lib_teapot")
    ap.add_argument("--mug", default="lib_mug")
    ap.add_argument("--meshes", default=Path("/data/any6d"), type=Path)
    ap.add_argument("--out", default=Path("/data/runs/library"), type=Path)
    a = ap.parse_args(argv)
    for obj, fn in ((a.teapot, teapot_symbols), (a.mug, mug_symbols)):
        mesh = a.meshes / f"{obj}.obj"
        cj = a.canon_run / "assets_canon" / obj / "canonical.json"
        for p in (mesh, cj):
            if not p.is_file():
                sys.exit(f"hand_symbols: {p} missing")
        sym, diag = fn(_vertices(mesh), _canon_R(cj))
        write(obj, mesh, cj, sym, diag, a.out / f"{obj}.symbols_hand.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
