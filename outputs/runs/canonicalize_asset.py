#!/usr/bin/env python3
"""Run dir -> upright, render-only canonical copy of a tracked mesh.

One command for what grounding needs before it can run on a hardware mesh:

    summary.json --canonical_frame--> R --render_asset(rotate=R)--> <run>/assets_canon/<name>/
                                                                    + canonical.json

Inputs all come from the run's summary.json -- the same three check_symbols
and the yaw decider use:
  objects[key].pose_on_final.cam_T_obj   registration of the tracked mesh
  packet.plane.n                         table normal (camera frame, toward camera = up)
  objects[key].decision.oa_real          Orient Anything's real-crop reading (front)

Mesh paths: --final defaults to pose_on_final.mesh (run_scene records it as
passed, a /data/any6d path that is mounted identically in PoseServer and
Ros2Bridge); --canonical (the TRELLIS GLB, for the texture) defaults to the
summary's trellis2.glb. A run registered with --final-mesh skips TRELLIS
and has no trellis2 block, so --canonical must be passed for it.

Writes, in <out>/<name>/ (default <run>/assets_canon/mug):
  render_asset.json  rotation_body_from_canon = R -- the rotation actually
                     applied to the vertices; the ONE place the grounding
                     seam reads R from to map symbols back (x_body = R x_c)
  canonical.json     how R was derived: the canonical_frame record, the
                     resolved inputs, sim provenance, the asset's obj_sha256
The driver asserts the two R's are identical before it exits 0.

Same-kind rebuilds overwrite in place (render_asset refuses to put the copy
in a tracking asset's dir). A masks tree rendered before a rebuild is stale
if obj_sha256 changed -- make_part_masks records the hash it rendered.

Runs in the ros2 container (needs the sim mount for refine_compat):

    export PYTHONPATH=/root/ros2_ws/src/manip_bridge:$PYTHONPATH
    python3 /data/runs/canonicalize_asset.py --run /data/runs/<stamp> \\
        [--key m2] [--canonical /data/meshes/<obj>.glb] [--check]

Exit: 0 written; 2 refused (missing/ambiguous inputs -- the message says
which flag fixes it).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    from manip_bridge.canonical_frame import canonical_frame
    from manip_bridge.render_asset import (RenderAssetError, build_render_asset, read_obj,
                                           render_check)
except ImportError as e:
    sys.exit(f"canonicalize_asset: cannot import manip_bridge ({e}). In the ros2 container:\n"
             "    export PYTHONPATH=/root/ros2_ws/src/manip_bridge:$PYTHONPATH")


class CanonicalizeError(Exception):
    """A refused input; the message names the flag or fix."""


def resolve_key(objects: dict, key: str | None) -> str:
    """The object to canonicalize. Never prompts: this runs in containers
    and from the seam, so ambiguity is a typed refusal listing the keys."""
    reg = sorted(k for k, v in objects.items() if isinstance(v, dict) and v.get("pose_on_final"))
    if key is not None:
        if key not in objects:
            raise CanonicalizeError(f"--key {key}: not in summary objects {sorted(objects)}")
        if key not in reg:
            raise CanonicalizeError(f"--key {key}: has no pose_on_final (registered: {reg})")
        return key
    if not reg:
        raise CanonicalizeError("no object in summary has pose_on_final -- nothing is registered "
                                "on a final mesh in this run")
    if len(reg) > 1:
        raise CanonicalizeError(f"several registered objects {reg}: pass --key")
    return reg[0]


def resolve_inputs(summary: dict, key: str | None, final: str | None,
                   canonical: str | None) -> dict:
    objects = summary.get("objects") or {}
    key = resolve_key(objects, key)
    rec = objects[key]
    pof = rec["pose_on_final"]
    T = np.asarray(pof.get("cam_T_obj"), dtype=float)
    if T.shape != (4, 4):
        raise CanonicalizeError(f"objects[{key}].pose_on_final.cam_T_obj is not 4x4")
    plane = (summary.get("packet") or {}).get("plane") or {}
    if plane.get("n") is None:
        raise CanonicalizeError("summary has no packet.plane.n (table normal) -- up cannot be "
                                "seeded; this run predates plane recording or the fit failed")
    final = final or pof.get("mesh")
    if not final:
        raise CanonicalizeError(f"objects[{key}].pose_on_final has no mesh path: pass --final")
    canonical = canonical or (rec.get("trellis2") or {}).get("glb")
    if not canonical:
        raise CanonicalizeError(
            f"objects[{key}] has no trellis2.glb (a --final-mesh run skips TRELLIS): pass "
            "--canonical /data/meshes/<the GLB final_mesh was scaled from>")
    for what, p in (("--final", final), ("--canonical", canonical)):
        if not Path(p).is_file():
            raise CanonicalizeError(f"{what} {p}: no such file (container path?)")
    return {"key": key, "cam_T_obj": T, "plane_n": np.asarray(plane["n"], float),
            "oa_real": (rec.get("decision") or {}).get("oa_real"),
            "final": str(final), "canonical": str(canonical)}


def canonicalize(run: Path, key=None, final=None, canonical=None, name=None, out=None,
                 check=False, provenance=None) -> dict:
    """The whole step; returns the canonical.json document. `provenance` is
    injectable for tests (default: refine_compat.provenance())."""
    run = Path(run)
    spath = run / "summary.json"
    try:
        summary = json.loads(spath.read_text())
    except (OSError, ValueError) as e:
        raise CanonicalizeError(f"{spath}: unreadable ({e})") from e
    inp = resolve_inputs(summary, key, final, canonical)
    name = name or Path(inp["final"]).stem.replace("final_mesh_", "")
    out_dir = (Path(out) if out else run / "assets_canon") / name

    V, F = read_obj(Path(inp["final"]))
    t0 = time.time()
    R, record = canonical_frame(V, F, inp["cam_T_obj"], inp["plane_n"], inp["oa_real"])
    t_frame = time.time() - t0

    asset = build_render_asset(Path(inp["final"]), Path(inp["canonical"]), out_dir, name,
                               rotate=R)
    applied = np.asarray(json.loads((out_dir / "render_asset.json").read_text())
                         ["rotation_body_from_canon"], float)
    if not np.array_equal(applied, R):
        raise RuntimeError("render_asset.json rotation differs from the derived R -- the "
                           "single-source-of-R invariant is broken")

    if provenance is None:
        from manip_bridge import refine_compat
        provenance = refine_compat.provenance()
    warnings = []
    if inp["oa_real"] is None:
        warnings.append("no decision.oa_real in summary: x is arbitrary (views still upright)")
    elif record["azimuth"]["alpha_reported"] not in (None, 1):
        warnings.append(f"Orient Anything alpha {record['azimuth']['alpha_reported']}: the front "
                        "is defined only up to symmetry, so the SIGN of canonical x is not "
                        "semantically fixed (nothing downstream depends on it)")
    if not record["azimuth"]["accepted"] and inp["oa_real"] is not None:
        warnings.append(f"azimuth not accepted: {record['azimuth']['note']}")
    doc = {
        **record,
        "inputs": {"run": str(run.resolve()), "summary": str(spath.resolve()),
                   "key": inp["key"], "final_mesh": inp["final"],
                   "canonical_glb": inp["canonical"],
                   "cam_T_obj": inp["cam_T_obj"].tolist(), "plane_n_cam": inp["plane_n"].tolist(),
                   "oa_real_present": inp["oa_real"] is not None},
        "asset": {"dir": str(out_dir.resolve()), "name": name,
                  "obj_sha256": asset.obj_sha256, "render_asset_json": "render_asset.json"},
        "sim": provenance,
        "warnings": warnings,
        "frame_seconds": round(t_frame, 2),
    }
    (out_dir / "canonical.json").write_text(json.dumps(doc, indent=1) + "\n")
    if check:
        from PIL import Image
        Image.fromarray(render_check(out_dir, name)).save(out_dir / f"{name}_check.png")
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path, help="run dir holding summary.json")
    ap.add_argument("--key", default=None, help="objects key (required if several registered)")
    ap.add_argument("--final", default=None, help="tracked mesh; default pose_on_final.mesh")
    ap.add_argument("--canonical", default=None,
                    help="TRELLIS GLB for the texture; default trellis2.glb (absent on "
                         "--final-mesh runs)")
    ap.add_argument("--name", default=None, help="asset name; default final mesh stem")
    ap.add_argument("--out", default=None, type=Path, help="asset root; default <run>/assets_canon")
    ap.add_argument("--check", action="store_true", help="render <name>_check.png")
    a = ap.parse_args(argv)
    try:
        doc = canonicalize(a.run, a.key, a.final, a.canonical, a.name, a.out, a.check)
    except (CanonicalizeError, RenderAssetError) as e:
        print(f"[canonicalize] REFUSED: {e}", file=sys.stderr)
        return 2
    u, az = doc["up"], doc["azimuth"]
    print(f"[canonicalize] {doc['inputs']['key']} -> {doc['asset']['dir']}")
    print(f"[canonicalize] up: {u['source']}, {u['angle_to_table_deg']:.2f} deg to table normal "
          f"(fit rms {1e3 * u['fit_rms_m']:.2f} mm, inliers {u['fit_inliers']})")
    oa = doc["oa_up_angle_deg"]
    print(f"[canonicalize] azimuth: {az['route']} accepted={az['accepted']} "
          f"alpha={az['alpha_reported']}; OA up {'n/a' if oa is None else f'{oa:.1f} deg'} off "
          "(diagnostic)")
    print(f"[canonicalize] obj sha256 {doc['asset']['obj_sha256'][:12]}; canonical.json written")
    for w in doc["warnings"]:
        print(f"[canonicalize] WARNING: {w}")
    if a.check:
        print(f"[canonicalize] check render -> {doc['asset']['dir']}/{doc['asset']['name']}_check.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
