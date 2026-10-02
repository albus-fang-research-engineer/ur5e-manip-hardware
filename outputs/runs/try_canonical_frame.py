#!/usr/bin/env python3
"""SUPERSEDED by canonicalize_asset.py (the documented path); kept as a REPL
probe only. Its input plumbing is a copy of the driver's and may drift from it.

Throwaway probe for canon step 1: run canonical_frame on a saved run and
print the record. Not part of the patch series -- step 3's driver replaces it.

    python3 try_canonical_frame.py --run /data/runs/20260928_190618 \
        [--key m1] [--mesh /data/any6d/final_mesh_mug.obj] [--out canonical_probe.json]

Inputs, all from summary.json: objects[key].pose_on_final (cam_T_obj + mesh,
the pose check_symbols judges), packet.plane.n, objects[key].decision.oa_real.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from manip_bridge.canonical_frame import canonical_frame
    from manip_bridge.render_compat import load_obj
except ImportError:
    here = Path(__file__).resolve()
    for root in [here.parents[i] for i in range(len(here.parents))]:
        cand = root / "ros2_ws" / "src" / "manip_bridge"
        if cand.is_dir():
            sys.path.insert(0, str(cand))
            break
    from manip_bridge.canonical_frame import canonical_frame
    from manip_bridge.render_compat import load_obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--key", default=None, help="objects key (default: the only one with pose_on_final)")
    ap.add_argument("--mesh", default=None, help="override pose_on_final.mesh (container path differs)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    s = json.load(open(Path(a.run) / "summary.json"))
    objs = {k: v for k, v in s.get("objects", {}).items() if v.get("pose_on_final")}
    if a.key is None:
        if len(objs) != 1:
            sys.exit(f"pass --key; objects with pose_on_final: {sorted(objs)}")
        a.key = next(iter(objs))
    rec = objs.get(a.key) or sys.exit(f"objects[{a.key}] has no pose_on_final")
    T = np.asarray(rec["pose_on_final"]["cam_T_obj"], float).reshape(4, 4)
    mesh = a.mesh or rec["pose_on_final"].get("mesh") or sys.exit("no mesh path: pass --mesh")
    plane = s.get("packet", {}).get("plane") or sys.exit("no packet.plane in summary")
    oa = (rec.get("decision") or {}).get("oa_real")
    print(f"[probe] key {a.key}  mesh {mesh}  oa_real {'present' if oa else 'MISSING'}")

    V, F = load_obj(Path(mesh))
    R, out = canonical_frame(V, F, T, plane["n"], oa)

    u, az = out["up"], out["azimuth"]
    print(f"[probe] up: source={u['source']}  angle_to_table={u['angle_to_table_deg']:.2f} deg  "
          f"fit snap={u['fit_snap_deg']:.2f} sigma={u['fit_sigma_deg']:.3f} deg "
          f"rms={1e3 * u['fit_rms_m']:.2f} mm inliers={u['fit_inliers']}")
    if u["fit_note"]:
        print(f"        fit note: {u['fit_note']}")
    oa_ang = out["oa_up_angle_deg"]
    print(f"[probe] OA up vs chosen up: {'n/a' if oa_ang is None else f'{oa_ang:.1f} deg'} (diagnostic only)")
    print(f"[probe] azimuth: route={az['route']} accepted={az['accepted']} "
          f"sigma={az['sigma_deg']} alpha={az['alpha_reported']}  {az['note']}")
    print(f"[probe] chosen up in mesh coords = {np.round(R[:, 2], 3).tolist()}, "
          f"{np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))):.1f} deg off mesh +z "
          f"(the D render measured ~67 for the mug)")
    print(f"[probe] note: {out['note']}")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
        print(f"[probe] wrote {a.out}")


if __name__ == "__main__":
    main()
