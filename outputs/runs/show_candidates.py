#!/usr/bin/env python3
"""Look at the proposed interaction points: every candidate in each grounded
object's WRITTEN pool, drawn on its eight canonical views (inspection only;
nothing downstream reads the output).

    <run>/<grounding>/canon/<name>/candidates.json   ground_asset's pool
            |  candidates_compat.render_pool (sim render_views + montage)
            v
    <run>/<grounding>/candidates_<name>.png   colour = class: red constructed
                                              (rim_center, spout_tip, ...), blue
                                              part samples, orange curvature,
                                              teal fps; filled = visible in that
                                              view, hollow = occluded; triad =
                                              the frames.json axes

and a table per object: id, class, part, symbol, views it is visible in. A
SURFACE point visible in no view is a problem (it can never be offered as a
mark); a constructed free-space point (rim_center, handle_center) hollow from
the side is expected.

This is the full pool. Call #2 is offered at most 8 of it (vlm_subset, biased
by the stage's parts) -- select_scene's call2/vlm/<name>/*.png shows that cut.

In the ros2 container (GL; no API key):

    export PYTHONPATH=/root/ros2_ws/src/manip_bridge:$PYTHONPATH
    python3 /data/runs/show_candidates.py --run /data/runs/<stamp>          # every grounded object
    python3 /data/runs/show_candidates.py --run /data/runs/<stamp> --names m2
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

try:
    from manip_bridge import candidates_compat as cc
except ImportError as e:
    sys.exit(f"show_candidates: cannot import manip_bridge ({e}). In the ros2 container:\n"
             "    export PYTHONPATH=/root/ros2_ws/src/manip_bridge:$PYTHONPATH")


def show(run: Path, grounding: str = "grounding_vlm", names=None) -> dict:
    """{name: (png, rows)}; rows = (id, source, part, symbol, n_views_visible)."""
    root = run / grounding / "canon"
    found = sorted(p.parent.name for p in root.glob("*/candidates.json"))
    if not found:
        raise SystemExit(f"no candidates.json under {root} -- run ground_asset.py first")
    names = list(names) if names else found
    unknown = sorted(set(names) - set(found))
    if unknown:
        raise SystemExit(f"no written pool for {unknown} under {root} (have {found})")
    out = {}
    for name in names:
        png = run / grounding / f"candidates_{name}.png"
        cands, vis = cc.render_pool(name, root / name, png)
        rows = [(int(c["id"]), c.get("source", "?"), c.get("part") or "", c.get("symbol") or "",
                 int(sum(bool(v[k]) for v in vis.values()))) for k, c in enumerate(cands)]
        out[name] = (png, rows)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--grounding", default="grounding_vlm", help="default grounding_vlm")
    ap.add_argument("--names", nargs="*", default=None, help="default: every grounded object")
    a = ap.parse_args(argv)
    for name, (png, rows) in show(a.run, a.grounding, a.names).items():
        by_src = Counter(r[1] for r in rows)
        by_part = Counter(r[2] for r in rows if r[2])
        print(f"[show-candidates] {name}: {len(rows)} candidates -> {png}")
        print(f"    by class: {dict(by_src)}   by part: {dict(by_part)}")
        print(f"    {'id':>3}  {'class':<11} {'part':<10} {'symbol':<22} views")
        for i, src, part, sym, n in rows:
            flag = "  <- visible in NO view" if n == 0 and src != "constructed" else ""
            print(f"    {i:>3}  {src:<11} {part:<10} {sym:<22} {n}/8{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
