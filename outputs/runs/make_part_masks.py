#!/usr/bin/env python3
"""Task D: produce the part-masks tree `ground_object --provider masks` reads.

Renders the tracked mesh's eight canonical views (sim's render_depth_views,
via render_compat -- the SAME renderer ground_object itself uses, so views
agree by construction), segments each view per part name with the SAM3
sidecar, and writes binary masks in exactly the layout sim reads:

    <masks-root>/<obj>/<view>/<part>.png       (L-mode, 0/255)

plus the rendered views and a manifest. Part names arrive as data (--parts
now; VLM call #1 later), so this driver is object-generic.

View naming: sim writes view PNGs with '+'->'p', '-'->'n' applied, and
read_mask_dir applies the SAME transform at lookup -- and silently skips
any view dir it doesn't find (grounding degrades to fewer views with no
error; SystemExit fires only when everything is empty). Mask dirs are
therefore keyed off the view PNG filenames the render actually wrote, never
by re-implementing the transform.

An empty SAM3 result writes no file but is recorded in the manifest and
printed. A part with no mask in ANY view exits nonzero: grounding cannot fit
it. Voting consequence: every rendered view gets a dir here, and sim's
read_mask_dir enters a view into masks_by_view as soon as its dir exists
(ground_parts.py:129-132), so lift_masks counts that view in `seen` for every
sample visible in it (part_grounding.py:129-132) -- a view where SAM3 found a
part nowhere is a "no" vote for that part, not an abstention. Same as the
oracle provider, which rasterizes every view.

The manifest records the asset geometry it rendered (render_asset.json's
obj_sha256 + whether it was a rotated canonical copy): views are rendered
from the asset's vertices, so a masks tree is only valid for the exact
geometry it was made from.

Runs in the ros2 container (MuJoCo + osmesa + the sim mount + zmq live
there; MUJOCO_GL=osmesa or egl per docker/Dockerfile.ros2):

    python3 /data/runs/make_part_masks.py /data/runs/<stamp>/assets_canon/mug \\
        --name mug --parts handle,rim \\
        --masks-root /data/runs/<stamp>/masks_canon --timeout-ms 180000

(/data/runs is ./outputs/runs inside Ros2Bridge; the asset is the upright
canonical copy from canonicalize_asset.py. `body` is not an M1 part: SAM3
has no noun that finds it -- see the README.)

Step-4 note for the downstream check: grounding's frames.json names symbols
<part>_center / <part>_axis (plus caller-supplied up_axis) with no
depth_check field until the seam patch emits one, so check_symbols wants
    --depth-check rim_center=skip handle_center=interior
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from manip_bridge.render_compat import VIEW_PX, load_obj, provenance, render_depth_views
    from manip_bridge.zmq_client import SidecarClient, SidecarError
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ros2_ws" / "src" / "manip_bridge"))
    from manip_bridge.render_compat import VIEW_PX, load_obj, provenance, render_depth_views
    from manip_bridge.zmq_client import SidecarClient, SidecarError

SAM3_ADDR = os.environ.get("SAM3_ADDR", "tcp://127.0.0.1:5670")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("asset_dir", help="render_asset output dir (<name>.xml + meshes/)")
    ap.add_argument("--name", required=True, help="object/asset name (mug)")
    ap.add_argument("--parts", required=True,
                    help="comma-separated part names, segmented as SAM3 text prompts")
    ap.add_argument("--masks-root", required=True,
                    help="root of the masks tree ground_object reads")
    ap.add_argument("--views-out", default=None,
                    help="where rendered views land (default <masks-root>/<name>/_views)")
    ap.add_argument("--px", type=int, default=VIEW_PX)
    ap.add_argument("--sam3", default=SAM3_ADDR)
    ap.add_argument("--timeout-ms", type=int, default=30_000)
    a = ap.parse_args(argv)

    asset_dir = Path(a.asset_dir)
    parts = [p.strip() for p in a.parts.split(",") if p.strip()]
    root = Path(a.masks_root)
    obj_out = root / a.name
    views_out = Path(a.views_out) if a.views_out else obj_out / "_views"

    mesh = asset_dir / "meshes" / f"{a.name}_visual.obj"
    if not mesh.exists():
        sys.exit(f"no {mesh} -- asset_dir must be render_asset output for --name {a.name}")
    V, _F = load_obj(mesh)

    t0 = time.time()
    cams, _project, view_paths = render_depth_views(a.name, asset_dir, V,
                                                    out_views=views_out, px=a.px)
    print(f"[views] {len(view_paths)} canonical views rendered in "
          f"{time.time() - t0:.1f}s -> {views_out}")

    sam = SidecarClient(a.sam3, timeout_ms=a.timeout_ms)
    ra_json = asset_dir / "render_asset.json"
    asset_rec = None
    if ra_json.is_file():
        ra = json.loads(ra_json.read_text())
        asset_rec = {"obj_sha256": ra.get("obj_sha256"),
                     "rotated": ra.get("rotation_body_from_canon") is not None}
    manifest = {"name": a.name, "asset_dir": str(asset_dir.resolve()), "asset": asset_rec,
                "px": a.px,
                "parts": parts, "sam3": a.sam3, "masks_root": str(root.resolve()),
                "views": {}, "sim_provenance": provenance()}
    hits = {p: 0 for p in parts}

    for vname, vpath in view_paths.items():
        rgb = np.asarray(Image.open(vpath).convert("RGB"))
        vdir = obj_out / Path(vpath).stem          # keyed off the WRITTEN filename
        vdir.mkdir(parents=True, exist_ok=True)
        vrec = {"file": Path(vpath).stem, "parts": {}}
        for part in parts:
            try:
                rep = sam.call({"cmd": "segment", "rgb": rgb, "prompt": part})
            except (TimeoutError, SidecarError) as e:
                sys.exit(f"sam3 {type(e).__name__} on {vname}/{part}: {e}")
            masks = np.asarray(rep.get("masks", []))
            if masks.size == 0 or not masks.shape[0]:
                vrec["parts"][part] = {"written": False, "instances": 0, "px": 0}
                print(f"    {vname:12s} {part:10s} EMPTY (recorded, no file)")
                continue
            m = masks[0].astype(bool)
            Image.fromarray((m * 255).astype(np.uint8)).save(vdir / f"{part}.png")
            vrec["parts"][part] = {"written": True, "instances": int(masks.shape[0]),
                                   "px": int(m.sum())}
            hits[part] += 1
            print(f"    {vname:12s} {part:10s} {int(m.sum()):6d} px"
                  + (f" ({masks.shape[0]} instances, took [0])" if masks.shape[0] > 1 else ""))
        manifest["views"][vname] = vrec

    obj_out.mkdir(parents=True, exist_ok=True)
    with open(obj_out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"[masks] {obj_out} ({sum(h for h in hits.values())} masks; manifest.json written)")

    missing = [p for p, h in hits.items() if h == 0]
    if missing:
        print(f"FAIL: no mask in any view for: {', '.join(missing)} -- grounding cannot fit "
              f"these parts (texture-driven? consider the real-frame fallback)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
