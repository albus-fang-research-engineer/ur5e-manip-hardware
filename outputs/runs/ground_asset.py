#!/usr/bin/env python3
"""The grounding seam: canonical asset + masks tree -> frames.json in the
tracked mesh's body frame, ready for check_symbols.

    <run>/assets_canon/<name>/     canonicalize_asset.py  (upright copy, R, canonical.json)
    <masks-root>/<name>/           make_part_masks.py     (SAM3 masks on its views + manifest)
            |
            v  ground_object (sim, via grounding_compat) on the canonical copy
    <out>/canon/<name>/frames.json     canonical frame (+ candidates.json, views)
            |
            v  frames_to_body: points/axes R x, quantities unchanged
    <out>/<key>/frames.json            tracked body frame -> check_symbols

Every input is checked before anything is grounded, and each refusal names
its fix (exit 2):
  asset     render_asset.json rotation_body_from_canon must be non-null (an
            identity asset is not upright: grounding it breaks sim's UP_BODY);
            the visual OBJ must hash to the recorded obj_sha256; canonical.json
            must carry the SAME R and hash.
  masks     manifest.json must exist (an aborted make_part_masks leaves a
            manifest-less tree); its name must be the asset's, and its asset
            record must be this exact geometry (obj_sha256) and rotated; every
            part requested must be in the manifest (a part SAM3 was never asked
            for is a command mismatch, not a perception result), and at least
            one of them must have a mask in >= 1 view.
  up sigma  only from a REFINED up: sigma = max(fit_sigma_deg,
            angle_to_table_deg). A table_normal up has no trusted sigma (a
            rejected refinement; and compile_tsr reads a missing sigma as 0.0,
            "perfectly known"), so it is refused until a specimen defines the
            policy.
  mesh      after grounding, provenance.depth_semantics.flipped_edges must
            be 0 (vertices merged by position, so UV seams do not hide a
            flipped chart): depth_check labels come from the winding number,
            which needs consistent orientation. pinch_edges -- consistently
            wound sheets touching along an edge, which TRELLIS makes where
            thin parts meet -- leave the winding number exact, so they are
            recorded, not refused (the 20261023_032004 mug: 544 inconsistent
            directed edges, all from 272 balanced four-face pinches, 0
            flipped). A sim that predates flipped_edges cannot tell the two
            apart, and any inconsistent edge is refused as before.

Marginal winding (recorded and printed, never refused): a closed mesh's
winding number saturates to ~0 / ~1 within a few mm of the surface, so a
point reading |w| in MARGINAL_WINDING is looking through a hole or an open
boundary -- an open cup's mouth subtends about half the sphere, ~0.5 -- and
its depth_check label is a coin flip. Such points are listed in
provenance.canonical.depth_marginal and in the console, so a check_symbols
verdict on them is read with that in mind. On the mug's closed final_mesh
(3 boundary edges after merging) none is expected.

Front: when canonical.json's azimuth was accepted, ground_object gets
front = canonical +x with its sigma; otherwise none. provenance.canonical.front
records {front_source, front_independent, alpha, front_sign_fixed} on every
run, schema constant: front_sign_fixed is (alpha == 1) -- at alpha 2 the
sigma describes a cone around +x but the true uncertainty includes the 180
deg flip, which M2's coupling must treat as +-x equivalent.

Dropped parts: a requested part that IS in the manifest but has no mask in
any view (SAM3 found it nowhere -- the mug's "body" in M2) is dropped, not
refused: the rest is grounded and the drop is recorded in
provenance.canonical.parts_dropped {part: reason}, beside parts_requested
(what was asked -- e.g. the VLM's part list) and parts_grounded (what was
handed to ground_object). Same policy as sim plan_stages' --grounding
parts_dropped. So plan_scene's printed commands run unedited.
Whether a drop is fatal is decided downstream, by plan_scene --grounding
(sim: "whether the drop is fatal is decided by the contract checks"): a
bound role whose stage has no surviving part on its object blocks there.
Hardware binds roles structurally (no task contract), so "handle dropped,
body survives" is NOT caught at binding -- see plan_scene's docstring.

Exit: 0 every requested part grounded; 1 some part was dropped or came back
ungrounded (frames still written, recorded); 2 refused (nothing body-frame written --
a previous run's <out>/<key>/frames.json is removed first, so a refusal
never leaves a stale success behind for check_symbols).

In the ros2 container (GL for the views; the sim checkout must be on the
grounding-seam branch):

    export PYTHONPATH=/root/ros2_ws/src/manip_bridge:$PYTHONPATH
    python3 /data/runs/ground_asset.py --run /data/runs/<stamp> \\
        [--name mug] [--masks-root <run>/masks_canon] [--parts handle,rim]
    python3 /data/runs/check_symbols.py /data/runs/<stamp> --key m2 \\
        --frames /data/runs/<stamp>/grounding/m2/frames.json
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

try:
    from manip_bridge.canonical_frame import frames_to_body
    from manip_bridge.render_asset import RenderAssetError, validate_rotation
except ImportError as e:
    sys.exit(f"ground_asset: cannot import manip_bridge ({e}). In the ros2 container:\n"
             "    export PYTHONPATH=/root/ros2_ws/src/manip_bridge:$PYTHONPATH")


MARGINAL_WINDING = (0.2, 0.8)   # |w| strictly inside: the label is not decided by geometry


class SeamRefusal(Exception):
    """An input the seam will not ground from; the message names the fix."""


def _read_json(path: Path, what: str) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        raise SeamRefusal(f"{what}: {path} not found") from None
    except (OSError, ValueError) as e:
        raise SeamRefusal(f"{what}: {path} unreadable ({e})") from e


def resolve_asset(run: Path, name: str | None, asset: str | None) -> Path:
    if asset:
        return Path(asset)
    root = run / "assets_canon"
    if name:
        return root / name
    found = sorted(p for p in root.glob("*") if (p / "render_asset.json").is_file())
    if len(found) != 1:
        raise SeamRefusal(f"{root}: {'no' if not found else len(found)} canonical assets "
                          f"{[p.name for p in found]} -- pass --name (or --asset); build one "
                          "with canonicalize_asset.py")
    return found[0]


def check_asset(asset_dir: Path) -> dict:
    """render_asset.json + canonical.json + the OBJ agree on one rotated geometry."""
    ra = _read_json(asset_dir / "render_asset.json", "asset")
    if ra.get("rotation_body_from_canon") is None:
        raise SeamRefusal(f"{asset_dir} is an identity (tracking) asset, not the upright "
                          "canonical copy -- grounding it would violate sim's UP_BODY = +z. "
                          "Build the copy: canonicalize_asset.py --run <run>")
    try:
        R = validate_rotation(ra["rotation_body_from_canon"])
    except RenderAssetError as e:
        raise SeamRefusal(f"{asset_dir}/render_asset.json: {e}") from e
    name = ra.get("name") or asset_dir.name
    obj = asset_dir / "meshes" / f"{name}_visual.obj"
    if not obj.is_file():
        raise SeamRefusal(f"{obj} missing -- rebuild with canonicalize_asset.py")
    sha = hashlib.sha256(obj.read_bytes()).hexdigest()
    if sha != ra.get("obj_sha256"):
        raise SeamRefusal(f"{obj} hashes to {sha[:12]} but render_asset.json records "
                          f"{str(ra.get('obj_sha256'))[:12]} -- the asset was modified after "
                          "its build; rebuild with canonicalize_asset.py")
    can = _read_json(asset_dir / "canonical.json", "canonical record")
    if not np.array_equal(np.asarray(can.get("R_body_from_canon"), float), R):
        raise SeamRefusal(f"{asset_dir}: canonical.json R differs from render_asset.json's "
                          "rotation_body_from_canon -- rebuild with canonicalize_asset.py")
    if (can.get("asset") or {}).get("obj_sha256") != sha:
        raise SeamRefusal(f"{asset_dir}: canonical.json describes a different geometry "
                          "(obj_sha256) -- rebuild with canonicalize_asset.py")
    return {"name": name, "R": R, "sha": sha, "canonical": can}


def check_masks(masks_root: Path, name: str, sha: str, parts_arg: str | None) -> tuple:
    """(requested, kept, dropped {part: reason}, manifest path). Refuses only
    when nothing requested has a mask; a part with none anywhere is dropped."""
    man_path = masks_root / name / "manifest.json"
    if not man_path.is_file():
        raise SeamRefusal(f"{man_path} not found: a masks tree without a manifest is an "
                          "aborted or foreign run -- re-run make_part_masks.py into a fresh "
                          "--masks-root")
    man = _read_json(man_path, "masks manifest")
    fix = "re-run make_part_masks.py on this asset into a fresh --masks-root"
    if man.get("name") != name:
        raise SeamRefusal(f"{man_path}: manifest is for {man.get('name')!r}, asset is "
                          f"{name!r} -- {fix}")
    a = man.get("asset")
    if not a:
        raise SeamRefusal(f"{man_path}: no asset record (made before make_part_masks recorded "
                          f"geometry) -- {fix}")
    if a.get("obj_sha256") != sha:
        raise SeamRefusal(f"{man_path}: masks were rendered from geometry "
                          f"{str(a.get('obj_sha256'))[:12]}, the asset is {sha[:12]} -- {fix}")
    if a.get("rotated") is not True:
        raise SeamRefusal(f"{man_path}: masks were not rendered from a rotated canonical "
                          f"copy -- {fix}")
    have = list(man.get("parts") or [])
    parts = [p.strip() for p in parts_arg.split(",") if p.strip()] if parts_arg else have
    if not parts:
        raise SeamRefusal(f"no parts to ground: --parts gave none and {man_path} lists none -- "
                          "frames.json would carry the canonical axes only and pass check_symbols "
                          "on up_axis alone")
    extra = [p for p in parts if p not in have]
    if extra:
        raise SeamRefusal(f"parts {extra} are not in the masks manifest {have} -- segment "
                          "them with make_part_masks.py --parts, or drop them")
    views = man.get("views") or {}
    dropped = {p: f"no mask in any of {len(views)} views of {man_path}" for p in parts
               if not any((v.get("parts") or {}).get(p, {}).get("written") for v in views.values())}
    kept = [p for p in parts if p not in dropped]
    if not kept:
        raise SeamRefusal(f"none of the requested parts {parts} has a mask in any view "
                          f"({man_path}) -- nothing to ground; look at make_part_masks' overlays "
                          "(prompt wording, view lighting) before re-running")
    return parts, kept, dropped, man_path


def up_sigma(canonical: dict) -> tuple[float, str]:
    u = canonical["up"]
    if u.get("source") != "refined":
        raise SeamRefusal(
            f"canonical up came from the table normal (refinement rejected: "
            f"{u.get('fit_note') or 'no note'}; fit_sigma_deg {u.get('fit_sigma_deg')}, "
            f"angle_to_table_deg {u.get('angle_to_table_deg')}). A rejected refinement has no "
            "trusted sigma and the angle to the table is 0 by construction, so there is no "
            "honest up_axis sigma -- and compile_tsr would read a missing one as 0.0. The "
            "policy for this case is parked until a specimen exists; bring the numbers back.")
    fit, tab = float(u["fit_sigma_deg"]), float(u["angle_to_table_deg"])
    return max(fit, tab), (f"max(fit_sigma_deg {fit:.3f}, angle_to_table_deg {tab:.3f}) -- the "
                           "refined axis's own 1-sigma vs its disagreement with the measured "
                           "table normal; up.source refined")


def front_block(canonical: dict) -> tuple:
    """(front or None, sigma, record). Schema constant: every key every run."""
    az = canonical["azimuth"]
    alpha = az.get("alpha_reported")
    independent = None
    summary = (canonical.get("inputs") or {}).get("summary")
    key = (canonical.get("inputs") or {}).get("key")
    if summary and key:
        try:
            dec = (json.loads(Path(summary).read_text())["objects"][key].get("decision") or {})
            independent = (dec.get("front") or {}).get("independent")
        except (OSError, ValueError, KeyError):
            independent = None
    if az.get("accepted"):
        rec = {"front_source": "oriany_real_crop: decision.oa_real.front_cam through "
                               "pose_on_final, projected perpendicular to up = canonical +x",
               "front_independent": independent, "alpha": alpha,
               "front_sign_fixed": alpha == 1, "sigma_deg": az.get("sigma_deg")}
        return np.array([1.0, 0.0, 0.0]), float(az["sigma_deg"]), rec
    return None, 0.0, {"front_source": None, "front_independent": independent, "alpha": alpha,
                       "front_sign_fixed": False, "sigma_deg": None}


def _backend():
    """(ground_object, GroundingError, provenance) from the sim, via the compat."""
    try:
        from manip_bridge import grounding_compat as gc
    except ImportError as e:
        raise SeamRefusal(str(e)) from e
    return gc.ground_object, gc.GroundingError, gc.provenance


def ground(run, name=None, asset=None, masks_root=None, parts=None, out=None, backend=None):
    """The whole seam; returns (exit_code, body_frames_doc_or_None, report)."""
    run = Path(run)
    asset_dir = resolve_asset(run, name, asset)
    a = check_asset(asset_dir)
    can = a["canonical"]
    key = (can.get("inputs") or {}).get("key") or a["name"]
    out = Path(out) if out else run / "grounding"
    body_path = out / key / "frames.json"
    if body_path.exists():
        body_path.unlink()                    # a refusal must not leave a stale success
    masks_root = Path(masks_root) if masks_root else run / "masks_canon"
    requested, part_list, dropped, man_path = check_masks(masks_root, a["name"], a["sha"], parts)
    sig_up, sig_basis = up_sigma(can)
    front, sig_front, front_rec = front_block(can)

    ground_object, GroundingError, prov = backend or _backend()
    canon_out = out / "canon"
    try:
        primitives = ground_object(a["name"], asset_dir, tuple(part_list), canon_out, "masks",
                                   masks_root, True, front=front, front_sigma_deg=sig_front,
                                   up_sigma_deg=sig_up)
    except GroundingError as e:
        raise SeamRefusal(f"ground_object: {e}") from e
    canon_path = canon_out / a["name"] / "frames.json"
    canon_frames = _read_json(canon_path, "canonical frames")
    ds = (canon_frames.get("provenance") or {}).get("depth_semantics")
    if ds is None:
        raise SeamRefusal(f"{canon_path} has no provenance.depth_semantics -- the sim "
                          "checkout predates the grounding seam")
    if "flipped_edges" in ds:
        if ds["flipped_edges"] > 0:
            raise SeamRefusal(
                f"mesh orientation: {ds['flipped_edges']} flipped edges (two faces traversing "
                f"an edge the same way) after merging vertices by position ({canon_path}) -- "
                "the winding-number depth_check labels are unreliable near the flipped "
                "region. Bring the count and the per-point winding values back before "
                "setting a policy.")
    elif ds.get("inconsistent_edges", 0) > 0:
        raise SeamRefusal(
            f"mesh orientation: {ds['inconsistent_edges']} inconsistent edges after merging "
            f"vertices by position ({canon_path}), and the sim checkout predates flipped_edges, "
            "so a flip cannot be told from a pinch -- update the sim mount")

    marginal = {n: p["winding"] for n, p in canon_frames.get("points", {}).items()
                if p.get("winding") is not None
                and MARGINAL_WINDING[0] < abs(p["winding"]) < MARGINAL_WINDING[1]}
    body = frames_to_body(canon_frames, a["R"], {
        "map": "x_body = R x_c (points, axes); quantities unchanged",
        "R_body_from_canon": a["R"].tolist(),
        "key": key, "asset_dir": str(asset_dir.resolve()), "asset_obj_sha256": a["sha"],
        "canonical_json": str((asset_dir / "canonical.json").resolve()),
        "canonical_frames": str(canon_path.resolve()),
        "masks_root": str(masks_root.resolve()), "masks_manifest": str(man_path.resolve()),
        "masks_manifest_sha256": hashlib.sha256(man_path.read_bytes()).hexdigest(),
        "parts_requested": requested, "parts_grounded": part_list, "parts_dropped": dropped,
        "primitives": primitives,
        "up_sigma": {"value_deg": sig_up, "basis": sig_basis},
        "front": front_rec, "depth_marginal": marginal, "sim": prov()})
    body_path.parent.mkdir(parents=True, exist_ok=True)
    body_path.write_text(json.dumps(body, indent=2) + "\n")
    ungrounded = sorted(p for p, k in primitives.items() if k == "ungrounded")
    report = {"key": key, "name": a["name"], "body_path": str(body_path),
              "canon_path": str(canon_path), "ungrounded": ungrounded, "dropped": dropped,
              "up_sigma": sig_up, "front": front_rec, "depth_semantics": ds,
              "depth_marginal": marginal, "out": str(out)}
    return (1 if ungrounded or dropped else 0), body, report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--name", default=None, help="asset name (default: the only one in "
                                                 "<run>/assets_canon)")
    ap.add_argument("--asset", default=None, help="canonical asset dir (overrides --name)")
    ap.add_argument("--masks-root", default=None, help="default <run>/masks_canon")
    ap.add_argument("--parts", default=None, help="comma list; default the manifest's parts")
    ap.add_argument("--out", default=None, help="default <run>/grounding")
    a = ap.parse_args(argv)
    try:
        code, body, rep = ground(a.run, a.name, a.asset, a.masks_root, a.parts, a.out)
    except SeamRefusal as e:
        print(f"[ground-asset] REFUSED: {e}", file=sys.stderr)
        return 2
    print(f"[ground-asset] {rep['name']} ({rep['key']}) -> {rep['body_path']}")
    for n, p in sorted(body["points"].items()):
        print(f"    {n:18s} {p.get('depth_check', '?'):9s} winding {p.get('winding', float('nan')):+.3f}  "
              f"{p.get('comment', '')}")
    ds = rep["depth_semantics"]
    print(f"[ground-asset] mesh: {ds.get('flipped_edges', '?')} flipped / "
          f"{ds.get('pinch_edges', '?')} pinch / {ds.get('inconsistent_edges')} inconsistent / "
          f"{ds.get('boundary_edges')} boundary / {ds.get('non_manifold_edges')} non-manifold "
          f"edges ({ds.get('merged_vertices')} of {ds.get('vertices')} vertices after merging)")
    for n, w in sorted(rep["depth_marginal"].items()):
        print(f"[ground-asset] WARNING: {n} winding {w:+.3f} is marginal -- it sees a hole or an "
              "open boundary, so its depth_check label is a coin flip; read its check_symbols "
              "depth verdict with that in mind")
    f = rep["front"]
    print(f"[ground-asset] up sigma {rep['up_sigma']:.2f} deg; front "
          f"{'none' if f['front_source'] is None else 'canonical +x'} (alpha {f['alpha']}, "
          f"sign fixed {f['front_sign_fixed']}, independent {f['front_independent']})")
    for n, why in sorted(rep["dropped"].items()):
        print(f"[ground-asset] DROPPED: {n} ({why}) -- recorded in parts_dropped; exit 1")
    if rep["ungrounded"]:
        print(f"[ground-asset] UNGROUNDED: {rep['ungrounded']} (frames written; exit 1)")
    print("[ground-asset] next:\n    python3 /data/runs/check_symbols.py "
          f"{Path(a.run)} --key {rep['key']} --frames {rep['body_path']} "
          f"--out-prefix {rep['out']}/symbols_{rep['key']}")
    return code


if __name__ == "__main__":
    sys.exit(main())
