"""Render-only MJCF asset from a TRELLIS.2 reconstruction -- the object
asset dir the sim repo's grounding renderers read, built WITHOUT
`convert_asset.py`.

Why this exists. The VLM grounding steps (`scripts/ground_parts.py`,
`scripts/render_candidates.py`, `scripts/render_stage_frames.py` in
ur5e-manip-sim) render the object mesh with MuJoCo from an asset dir laid
out as

    <out>/<name>.xml                 MJCF, body "object" carries the visual geom
    <out>/meshes/<name>_visual.obj   the mesh `manip_sim.proposal.load_obj` samples

and every symbol in frames.json is expressed in the body frame of THAT
mesh. On hardware the pose tracker (FoundationPose) tracks the TRELLIS
metric GLB, so the body frame of record is the metric GLB's frame, and the
render asset must be that same geometry under the identity transform.
`convert_asset.py` cannot produce it: it recenters at the bbox centroid
unconditionally, drops the texture on OBJ export, and runs CoACD, which
nothing on hardware consumes (attached-object collision is cuRobo spheres).
Feeding it the metric GLB would offset every frames.json point by the
bbox centroid, silently -- it would present as a bad VLM pick or tracker
drift.

What the TRELLIS.2 sidecar actually writes (trellis2_server/server.py):

    <name>.glb           canonical: textured, unit AABB, the mesh TRELLIS made
    <name>_metric.glb    tm = concatenate(scene.geometry); tm.apply_scale(s);
                         tm.export(...)  -- GEOMETRY ONLY, no texture

so the textured geometry in the tracked frame is (canonical GLB x scale),
and this module builds the asset from exactly that, then VERIFIES it
against the metric GLB the tracker registered on: every vertex of one must
lie within `tol_m` of a vertex of the other (symmetric nearest-neighbour;
set-based because trimesh's load/export may reorder or merge vertices).
The verification is the guard against the A2 failure mode -- if the two
files ever disagree (different scale, a recentred copy, a re-run TRELLIS),
the build refuses rather than producing an asset in a frame nothing tracks.

Texture. MuJoCo ignores MTL, so the OBJ carries `vt` per corner and the
MJCF declares <texture type="2d"> + <material> explicitly and puts the
material on the geom. The texture file is referenced by ABSOLUTE path:
`render_candidates.build_model` loads the XML via `from_xml_string` with an
injected `<compiler meshdir=...>` and no texturedir, so a relative texture
path would resolve against the cwd. Consequence: an asset dir is bound to
the path it was built at (the same path inside Ros2Bridge, /data/runs/...);
rebuild rather than move. `render_asset.json` records both source paths,
the scale, the deviation measured, and whether a texture was found, so a
rebuild is a one-liner.

Not done here, on purpose: no collision hulls, no sites, no mass. This is
a render asset. The sim XML's robosuite sites/hulls are absent and the
sim renderers do not need them (`build_model` strips `_col_` entries
anyway).

    ros2 run manip_bridge render_asset -- \\
        --from-summary /data/runs/<stamp>/summary.json --object teapot \\
        --out /data/runs/<stamp>/assets --check
    ros2 run manip_bridge render_asset -- \\
        --canonical /data/meshes/teapot_x.glb --scale 0.1372 \\
        --metric /data/meshes/teapot_x_metric.glb --name teapot --out DIR

`--check` renders one view with MuJoCo (MUJOCO_GL=egl in Ros2Bridge; osmesa
without a GPU) to <out>/<name>_check.png -- if that image is flat grey, the
texture wiring is broken and SAM3 on the canonical renders will be too.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

DEFAULT_TOL_M = 1e-6         # vertex-set deviation the build tolerates
BODY_NAME = "object"         # what frames.json / PoseReader name the body


class RenderAssetError(RuntimeError):
    pass


@dataclass(frozen=True)
class RenderAsset:
    name: str
    out_dir: str
    xml: str
    obj: str
    texture: str | None
    canonical_glb: str
    metric_glb: str | None
    scale: float
    transform: str
    max_vertex_deviation_m: float | None
    n_vertices: int
    n_faces: int
    bounds_min: list[float]
    bounds_max: list[float]


# ------------------------------------------------------------------ loading

def load_textured_mesh(glb_path: Path):
    """The canonical GLB as ONE trimesh.Trimesh, unprocessed (no vertex
    merging -- processing would change the vertex set the deviation check
    compares). TRELLIS.2 emits a single geometry; anything else is refused
    because merging several textures into one OBJ/texture pair is not an
    identity operation."""
    import trimesh
    scene = trimesh.load(str(glb_path), process=False)
    if isinstance(scene, trimesh.Scene):
        geoms = list(scene.geometry.values())
        if len(geoms) != 1:
            raise RenderAssetError(
                f"{glb_path}: {len(geoms)} geometries; this builder handles the "
                "single-geometry GLB TRELLIS.2 writes (a multi-material atlas "
                "merge is not an identity transform)")
        mesh = geoms[0]
        # a scene node may carry a transform; TRELLIS writes identity, but
        # bake it if present so the exported vertices are the scene-frame ones
        node = next(iter(scene.graph.nodes_geometry), None)
        if node is not None:
            T, _ = scene.graph.get(node)
            if not np.allclose(T, np.eye(4)):
                mesh = mesh.copy()
                mesh.apply_transform(T)
    else:
        mesh = scene
    if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
        raise RenderAssetError(f"{glb_path}: no faces")
    return mesh


def load_metric_vertices(metric_glb: Path) -> np.ndarray:
    """Vertices of the geometry-only metric GLB, as the tracker sees them."""
    import trimesh
    m = trimesh.load(str(metric_glb), process=False, force="mesh")
    return np.asarray(m.vertices, dtype=np.float64)


def vertex_set_deviation(a: np.ndarray, b: np.ndarray) -> float:
    """max over both directions of nearest-neighbour distance: 0 iff the
    two vertex SETS coincide (order-free)."""
    from scipy.spatial import cKDTree
    da, _ = cKDTree(b).query(a, k=1)
    db, _ = cKDTree(a).query(b, k=1)
    return float(max(da.max(), db.max()))


# ------------------------------------------------------------------ texture

def _texture_image(mesh):
    """PIL image of the mesh's colour texture, or None. Handles the two
    trimesh visual kinds a GLB yields (PBRMaterial / SimpleMaterial)."""
    vis = getattr(mesh, "visual", None)
    if vis is None or vis.kind != "texture":
        return None
    mat = getattr(vis, "material", None)
    img = None
    if mat is not None:
        img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    if img is None or getattr(vis, "uv", None) is None or len(vis.uv) == 0:
        return None
    return img.convert("RGB") if img.mode != "RGB" else img


# ------------------------------------------------------------------ writers

def write_obj(path: Path, V: np.ndarray, F: np.ndarray, uv: np.ndarray | None) -> None:
    """Wavefront OBJ: `v` (exactly the mesh vertices, full precision), `vt`
    per vertex when textured, `f v/vt` (MuJoCo reads it, and the sim's
    load_obj takes the `v` index before the slash). No `mtllib`: MuJoCo
    ignores MTL and the MJCF carries the material."""
    with open(path, "w") as f:
        f.write(f"# render asset, identity transform, {len(V)} v {len(F)} f\n")
        for v in V:
            f.write(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n")
        if uv is not None:
            for t in uv:
                f.write(f"vt {t[0]:.7g} {t[1]:.7g}\n")
            for a, b, c in F + 1:
                f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")
        else:
            for a, b, c in F + 1:
                f.write(f"f {a} {b} {c}\n")


def write_mjcf(path: Path, name: str, obj_rel: str, texture_abs: Path | None) -> None:
    """The sim asset layout (`<body><body name="object"><geom group=1/>`)
    so `render_candidates.build_model` and `ground_parts.render_depth_views`
    consume it unchanged. Mesh path relative to the asset dir (build_model
    injects meshdir); texture absolute (see module docstring)."""
    root = ET.Element("mujoco", model=name)
    asset = ET.SubElement(root, "asset")
    geom_attrs = dict(type="mesh", mesh=f"{name}_visual_mesh", group="1",
                      contype="0", conaffinity="0", mass="0.0001")
    if texture_abs is not None:
        ET.SubElement(asset, "texture", name=f"{name}_tex", type="2d",
                      file=str(texture_abs))
        ET.SubElement(asset, "material", name=f"{name}_mat", texture=f"{name}_tex",
                      specular="0.1", shininess="0.1")
        geom_attrs["material"] = f"{name}_mat"
    else:
        geom_attrs["rgba"] = "0.8 0.8 0.85 1"
    ET.SubElement(asset, "mesh", name=f"{name}_visual_mesh", file=obj_rel)
    wb = ET.SubElement(root, "worldbody")
    outer = ET.SubElement(wb, "body")
    body = ET.SubElement(outer, "body", name=BODY_NAME)
    ET.SubElement(body, "geom", **geom_attrs)
    ET.indent(root)
    ET.ElementTree(root).write(path)


# ------------------------------------------------------------------ build

def build_render_asset(canonical_glb: Path, scale: float, out_dir: Path, name: str,
                       metric_glb: Path | None = None, tol_m: float = DEFAULT_TOL_M,
                       require_texture: bool = False) -> RenderAsset:
    """(canonical textured GLB x scale) -> render asset dir, verified
    against the metric GLB when given. Refuses on any deviation > tol_m,
    on a multi-geometry GLB, and (with require_texture) on a mesh with no
    colour texture."""
    canonical_glb, out_dir = Path(canonical_glb), Path(out_dir)
    if not canonical_glb.is_file():
        raise RenderAssetError(f"canonical GLB missing: {canonical_glb}")
    if not (np.isfinite(scale) and scale > 0):
        raise RenderAssetError(f"scale must be a positive finite number, got {scale!r}")

    mesh = load_textured_mesh(canonical_glb)
    V = np.asarray(mesh.vertices, dtype=np.float64) * float(scale)   # uniform scale, no recentre
    F = np.asarray(mesh.faces, dtype=np.int64)

    dev = None
    if metric_glb is not None:
        metric_glb = Path(metric_glb)
        if not metric_glb.is_file():
            raise RenderAssetError(f"metric GLB missing: {metric_glb}")
        Vm = load_metric_vertices(metric_glb)
        dev = vertex_set_deviation(V, Vm)
        if dev > tol_m:
            raise RenderAssetError(
                f"{name}: (canonical x {scale:g}) deviates from {metric_glb.name} by "
                f"{dev:.3e} m > {tol_m:.1e} m -- the asset would live in a frame the "
                "tracker does not track. Wrong scale, recentred copy, or a re-run "
                "TRELLIS output?")

    img = _texture_image(mesh)
    if img is None and require_texture:
        raise RenderAssetError(f"{name}: {canonical_glb.name} carries no colour texture")

    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    obj_path = mesh_dir / f"{name}_visual.obj"
    tex_path = None
    uv = None
    if img is not None:
        tex_path = (mesh_dir / f"{name}_texture.png").resolve()
        img.save(tex_path)
        uv = np.asarray(mesh.visual.uv, dtype=np.float64)
        if len(uv) != len(V):
            raise RenderAssetError(f"{name}: {len(uv)} uv for {len(V)} vertices")
    write_obj(obj_path, V, F, uv)
    xml_path = out_dir / f"{name}.xml"
    write_mjcf(xml_path, name, f"meshes/{obj_path.name}", tex_path)

    rec = RenderAsset(
        name=name, out_dir=str(out_dir.resolve()), xml=str(xml_path.resolve()),
        obj=str(obj_path.resolve()), texture=str(tex_path) if tex_path else None,
        canonical_glb=str(canonical_glb.resolve()),
        metric_glb=str(metric_glb.resolve()) if metric_glb else None,
        scale=float(scale),
        transform="identity: vertices = canonical GLB vertices x scale, no recentre, no rotation",
        max_vertex_deviation_m=dev, n_vertices=int(len(V)), n_faces=int(len(F)),
        bounds_min=V.min(0).tolist(), bounds_max=V.max(0).tolist())
    (out_dir / "render_asset.json").write_text(json.dumps(asdict(rec), indent=2) + "\n")
    return rec


# ------------------------------------------------------------------ check

def render_check(out_dir: Path, name: str, px: int = 320, elev_deg: float = 35.0,
                 azim_deg: float = 40.0) -> np.ndarray:
    """One MuJoCo render of the asset from a canonical-style camera, loaded
    the way `render_candidates.build_model` loads it (from_xml_string with
    an injected meshdir). Returns HxWx3 uint8."""
    import mujoco
    from scipy.spatial.transform import Rotation as R

    out_dir = Path(out_dir)
    tree = ET.parse(out_dir / f"{name}.xml")
    root = tree.getroot()
    ET.SubElement(root, "compiler", meshdir=str(out_dir.resolve()))
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth=str(px), offheight=str(px))
    ET.SubElement(vis, "headlight", ambient="0.45 0.45 0.45", diffuse="0.6 0.6 0.6")

    rec = json.loads((out_dir / "render_asset.json").read_text())
    lo, hi = np.array(rec["bounds_min"]), np.array(rec["bounds_max"])
    center, radius = 0.5 * (lo + hi), 0.5 * float(np.linalg.norm(hi - lo))
    el, az = np.radians(elev_deg), np.radians(azim_deg)
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    pos = center + 2.6 * radius * d
    # MuJoCo camera looks down its -z; build a frame with -z toward the target
    z = -(center - pos) / np.linalg.norm(center - pos)
    up = np.array([0.0, 0.0, 1.0])
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    q = R.from_matrix(np.stack([x, y, z], axis=1)).as_quat()   # xyzw
    wb = root.find("worldbody")
    ET.SubElement(wb, "camera", name="check", fovy="45",
                  pos=" ".join(f"{v:.6f}" for v in pos),
                  quat=f"{q[3]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f}")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    r = mujoco.Renderer(model, px, px)
    try:
        r.update_scene(data, camera="check")
        return r.render().copy()
    finally:
        r.close()


# ------------------------------------------------------------------ CLI

def _from_summary(summary: Path, obj: str) -> tuple[Path, Path, float]:
    """run_scene's summary.json -> (canonical glb, metric glb, scale)."""
    doc = json.loads(Path(summary).read_text())
    rec = doc.get("objects", {}).get(obj, {}).get("trellis2")
    if not rec:
        raise SystemExit(f"[render-asset] no trellis2 record for {obj!r} in {summary}")
    if not rec.get("metric_valid"):
        raise SystemExit(f"[render-asset] {obj!r}: metric scale not valid in {summary}")
    return Path(rec["glb"]), Path(rec["metric_glb"]), float(rec["scale"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--canonical", type=Path, help="textured canonical GLB")
    ap.add_argument("--metric", type=Path, default=None,
                    help="geometry-only metric GLB the tracker registered on")
    ap.add_argument("--scale", type=float, default=None, help="canonical -> metres")
    ap.add_argument("--from-summary", type=Path, default=None,
                    help="run_scene summary.json; with --object, fills the three above")
    ap.add_argument("--object", default=None)
    ap.add_argument("--name", default=None, help="asset name; default --object")
    ap.add_argument("--out", type=Path, required=True,
                    help="asset root; the asset lands in <out>/<name>/")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL_M)
    ap.add_argument("--no-metric-check", action="store_true")
    ap.add_argument("--require-texture", action="store_true")
    ap.add_argument("--check", action="store_true", help="render <name>_check.png")
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    args = ap.parse_args(argv)

    if args.from_summary:
        if not args.object:
            ap.error("--from-summary needs --object")
        canonical, metric, scale = _from_summary(args.from_summary, args.object)
    else:
        if args.canonical is None or args.scale is None:
            ap.error("--canonical and --scale (or --from-summary --object)")
        canonical, metric, scale = args.canonical, args.metric, args.scale
    name = args.name or args.object or canonical.stem
    if args.no_metric_check:
        metric = None
    elif metric is None:
        ap.error("--metric (or --no-metric-check)")

    out = args.out / name
    rec = build_render_asset(canonical, scale, out, name, metric_glb=metric,
                             tol_m=args.tol, require_texture=args.require_texture)
    print(f"[render-asset] {name}: {rec.n_vertices} v {rec.n_faces} f, scale {rec.scale:g}, "
          f"deviation {rec.max_vertex_deviation_m}, "
          f"texture {'yes' if rec.texture else 'NO'} -> {rec.out_dir}")
    if args.check:
        from PIL import Image
        img = render_check(out, name)
        p = out / f"{name}_check.png"
        Image.fromarray(img).save(p)
        print(f"[render-asset] check render -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
