"""Render-only MJCF asset from Any6D's tracked mesh -- the object asset dir
the sim repo's grounding renderers read, built from the file the pose
tracker tracks, WITHOUT `convert_asset.py`.

Why this exists. The VLM grounding steps (`scripts/ground_parts.py`,
`scripts/render_candidates.py`, `scripts/render_stage_frames.py` in
ur5e-manip-sim) render the object mesh with MuJoCo from an asset dir laid
out as

    <out>/<name>.xml                 MJCF, body "object" carries the visual geom
    <out>/meshes/<name>_visual.obj   the mesh `manip_sim.proposal.load_obj` samples
    <out>/meshes/<name>_fp.obj       same geometry + mtllib -> textured input for FoundationPose

and every symbol in frames.json is expressed in the body frame of THAT
mesh. On hardware the body frame of record is Any6D's `final_mesh_<obj>.obj`:
FoundationPose registers and tracks on it, and Any6D's own `cam_T_obj` is
expressed in it (Any6D's last reset_object runs on the already-centred
scaled mesh, so its centring compensation is the identity -- see
taeyeopl/Any6D estimater.py, register_any6d; numerical, hence asserted
below). The render asset must therefore be that file's geometry under the
identity transform. Two things rule out the obvious shortcuts:

  * `convert_asset.py` recentres at the bbox centroid unconditionally and
    drops the texture -- every frames.json point would be offset from the
    tracked frame, silently.
  * `final_mesh` is NOT `canonical x one scale`: Any6D bbox-centres and
    scales each axis by its own factor (measured on the mug run:
    diag(0.922, 0.664, 1.572) with 1e-8 m residual), so nothing can be
    rebuilt from the canonical GLB plus a number.

What survives Any6D unchanged is the vertex ORDER and the FACES: every step
does `mesh.vertices = ...` and nothing else (verified in estimater.py), and
the trimesh OBJ export writes vertices in order. So:

    geometry  = final_mesh vertices + faces, verbatim (parsed from the OBJ
                text with the sim's own `v` / first-face-index rule -- no
                library reprocessing in between)
    texture   = per-vertex uv + image from the canonical TRELLIS GLB, carried
                over BY VERTEX INDEX (final_mesh's own OBJ is untextured or
                points at a shared material.png Any6D overwrites per object)

and the build REFUSES unless the correspondence is proven:

    1. same vertex count and identical face arrays;
    2. final == diag(s) . (canonical - c) with residual <= tol_m (1 um),
       s > 0 on every axis -- a scrambled order, a rotated copy, or a
       different reconstruction all fail this; s and c are recorded;
    3. final_mesh's bbox centre == 0 within centre_tol_m (0.1 mm) -- the
       condition under which Any6D's and FoundationPose's poses are in
       this file's frame.

The canonical GLB is loaded exactly as Any6D loads its `mesh=` input
(`trimesh.load(path, force="mesh")`, default processing) so both sides see
the same vertex order; checks 1-2 catch a divergence between trimesh
versions if there ever is one.

Texture wiring. MuJoCo ignores MTL, so the OBJ carries `vt` per corner and
the MJCF declares <texture type="2d"> + <material> explicitly and puts the
material on the geom. The texture file is referenced by ABSOLUTE path:
`render_candidates.build_model` loads the XML via `from_xml_string` with an
injected `<compiler meshdir=...>` and no texturedir, so a relative texture
path would resolve against the cwd. Consequence: an asset dir is bound to
the path it was built at (the same path inside Ros2Bridge, /data/runs/...);
rebuild rather than move. `render_asset.json` records both source paths,
the fitted s and c, the residuals, and whether a texture was found.

Canonical copy (`rotate=R`, CLI `--rotate-from JSON`). The sim's grounding
stack assumes assets are converted upright (ground_parts.py UP_BODY = +z,
canonical_cameras look-at z-up); a TRELLIS frame is not (the mug's up is 67
deg off +z). With `rotate`, the same build writes a RENDER-ONLY copy whose
vertices are `final_mesh @ R` (canonical_frame.to_canonical: R's columns are
the canonical front/lateral/up axes in final_mesh coordinates; rotation about
the body origin, no re-centring, so the map back is exactly x_body = R x_c).
  * every correspondence / centre proof above still runs on the UNROTATED
    final_mesh -- they are claims about the tracked file, not the copy;
  * faces, uv and texture are written unchanged; the OBJ carries no `vn`, so
    MuJoCo computes normals from the rotated geometry;
  * no `<name>_fp.obj` / `.mtl`: the copy can never become a tracking mesh;
  * `render_asset.json` records `rotation_body_from_canon` (null on identity
    builds) -- the rotation actually applied to the vertices, and the one the
    grounding seam reads to map symbols back -- plus `obj_sha256` of the
    written visual OBJ, so a masks tree rendered from an earlier geometry
    can be told apart from this one;
  * an output dir holds ONE kind: a rotated build refuses a dir holding a
    tracking asset (identity json or `_fp.obj`), an identity build refuses a
    rotated dir; same-kind rebuilds overwrite in place.
Every build reads its written OBJ back with `read_obj` and refuses unless the
faces are identical and the vertices match what was meant to be written
within tol_m (`%.9g` keeps ~1e-10 m for a 0.1 m object).

Not done here, on purpose: no collision hulls, no sites, no mass. This is
a render asset; `build_model` strips `_col_` entries anyway.

    ros2 run manip_bridge render_asset -- \\
        --from-summary /data/runs/<stamp>/summary.json --object mug \\
        --out /data/runs/<stamp>/assets --check
    ros2 run manip_bridge render_asset -- \\
        --final /data/any6d/final_mesh_mug.obj \\
        --canonical /data/meshes/mug_<stamp>.glb --name mug --out DIR --check
    ros2 run manip_bridge render_asset -- \\
        --final /data/any6d/final_mesh_mug.obj \\
        --canonical /data/meshes/mug_<stamp>.glb --name mug \\
        --out /data/runs/<stamp>/assets_canon --rotate-from canonical.json --check

`--check` renders one view with MuJoCo (MUJOCO_GL=egl in Ros2Bridge; osmesa
without a GPU) to <out>/<name>_check.png -- if that image is flat grey, the
texture wiring is broken and SAM3 on the canonical renders will be too.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from manip_bridge.canonical_frame import to_canonical

DEFAULT_TOL_M = 1e-6          # per-axis affine fit residual the build tolerates
DEFAULT_CENTRE_TOL_M = 1e-4   # |bbox centre| of final_mesh -> pose frame == file frame
ROTATION_TOL = 1e-6           # |R^T R - I|, |det R - 1| a canonical rotation may carry
ROTATION_KEY = "R_body_from_canon"   # canonical_frame's record key, read by --rotate-from
BODY_NAME = "object"          # what frames.json / PoseReader name the body


class RenderAssetError(RuntimeError):
    pass


@dataclass(frozen=True)
class RenderAsset:
    name: str
    out_dir: str
    xml: str
    obj: str
    texture: str | None
    fp_obj: str | None
    final_mesh: str
    canonical_glb: str
    transform: str
    scale_xyz: list[float]
    centre_xyz: list[float]
    fit_residual_m: float
    bbox_centre_m: list[float]
    n_vertices: int
    n_faces: int
    bounds_min: list[float]
    bounds_max: list[float]
    rotation_body_from_canon: list[list[float]] | None = None
    obj_sha256: str = ""


# ------------------------------------------------------------------ loading

def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (every `v` line, in file order, full text precision) and
    triangle faces (first index of each face token, 1-based -> 0-based,
    negative = relative) -- the same rule as ur5e-manip-sim's
    manip_sim.proposal.load_obj, so what this returns is what the grounding
    code will sample. No library reprocessing: no merge, no reorder."""
    vs, fs = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                vs.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = []
                for tok in line.split()[1:]:
                    i = int(tok.split("/")[0])
                    idx.append(i - 1 if i > 0 else len(vs) + i)
                for k in range(1, len(idx) - 1):
                    fs.append([idx[0], idx[k], idx[k + 1]])
    if not vs:
        raise RenderAssetError(f"{path}: no vertices")
    if not fs:
        raise RenderAssetError(f"{path}: no faces")
    return np.asarray(vs, dtype=np.float64), np.asarray(fs, dtype=np.int64)


def load_canonical_as_any6d(glb_path: Path):
    """The canonical GLB exactly as any6d_server loads its `mesh=` input:
    trimesh.load(path, force="mesh") with default processing. Any6D then
    only ever replaces .vertices, so this object's vertex order and faces
    are final_mesh's."""
    import trimesh
    mesh = trimesh.load(str(glb_path), force="mesh")
    if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
        raise RenderAssetError(f"{glb_path}: no faces")
    return mesh


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


# ------------------------------------------------------------------ checks

def fit_axis_scale(canonical: np.ndarray, final: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-axis least squares final_k = s_k * canonical_k + b_k, index
    correspondence assumed. Returns (s, c, max_abs_residual) with
    c = -b / s so that final = diag(s) . (canonical - c)."""
    s = np.empty(3)
    b = np.empty(3)
    for k in range(3):
        A = np.stack([canonical[:, k], np.ones(len(canonical))], axis=1)
        (s[k], b[k]), *_ = np.linalg.lstsq(A, final[:, k], rcond=None)
    pred = canonical * s + b
    resid = float(np.abs(pred - final).max())
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(s != 0, -b / s, np.nan)
    return s, c, resid


def check_correspondence(name: str, V_final: np.ndarray, F_final: np.ndarray,
                         V_canon: np.ndarray, F_canon: np.ndarray, tol_m: float
                         ) -> tuple[np.ndarray, np.ndarray, float]:
    if len(V_final) != len(V_canon):
        raise RenderAssetError(
            f"{name}: final_mesh has {len(V_final)} vertices, canonical GLB {len(V_canon)} "
            "-- not the same reconstruction, or one side was reprocessed")
    if F_final.shape != F_canon.shape or not np.array_equal(F_final, F_canon):
        raise RenderAssetError(
            f"{name}: face arrays differ ({F_final.shape} vs {F_canon.shape}) -- vertex "
            "index correspondence cannot be assumed")
    s, c, resid = fit_axis_scale(V_canon, V_final)
    if not np.all(np.isfinite(s)) or np.any(s <= 0):
        raise RenderAssetError(f"{name}: per-axis scale {s} not positive -- flipped or degenerate")
    if resid > tol_m:
        raise RenderAssetError(
            f"{name}: final_mesh is not diag(s).(canonical - c): residual {resid:.3e} m > "
            f"{tol_m:.1e} m (s={np.round(s, 4).tolist()}). Scrambled order, a rotated copy, "
            "or a different TRELLIS run?")
    return s, c, resid


def validate_rotation(R) -> np.ndarray:
    """A proper rotation or RenderAssetError: 3x3, orthonormal, det +1. A
    reflection would mirror the mesh (handle on the wrong side) and still
    render plausibly, so it is refused here rather than noticed later."""
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3) or not np.all(np.isfinite(R)):
        raise RenderAssetError(f"rotation must be a finite 3x3 matrix, got shape {R.shape}")
    err = float(np.abs(R.T @ R - np.eye(3)).max())
    if err > ROTATION_TOL:
        raise RenderAssetError(f"rotation is not orthonormal (|R^T R - I| = {err:.2e})")
    if abs(np.linalg.det(R) - 1.0) > ROTATION_TOL:
        raise RenderAssetError(f"rotation has det {np.linalg.det(R):+.6f}: a reflection, "
                               "not a rotation")
    return R


def check_dir_kind(out_dir: Path, name: str, rotated: bool) -> None:
    """One kind of asset per dir. A rotated visual mesh next to a stale
    identity `_fp.obj` (or the reverse) is the hazard: tracking and grounding
    would read different frames from one directory. Same-kind rebuilds
    (e.g. a new R) overwrite in place."""
    rec_path = out_dir / "render_asset.json"
    has_prev, prev_rot = rec_path.is_file(), None
    if has_prev:
        try:
            prev_rot = json.loads(rec_path.read_text()).get("rotation_body_from_canon")
        except (OSError, ValueError) as e:
            raise RenderAssetError(f"{rec_path}: unreadable ({e}); remove the dir and rebuild") from e
    fp = out_dir / "meshes" / f"{name}_fp.obj"
    if rotated and (fp.exists() or (has_prev and prev_rot is None)):
        raise RenderAssetError(
            f"{out_dir} holds a tracking (identity) asset; the canonical copy must not share its "
            "dir -- write it elsewhere (e.g. <run>/assets_canon)")
    if not rotated and has_prev and prev_rot is not None:
        raise RenderAssetError(
            f"{out_dir} holds a rotated canonical copy; an identity (tracking) asset must not "
            "share its dir -- write it elsewhere (e.g. <run>/assets)")


def rotation_from_json(path: Path) -> np.ndarray:
    """R from a canonical_frame record (canonical.json, or a probe dump)."""
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise RenderAssetError(f"--rotate-from {path}: unreadable ({e})") from e
    if not isinstance(doc, dict) or ROTATION_KEY not in doc:
        raise RenderAssetError(f"--rotate-from {path}: no {ROTATION_KEY!r} key -- not a "
                               "canonical_frame record")
    return validate_rotation(doc[ROTATION_KEY])


# ------------------------------------------------------------------ writers

def write_obj(path: Path, V: np.ndarray, F: np.ndarray, uv: np.ndarray | None,
              header: str = "identity transform of final_mesh") -> None:
    """Wavefront OBJ: `v` (exactly the final_mesh vertices), `vt` per vertex
    when textured, `f v/vt` (MuJoCo reads it; the sim's load_obj takes the
    `v` index before the slash). No `mtllib`: MuJoCo ignores MTL and the
    MJCF carries the material."""
    with open(path, "w") as f:
        f.write(f"# render asset, {header}, {len(V)} v {len(F)} f\n")
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


def write_fp_obj(path: Path, V: np.ndarray, F: np.ndarray, uv: np.ndarray, texture_name: str) -> None:
    """The same geometry as write_obj, but with `mtllib`/`usemtl` and an MTL
    whose map_Kd points at the texture -- what trimesh (and therefore
    FoundationPose's pose_server, which loads with trimesh.load(force="mesh"))
    needs to see a TextureVisuals with material.image set. MuJoCo ignores
    MTL, so the render asset's own OBJ stays MTL-free; this file exists so
    FoundationPose's scorer (c_in = RGB + XYZ) gets the RGB channel, which on a
    yaw-symmetric body is the only cue that can break the yaw tie. Identical
    vertices and faces: registering on this file IS registering on final_mesh."""
    mtl = path.with_suffix(".mtl")
    with open(path, "w") as f:
        f.write("# FoundationPose-facing textured copy of the render asset, identity geometry\n")
        f.write(f"mtllib {mtl.name}\nusemtl textured\n")
        for v in V:
            f.write(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n")
        for t in uv:
            f.write(f"vt {t[0]:.7g} {t[1]:.7g}\n")
        for a, b, c in F + 1:
            f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")
    mtl.write_text(f"newmtl textured\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nmap_Kd {texture_name}\n")


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

def build_render_asset(final_mesh: Path, canonical_glb: Path, out_dir: Path, name: str,
                       tol_m: float = DEFAULT_TOL_M,
                       centre_tol_m: float = DEFAULT_CENTRE_TOL_M,
                       require_texture: bool = False,
                       rotate: np.ndarray | None = None) -> RenderAsset:
    """final_mesh (tracked geometry) + canonical GLB (texture by index) ->
    render asset dir. Refuses on any failed correspondence or centre check,
    and (with require_texture) on a canonical mesh with no colour texture.
    With `rotate` (R, columns = canonical axes in final_mesh coords), writes
    the render-only canonical copy final_mesh @ R instead -- see the module
    docstring."""
    final_mesh, canonical_glb, out_dir = Path(final_mesh), Path(canonical_glb), Path(out_dir)
    R = None if rotate is None else validate_rotation(rotate)
    if not final_mesh.is_file():
        raise RenderAssetError(f"final_mesh missing: {final_mesh}")
    if not canonical_glb.is_file():
        raise RenderAssetError(f"canonical GLB missing: {canonical_glb}")

    V, F = read_obj(final_mesh)
    canon = load_canonical_as_any6d(canonical_glb)
    Vc = np.asarray(canon.vertices, dtype=np.float64)
    Fc = np.asarray(canon.faces, dtype=np.int64)
    s, c, resid = check_correspondence(name, V, F, Vc, Fc, tol_m)

    bbox_centre = 0.5 * (V.min(0) + V.max(0))
    if np.abs(bbox_centre).max() > centre_tol_m:
        raise RenderAssetError(
            f"{name}: final_mesh bbox centre {np.round(bbox_centre, 5).tolist()} m is not 0 "
            f"(tol {centre_tol_m:g}) -- Any6D's pose would not be in this file's frame. "
            "Not an Any6D final_mesh export?")

    img = _texture_image(canon)
    if img is None and require_texture:
        raise RenderAssetError(f"{name}: {canonical_glb.name} carries no colour texture")

    # every proof above is on the UNROTATED final_mesh; rotation is a write-time step
    V_out = V if R is None else to_canonical(V, R)
    check_dir_kind(out_dir, name, rotated=R is not None)

    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    stale_check = out_dir / f"{name}_check.png"
    if stale_check.exists():
        stale_check.unlink()          # a check render of an earlier geometry would mislead
    obj_path = mesh_dir / f"{name}_visual.obj"
    tex_path = None
    uv = None
    if img is not None:
        uv = np.asarray(canon.visual.uv, dtype=np.float64)
        if len(uv) != len(V):
            raise RenderAssetError(f"{name}: {len(uv)} uv for {len(V)} vertices")
        tex_path = (mesh_dir / f"{name}_texture.png").resolve()
        img.save(tex_path)
    write_obj(obj_path, V_out, F, uv,
              header="identity transform of final_mesh" if R is None else
                     "CANONICAL COPY final_mesh @ R (render-only, not a tracking mesh)")
    V_back, F_back = read_obj(obj_path)
    if F_back.shape != F.shape or not np.array_equal(F_back, F):
        raise RenderAssetError(f"{name}: written OBJ faces do not read back identically")
    w_resid = float(np.abs(V_back - V_out).max())
    if w_resid > tol_m:
        raise RenderAssetError(f"{name}: written OBJ vertices read back {w_resid:.2e} m off "
                               f"(tol {tol_m:.1e})")
    obj_sha = hashlib.sha256(obj_path.read_bytes()).hexdigest()
    fp_obj = None
    if uv is not None and R is None:
        fp_obj = mesh_dir / f"{name}_fp.obj"
        write_fp_obj(fp_obj, V, F, uv, tex_path.name)
    xml_path = out_dir / f"{name}.xml"
    write_mjcf(xml_path, name, f"meshes/{obj_path.name}", tex_path)

    rec = RenderAsset(
        name=name, out_dir=str(out_dir.resolve()), xml=str(xml_path.resolve()),
        obj=str(obj_path.resolve()), texture=str(tex_path) if tex_path else None,
        fp_obj=str(fp_obj.resolve()) if fp_obj else None,
        final_mesh=str(final_mesh.resolve()), canonical_glb=str(canonical_glb.resolve()),
        transform=("identity: vertices = final_mesh vertices verbatim; "
                   "final = diag(scale_xyz) . (canonical - centre_xyz) is the recorded fit"
                   if R is None else
                   "canonical copy: vertices = final_mesh @ rotation_body_from_canon (columns = "
                   "canonical axes in final_mesh coords); map back x_body = R x_c; render-only, "
                   "NOT a tracking asset; scale/centre fit is of the unrotated final_mesh"),
        scale_xyz=s.tolist(), centre_xyz=c.tolist(), fit_residual_m=resid,
        bbox_centre_m=bbox_centre.tolist(),
        n_vertices=int(len(V)), n_faces=int(len(F)),
        bounds_min=V_out.min(0).tolist(), bounds_max=V_out.max(0).tolist(),
        rotation_body_from_canon=None if R is None else R.tolist(),
        obj_sha256=obj_sha)
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
    z = -(center - pos) / np.linalg.norm(center - pos)     # camera looks down -z
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

def from_summary(summary: Path, obj: str) -> tuple[Path, Path]:
    """run_scene's summary.json -> (final_mesh, canonical glb). Paths are
    the container paths the sidecars reported; both are mounted in
    Ros2Bridge (/data/any6d, /data/meshes)."""
    doc = json.loads(Path(summary).read_text())
    rec = doc.get("objects", {}).get(obj, {})
    a6d, t2 = rec.get("any6d"), rec.get("trellis2")
    if not a6d or not a6d.get("mesh"):
        raise RenderAssetError(f"no any6d.mesh for {obj!r} in {summary}")
    if not t2 or not t2.get("glb"):
        raise RenderAssetError(f"no trellis2.glb for {obj!r} in {summary}")
    if a6d.get("source", "trellis") != "trellis":
        raise RenderAssetError(
            f"{obj!r}: any6d ran with source={a6d.get('source')!r}, so final_mesh is not "
            "a scaling of the TRELLIS GLB and no texture correspondence exists")
    return Path(a6d["mesh"]), Path(t2["glb"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", type=Path, help="Any6D final_mesh_<obj>.obj (tracked geometry)")
    ap.add_argument("--canonical", type=Path, help="TRELLIS.2 canonical GLB Any6D was given")
    ap.add_argument("--from-summary", type=Path, default=None,
                    help="run_scene summary.json; with --object, fills the two above")
    ap.add_argument("--object", default=None)
    ap.add_argument("--name", default=None, help="asset name; default --object / final stem")
    ap.add_argument("--out", type=Path, required=True,
                    help="asset root; the asset lands in <out>/<name>/")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL_M)
    ap.add_argument("--centre-tol", type=float, default=DEFAULT_CENTRE_TOL_M)
    ap.add_argument("--require-texture", action="store_true")
    ap.add_argument("--check", action="store_true", help="render <name>_check.png")
    ap.add_argument("--rotate-from", type=Path, default=None,
                    help="canonical_frame record (canonical.json): write the render-only "
                         "canonical copy final_mesh @ R_body_from_canon; no _fp.obj")
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    args = ap.parse_args(argv)

    if args.from_summary:
        if not args.object:
            ap.error("--from-summary needs --object")
        final, canonical = from_summary(args.from_summary, args.object)
    else:
        if args.final is None or args.canonical is None:
            ap.error("--final and --canonical (or --from-summary --object)")
        final, canonical = args.final, args.canonical
    name = args.name or args.object or final.stem.replace("final_mesh_", "")

    out = args.out / name
    rotate = rotation_from_json(args.rotate_from) if args.rotate_from else None
    rec = build_render_asset(final, canonical, out, name, tol_m=args.tol,
                             centre_tol_m=args.centre_tol, require_texture=args.require_texture,
                             rotate=rotate)
    print(f"[render-asset] {name}: {rec.n_vertices} v {rec.n_faces} f, "
          f"scale {np.round(rec.scale_xyz, 4).tolist()} residual {rec.fit_residual_m:.1e} m, "
          f"texture {'yes' if rec.texture else 'NO'} -> {rec.out_dir}")
    if rotate is not None:
        print(f"[render-asset] CANONICAL COPY (render-only, no _fp.obj): vertices = final_mesh @ R "
              f"from {args.rotate_from}; obj sha256 {rec.obj_sha256[:12]}")
    if rec.fp_obj:
        print(f"[render-asset] FoundationPose-facing textured OBJ -> {rec.fp_obj} "
              f"(+ .mtl, same PNG); register on this file for a textured scorer input")
    if args.check:
        from PIL import Image
        img = render_check(out, name)
        p = out / f"{name}_check.png"
        Image.fromarray(img).save(p)
        print(f"[render-asset] check render -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
