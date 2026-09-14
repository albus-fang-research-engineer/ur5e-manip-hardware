"""Offline tests for pose_server/rerank.py -- no GPU, no sidecar.

Fixture: a synthetic tapered mug (frustum + torus handle) in a tilted
canonical frame, a camera pitched ~40 deg, and a hypothesis set shaped like
FoundationPose's refined output on a near-symmetric object: the scorer's
top-K are jittered copies of a WRONG-yaw pose (handle behind the body), a
correct-yaw cluster sits lower in the ranking with a slightly lower score,
and a few tumbled poses are present. Silhouettes are rasterised from faces
with PIL, mask and depth are rendered from the true pose.

Covers the positive path (must select the correct cluster) and each decline
path (nothing unexplained; no hypothesis reaches U; tumbled body covering U
rejected by the precision floor; top-K disagreeing on the body), plus the
depth guard against a mask leak onto the table.

Run from the repo root:  python -m pytest test/test_rerank.py -v
Host deps: pip install numpy scipy pillow trimesh
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pose_server"))
trimesh = pytest.importorskip("trimesh")
pytest.importorskip("scipy")
PIL_Image = pytest.importorskip("PIL.Image")
from PIL import ImageDraw                                       # noqa: E402
from scipy.spatial.transform import Rotation                    # noqa: E402

from rerank import rerank_hypotheses, RerankParams              # noqa: E402

H, W = 360, 640
K = np.array([[450.0, 0, 320], [0, 450.0, 180], [0, 0, 1]])


def _mug():
    n, hs = 96, np.linspace(-0.05, 0.05, 24)
    rings = []
    for hh in hs:
        r = 0.035 + 0.015 * (hh + 0.05) / 0.10
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        rings.append(np.c_[r * np.cos(th), np.full(n, hh), r * np.sin(th)])
    V = np.vstack(rings); F = []
    for i in range(len(hs) - 1):
        for j in range(n):
            a, b, c, d = i * n + j, i * n + (j + 1) % n, (i + 1) * n + j, (i + 1) * n + (j + 1) % n
            F += [[a, b, c], [b, d, c]]
    V = np.vstack([V, [[0, -0.05, 0]]]); ci = len(V) - 1
    for j in range(n):
        F.append([ci, (j + 1) % n, j])
    body = trimesh.Trimesh(V, np.array(F), process=False).subdivide()
    handle = trimesh.creation.torus(major_radius=0.03, minor_radius=0.006, major_sections=40, minor_sections=10)
    handle.apply_translation([0.068, 0, 0])
    mug = trimesh.util.concatenate([body, handle])
    Rt = Rotation.from_euler("x", 38, degrees=True).as_matrix()       # tilted canonical frame
    mug.vertices = mug.vertices @ Rt.T
    shift = -mug.bounds.mean(0); mug.apply_translation(shift)
    axis = Rt @ np.array([0, 1.0, 0]); cb = shift.copy()               # body axis point (body was at origin)
    return mug, axis, cb, Rt


MUG, AXIS, CB, RT = _mug()
VV, FF = np.asarray(MUG.vertices), np.asarray(MUG.faces)
T_TRUE = np.eye(4); T_TRUE[:3, :3] = Rotation.from_euler("xy", [-40, 180], degrees=True).as_matrix() @ RT.T
T_TRUE[:3, 3] = [-0.05, 0.08, 0.5]
DIAMETER = float(np.linalg.norm(MUG.extents))


def _pose_yaw(deg, T=T_TRUE, jitter=None):
    """Rotate the object about its body axis (through cb) by deg, in the mesh frame,
    then apply T; optional small SE(3) jitter in the camera frame."""
    Ry = Rotation.from_rotvec(AXIS * np.radians(deg)).as_matrix()
    Tp = T.copy(); Tp[:3, :3] = T[:3, :3] @ Ry; Tp[:3, 3] = T[:3, 3] + T[:3, :3] @ (CB - Ry @ CB)
    if jitter is not None:
        # small rotation ABOUT THE OBJECT plus a small translation, in the camera
        # frame -- what refined hypotheses that agree on the body look like
        J = Rotation.from_euler("xyz", jitter[:3], degrees=True).as_matrix()
        Tp[:3, :3] = J @ Tp[:3, :3]
        Tp[:3, 3] = Tp[:3, 3] + np.asarray(jitter[3:])
    return Tp


def _tumbled(deg):
    Tp = T_TRUE.copy(); Tp[:3, :3] = Rotation.from_euler("x", deg, degrees=True).as_matrix() @ T_TRUE[:3, :3]
    return Tp


def _inverted(jitter_deg=0.0):
    """The mug upside down in place: 180 deg about the camera-horizontal axis
    through the object, then yawed so the handle lands where the real one is.
    Its silhouette is nearly the true one (frustum outline + handle), its
    rendered depth is not (flat base where the cavity is)."""
    T = T_TRUE.copy()
    T[:3, :3] = Rotation.from_euler("x", 180 + jitter_deg, degrees=True).as_matrix() @ T_TRUE[:3, :3]
    # bring the handle back to the true side: search the yaw that maximises
    # silhouette agreement with the true mask
    best, bestT = -1, T
    for yaw in range(0, 360, 10):
        Tp = _pose_yaw(yaw, T=T)
        iou = (_render(Tp)[0] & MASK).sum() / ((_render(Tp)[0] | MASK).sum())
        if iou > best:
            best, bestT = iou, Tp
    return bestT


from scipy.ndimage import distance_transform_edt                # noqa: E402


def _render(T):
    """(silhouette, dense rendered depth): faces rasterised with PIL; depth from
    the vertex z-buffer, nearest-filled inside the silhouette (a stand-in for
    the sidecar's nvdiffrast depth)."""
    P = VV @ T[:3, :3].T + T[:3, 3]; u = P @ K.T; u = u[:, :2] / u[:, 2:3]
    img = PIL_Image.new("1", (W, H), 0); d = ImageDraw.Draw(img)
    for row in np.clip(u[FF].reshape(-1, 6), -1e6, 1e6).tolist():
        d.polygon(row, fill=1)
    sil = np.array(img, bool)
    zb = np.full((H, W), np.inf); q = np.round(u).astype(int)
    ok = (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H)
    np.minimum.at(zb, (q[ok, 1], q[ok, 0]), P[ok, 2])
    valid = np.isfinite(zb)
    if not valid.any():
        return sil, np.zeros((H, W))
    idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    depth = np.where(sil, zb[idx[0], idx[1]], 0.0)
    return sil, depth


MASK, DEPTH_TRUE = _render(T_TRUE)
DEPTH = np.where(MASK, DEPTH_TRUE, 0.9)                     # table at 0.9 m where the object is not


def _hypothesis_set(include_correct=True, tumbled=True, filler=True, n_wrong=10, inverted=False, rng_seed=0):
    """scorer-sorted: n_wrong wrong-yaw jittered (handle behind), then (optionally)
    5 correct-yaw, then (optionally) 3 tumbled, then (optionally) filler at other
    yaws, then (optionally) 3 inverted copies. Returns (sils, scores, poses, depths)."""
    rng = np.random.default_rng(rng_seed)
    poses, scores = [], []
    for i in range(n_wrong):                                   # the flat top group, wrong yaw
        j = np.r_[rng.normal(0, 1.5, 3), rng.normal(0, 0.002, 3)]
        poses.append(_pose_yaw(140, jitter=j)); scores.append(71.4 - 0.01 * i)
    if include_correct:
        for i in range(5):
            j = np.r_[rng.normal(0, 1.0, 3), rng.normal(0, 0.002, 3)]
            poses.append(_pose_yaw(0, jitter=j)); scores.append(71.2 - 0.01 * i)
    if tumbled:
        for deg in (60, -60, 90):
            poses.append(_tumbled(deg)); scores.append(70.4)
    if filler:
        for yaw in (60, 90, 200, 250, 300):
            poses.append(_pose_yaw(yaw)); scores.append(70.0)
    if inverted:
        for jd in (0.0, 1.0, -1.0):
            poses.append(_inverted(jd)); scores.append(70.9)     # scores above the filler, below the correct cluster
    order = np.argsort(-np.asarray(scores))
    poses = np.asarray(poses)[order]; scores = np.asarray(scores)[order]
    rend = [_render(T) for T in poses]
    sils = np.stack([r[0] for r in rend]); depths = np.stack([r[1] for r in rend])
    return sils, scores, poses, depths


def _yaw_from_true(T):
    return np.degrees(np.linalg.norm(Rotation.from_matrix(T[:3, :3] @ T_TRUE[:3, :3].T).as_rotvec()))


# ------------------------------------------------------------------ positive

def test_selects_correct_yaw_cluster():
    sils, scores, poses, depths = _hypothesis_set()
    chosen, rec = rerank_hypotheses(sils, scores, poses, MASK, DEPTH, DIAMETER, depths=depths)
    assert rec.changed and rec.reason == "re-ranked"
    assert _yaw_from_true(poses[chosen]) < 8, f"chose a pose {_yaw_from_true(poses[chosen]):.0f} deg from true"
    assert _yaw_from_true(poses[0]) > 120                       # the scorer's pick was the wrong family
    assert rec.expl_to > 0.4 and rec.expl_from < 0.15
    assert rec.n_survivors >= 1 and rec.u_px > 0
    assert rec.scorer_to < rec.scorer_from                      # it overrode the scorer, deliberately


def test_deterministic():
    sils, scores, poses, depths = _hypothesis_set()
    a = rerank_hypotheses(sils, scores, poses, MASK, DEPTH, DIAMETER, depths=depths)[0]
    b = rerank_hypotheses(sils, scores, poses, MASK, DEPTH, DIAMETER, depths=depths)[0]
    assert a == b


# ------------------------------------------------------------------ declines

def test_declines_when_nothing_unexplained():
    """Mask WITHOUT the handle (rendered from the wrong-yaw pose itself):
    the top-K body explains everything -> scorer's pick stands, reason says so."""
    sils, scores, poses, depths = _hypothesis_set()
    mask_no_handle = _render(poses[0])[0]
    chosen, rec = rerank_hypotheses(sils, scores, poses, mask_no_handle, None, DIAMETER, depths=depths)
    assert chosen == 0 and not rec.changed
    assert "below floor" in rec.reason
    assert rec.u_frac < RerankParams().u_floor


def test_declines_when_no_hypothesis_reaches_U():
    """The mask has a region no refined hypothesis can reach (here: a part the
    mesh does not have, simulated as a blob at object depth beside the body):
    max expl ~0 -> decline; must not guess."""
    sils, scores, poses, depths = _hypothesis_set(include_correct=False, tumbled=False, filler=False, n_wrong=14)
    mask_body = _render(poses[0])[0]                       # the top-K explain this exactly
    ys, xs = np.nonzero(mask_body)
    blob = mask_body.copy(); x1 = int(xs.max()) + 12
    blob[int(ys.mean()) - 30:int(ys.mean()) + 30, x1:x1 + 40] = True
    depth = DEPTH.copy(); depth[blob & ~mask_body] = float(np.median(DEPTH[mask_body]))   # at object depth
    chosen, rec = rerank_hypotheses(sils, scores, poses, blob, depth, DIAMETER, depths=depths)
    assert chosen == 0 and not rec.changed
    assert rec.u_px > 0 and rec.max_expl < RerankParams().expl_floor
    assert "no hypothesis reaches" in rec.reason


def test_declines_when_only_wrong_family_present():
    """Handle in the mask, every hypothesis has it hidden: whichever decline
    fires (expl floor or precision floor), the pick must stand unchanged."""
    sils, scores, poses, depths = _hypothesis_set(include_correct=False, tumbled=False, filler=False, n_wrong=14)
    chosen, rec = rerank_hypotheses(sils, scores, poses, MASK, DEPTH, DIAMETER, depths=depths)
    assert chosen == 0 and not rec.changed and rec.reason != "re-ranked"


def test_tumbled_body_covering_U_is_rejected():
    """A tumbled hypothesis can lie across U with its body (expl ~1) but has
    poor precision; it must never be chosen over the correct cluster."""
    sils, scores, poses, depths = _hypothesis_set(tumbled=True)
    chosen, rec = rerank_hypotheses(sils, scores, poses, MASK, DEPTH, DIAMETER, depths=depths)
    tumb = [i for i, T in enumerate(poses) if abs(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3] @ T_TRUE[:3, :3].T) - 1) / 2, -1, 1)))) > 50
            and _yaw_from_true(T) > 50 and (sils[i] & MASK).sum() / max(sils[i].sum(), 1) < 0.9]
    assert chosen not in tumb
    assert _yaw_from_true(poses[chosen]) < 8


def test_declines_when_topK_disagree_on_body():
    """If the scorer's top-K are all over the place, U is most of the mask:
    decline with the hard-stop reason rather than pick anything."""
    sils, scores, poses, depths = _hypothesis_set()
    def scattered(i):
        T = _tumbled(float(np.linspace(-120, 120, 10)[i]))
        T[:3, 3] = T[:3, 3] + np.array([0.04 * ((i % 3) - 1), 0.03 * ((i % 2) - 0.5), 0.0])   # displaced bodies
        return T
    sc_poses = np.stack([scattered(i) for i in range(10)])
    tumb = np.stack([_render(T)[0] for T in sc_poses])
    sils2 = np.concatenate([tumb, sils]); scores2 = np.r_[np.full(10, 72.0), scores]
    poses2 = np.concatenate([sc_poses, poses])
    chosen, rec = rerank_hypotheses(sils2, scores2, poses2, MASK, DEPTH, DIAMETER)
    assert chosen == 0 and not rec.changed
    assert "disagree on the body" in rec.reason and rec.u_frac > RerankParams().u_frac_max


# ------------------------------------------------------------------ inverted body

def test_inverted_body_is_rejected_by_depth_gate():
    """An upside-down mug with the handle in the right place covers U like the
    correct pose and passes the precision floor; only the rendered-vs-measured
    depth (cavity vs flat base) tells them apart. With the correct cluster
    present it must be chosen over the inverted copies; without it the re-rank
    must DECLINE rather than pick an inverted body."""
    sils, scores, poses, depths = _hypothesis_set(inverted=True)
    # inverted copies score 70.9: identify them by score, not by rotation (a
    # filler at yaw 200 is also >150 deg from true without being inverted)
    inv = [i for i, sc in enumerate(scores) if abs(sc - 70.9) < 1e-6]
    assert len(inv) == 3
    chosen, rec = rerank_hypotheses(sils, scores, poses, MASK, DEPTH, DIAMETER, depths=depths)
    assert rec.changed and chosen not in inv and _yaw_from_true(poses[chosen]) < 8
    assert rec.depth_bad_to <= RerankParams().depth_bad_max

    sils2, scores2, poses2, depths2 = _hypothesis_set(include_correct=False, inverted=True)
    chosen2, rec2 = rerank_hypotheses(sils2, scores2, poses2, MASK, DEPTH, DIAMETER, depths=depths2)
    assert chosen2 == 0 and not rec2.changed
    assert "gates" in rec2.reason or "gated" in rec2.reason
    # and WITHOUT rendered depth the inverted copy would have been chosen -- the gate is load-bearing
    chosen3, rec3 = rerank_hypotheses(sils2, scores2, poses2, MASK, DEPTH, DIAMETER, depths=None)
    assert rec3.changed and abs(scores2[chosen3] - 70.9) < 1e-6


# ------------------------------------------------------------------ scorer margin / gate order

def test_scorer_margin_excludes_hypotheses_the_scorer_rejected():
    """Where the scorer is NOT flat -- the only handle-visible hypotheses sit
    well below the pick -- the re-rank must not override it."""
    sils, scores, poses, depths = _hypothesis_set(tumbled=False, filler=False)
    correct = np.array([_yaw_from_true(T) < 8 for T in poses])
    scores2 = scores.copy(); scores2[correct] = scores[0] - 2.5           # scorer clearly disfavours them
    order = np.argsort(-scores2)
    chosen, rec = rerank_hypotheses(sils[order], scores2[order], poses[order], MASK, DEPTH, DIAMETER,
                                    depths=depths[order])
    assert chosen == 0 and not rec.changed
    chosen2, rec2 = rerank_hypotheses(sils[order], scores2[order], poses[order], MASK, DEPTH, DIAMETER,
                                      depths=depths[order], params=RerankParams(scorer_margin=5.0))
    assert rec2.changed and _yaw_from_true(poses[order][chosen2]) < 8   # margin is the only thing holding it back


def test_relative_threshold_taken_after_gating():
    """A tumbled body covering U at expl ~1.0 is disqualified by precision; it
    must not set the relative expl threshold that the correct cluster is held to."""
    sils, scores, poses, depths = _hypothesis_set(tumbled=True)
    _, rec = rerank_hypotheses(sils, scores, poses, MASK, DEPTH, DIAMETER, depths=depths)
    assert rec.max_expl_gated <= rec.max_expl
    assert rec.n_gated >= rec.n_survivors >= 1


# ------------------------------------------------------------------ depth guard

def test_depth_guard_drops_table_leak():
    """SAM leaks a patch of table (0.9 m, object at 0.5 m) into the mask: the
    patch is dropped from U and the selection is unaffected."""
    sils, scores, poses, depths = _hypothesis_set()
    leak = MASK.copy()
    ys, xs = np.nonzero(MASK); y0, x0 = int(ys.min()) - 40, int(xs.mean())
    leak[max(0, y0 - 25):y0, x0 - 25:x0 + 25] = True           # a 25x50 patch above the object
    chosen_l, rec_l = rerank_hypotheses(sils, scores, poses, leak, DEPTH, DIAMETER, depths=depths)
    chosen, rec = rerank_hypotheses(sils, scores, poses, MASK, DEPTH, DIAMETER, depths=depths)
    assert rec_l.u_depth_dropped > 0
    assert chosen_l == chosen and rec_l.changed
    # without the guard the leak would inflate U and dilute expl
    _, rec_ng = rerank_hypotheses(sils, scores, poses, leak, None, DIAMETER, depths=depths)
    assert rec_ng.u_px > rec_l.u_px


# ------------------------------------------------------------------ scaling

def test_dilation_scales_with_mask_and_band_with_mesh():
    sils, scores, poses, depths = _hypothesis_set()
    _, rec = rerank_hypotheses(sils, scores, poses, MASK, DEPTH, DIAMETER, depths=depths)
    ys, xs = np.nonzero(MASK); diag = np.hypot(np.ptp(ys) + 1, np.ptp(xs) + 1)
    assert rec.dilate_px == max(1, int(round(0.01 * diag)))
    assert abs(rec.depth_band_m - 0.5 * DIAMETER) < 1e-9
    assert rec.u_px_raw >= rec.u_px
