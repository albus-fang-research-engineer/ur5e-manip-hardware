"""Kinematic MuJoCo replay of plan_stages_hw --method vlm plans (transport + pour).

Offline only. The UR5e (cuRobo's collision meshes, URDF imported, FK checked
against um.chain), the 2F-85 gripper mesh (tool0 frame), the teapot rigidly at
T_ee_canon, the mug at its registered pose and the table plane. Each stage is
timed as exec_plan times it (time_optimal(densify(Q)) at 10% speed) and replayed
by setting qpos. Clearances are exact-mesh approximations from dense surface
samples (MuJoCo's convex-hull contacts would fill the mug opening and the
handle), plus a vertical "stream" proxy from the spout tip into the opening.
Log only: nothing here gates a run; FLAGS name any clearance under 5 mm.

On the host (manip-tsr env; MUJOCO_GL=glfw -- egl and osmesa fail on this
machine; OMP_NUM_THREADS=4 or the KD-tree queries oversubscribe the CPU):

    SIM=~/ur5e-manip-sim-critic; R=outputs/runs/<stamp>
    MUJOCO_GL=glfw OMP_NUM_THREADS=4 SIM_DIR=$SIM \
    PYTHONPATH="$PWD/../manip-tsr:$SIM:$PWD/../manip-cbirrt:$PWD/ros2_ws/src/manip_bridge" \
    ~/miniconda3/envs/manip-tsr/bin/python scripts/mujoco_replay/replay_mujoco.py \
        --run $R --emissions $R/call3/emissions.json --selections $R/call2/selections.json \
        --plans $R/m4_vlm --out $R/m4_vlm/replay --video

Inputs: each plan dir's stages.json (plan_stages_hw --method vlm), the run's
canonical assets, and the 2F-85 mesh critic_scene.py export-gripper writes
to <run>/critic_assets/gripper_2f85.obj. meshes/ holds the UR5e collision
STLs cuRobo plans with (ur_description's, as in the CuroboServer image).
Outputs in --out: summary.json (per plan: stage times, the closest approach
per pair, stream frames inside the opening), contact_sheet.png, <plan>.mp4.
"""
import argparse, importlib.util, json, re, sys
from pathlib import Path
import numpy as np, trimesh, mujoco, cv2
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--emissions", required=True); ap.add_argument("--selections", required=True)
ap.add_argument("--plans", nargs="+", required=True, help="plan dirs holding stages.json")
ap.add_argument("--out", required=True)
ap.add_argument("--video", nargs="*", default=None, help="plan dir names to render (default all)")
ap.add_argument("--speed", type=float, default=0.1)
ap.add_argument("--check-every", type=int, default=2, help="check every k-th 20 ms sample")
ap.add_argument("--video-every", type=int, default=5, help="render every k-th sample (5 -> 10 fps)")
ap.add_argument("--gripper-mesh", default=None, help="default <run>/critic_assets/gripper_2f85.obj")
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HW = REPO / "outputs" / "runs"
FLAG_M = 0.005

spec_ = importlib.util.spec_from_file_location("psh", HW / "plan_stages_hw.py")
psh = importlib.util.module_from_spec(spec_); sys.modules["psh"] = psh; spec_.loader.exec_module(psh)
ns = argparse.Namespace(plan=None, selections=a.selections, emissions=a.emissions)
vin = psh.load_vlm_inputs(ns, Path(a.run), OUT / "_vin")
from manip_bridge import ur_motion as um, scene_ops as so
ch = um.chain()
sym, poses = vin.symbols, vin.poses

# ------------------------------------------------------------- objects (as inspect_plan.py)
def P(obj, kind, name): return np.asarray(getattr(sym[obj], kind)[name], float)
tp_spout, tp_body = P("teapot", "points", "spout_center"), P("teapot", "points", "body_center")
T_mug = np.asarray(poses["mug"], float)
mug_open = T_mug[:3, :3] @ P("mug", "points", "opening_center") + T_mug[:3, 3]
mug_up = T_mug[:3, :3] @ P("mug", "axes", "up_axis")
rim = float(sym["mug"].quantities["rim_radius"])
Rb = {o: np.asarray(getattr(sym[o], "R_body_from_canon", np.eye(3)), float) for o in ("teapot", "mug")}
def load_canon(o):
    m = trimesh.load(str(vin.asset_dirs[o] / "meshes" / f"{o}_visual.obj"), force="mesh")
    return trimesh.Trimesh(np.asarray(m.vertices) @ Rb[o], m.faces, process=False)
tp_c, mug_c = load_canon("teapot"), load_canon("mug")
mug_w = trimesh.Trimesh(np.asarray(mug_c.vertices) @ T_mug[:3, :3].T + T_mug[:3, 3], mug_c.faces, process=False)
grip_path = Path(a.gripper_mesh) if a.gripper_mesh else Path(a.run) / "critic_assets" / "gripper_2f85.obj"
if not grip_path.exists():
    sys.exit(f"no gripper mesh at {grip_path}: run critic_scene.py export-gripper --run {a.run} first")
grip = trimesh.load(str(grip_path), force="mesh")
n_pl, d_pl = so.table_plane_base(vin.summary)

def xf(T, V): return np.asarray(V) @ T[:3, :3].T + T[:3, 3]
def samples(mesh, k): return np.asarray(trimesh.sample.sample_surface_even(mesh, k)[0]) if k else np.asarray(mesh.vertices)

# ------------------------------------------------------------- MuJoCo model
urdf = OUT / "_ur5e_mj.urdf"        # cuRobo's UR5e URDF, collision meshes only, for MuJoCo
u = (REPO / "curobo_runtime" / "robot" / "ur5e.urdf").read_text()
u = re.sub(r"<visual>.*?</visual>", "", u, flags=re.S)
u = re.sub(r"<transmission.*?</transmission>", "", u, flags=re.S)
u = u.replace("meshes/ur5e/collision/", "")
u = u.replace("</robot>", f'<mujoco><compiler meshdir="{HERE}/meshes" fusestatic="false" discardvisual="false" '
              f'strippath="true" balanceinertia="true"/></mujoco></robot>')
urdf.write_text(u)
spec = mujoco.MjSpec.from_file(str(urdf))
spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 960
for g in spec.geoms:
    g.contype = g.conaffinity = 0; g.rgba = [0.72, 0.74, 0.78, 1]
def add_mesh(name, V, F):
    m = spec.add_mesh(); m.name = name
    m.uservert = np.asarray(V, np.float32).ravel().tolist(); m.userface = np.asarray(F, np.int32).ravel().tolist()
    m.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
def add_geom(body, mesh, rgba):
    g = body.add_geom(); g.type = mujoco.mjtGeom.mjGEOM_MESH; g.meshname = mesh
    g.contype = g.conaffinity = 0; g.rgba = rgba; g.mass = 1e-4
    return g
tool0 = spec.body("tool0")
add_mesh("gripper", grip.vertices, grip.faces); add_geom(tool0, "gripper", [0.15, 0.15, 0.17, 1])
add_mesh("mug", mug_w.vertices, mug_w.faces); add_geom(spec.worldbody, "mug", [0.2, 0.45, 0.85, 1])
# teapot vertices go in per plan (T_ee_canon may differ per plan dir): placeholder body, mesh set below
tp_body_spec = tool0.add_body(); tp_body_spec.name = "teapot"
add_mesh("teapot", tp_c.vertices, tp_c.faces); add_geom(tp_body_spec, "teapot", [0.95, 0.55, 0.15, 1])
# table: box under the plane, around the workspace
z = n_pl / np.linalg.norm(n_pl); xa = np.cross([0, 1, 0], z); xa /= np.linalg.norm(xa)
Rt = np.column_stack([xa, np.cross(z, xa), z]); c_xy = 0.5 * (mug_open[:2] + np.array([0.4, 0.2]))
p0 = np.array([c_xy[0], c_xy[1], float(so.table_z((n_pl, d_pl), *c_xy))])
tb = spec.worldbody.add_geom(); tb.type = mujoco.mjtGeom.mjGEOM_BOX; tb.size = [0.6, 0.6, 0.01]
tb.pos = p0 - 0.01 * z; q = np.zeros(4); mujoco.mju_mat2Quat(q, Rt.ravel()); tb.quat = q
tb.contype = tb.conaffinity = 0; tb.rgba = [0.78, 0.68, 0.55, 1]
# stream marker: a thin vertical capsule from the spout tip, moved per frame (mocap)
mk = spec.worldbody.add_body(); mk.name = "stream"; mk.mocap = True
sg = mk.add_geom(); sg.name = "stream_g"; sg.type = mujoco.mjtGeom.mjGEOM_CYLINDER; sg.size = [0.002, 0.05, 0]
sg.contype = sg.conaffinity = 0; sg.rgba = [0.1, 0.8, 0.9, 0.0]
L = spec.worldbody.add_light(); L.pos = [0.4, 0, 1.5]; L.dir = [0, 0, -1]; L.diffuse = [0.7, 0.7, 0.7]
model = spec.compile(); data = mujoco.MjData(model)
qadr = [model.joint(n).qposadr[0] for n in um.DH_ORDER]
tp_bid = model.body("teapot").id; SG = model.geom("stream_g").id

# verify FK once more on the compiled scene
rng = np.random.default_rng(1); fk_err = 0.0
for _ in range(10):
    qq = rng.uniform(-np.pi, np.pi, 6); data.qpos[qadr] = qq; mujoco.mj_forward(model, data)
    fk_err = max(fk_err, float(np.linalg.norm(data.body("tool0").xpos - ch.fk(qq)[:3, 3])))
print(f"FK check (scene): max tool0 position error {fk_err:.2e} m")

# arm link surface samples in geom frames (MuJoCo's recentered mesh data)
ARM = {}
for gi in range(model.ngeom):
    gm = model.geom(gi)
    if gm.type[0] != mujoco.mjtGeom.mjGEOM_MESH: continue
    bname = model.body(gm.bodyid[0]).name
    if bname in ("tool0", "teapot", "world"): continue
    mid = gm.dataid[0]
    V = model.mesh_vert[model.mesh_vertadr[mid]: model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
    F = model.mesh_face[model.mesh_faceadr[mid]: model.mesh_faceadr[mid] + model.mesh_facenum[mid]]
    ARM[gi] = (bname, samples(trimesh.Trimesh(V, F, process=False), 4000))
print("arm geoms:", sorted({b for b, _ in ARM.values()}))
NO_TABLE = {"base_link_inertia", "shoulder_link", "base_link"}    # mounted at the table
NO_TEAPOT = {"wrist_3_link"}                                     # holds the gripper
S_mug = samples(mug_w, 30000); mug_tree = cKDTree(S_mug)
S_tp_c = samples(tp_c, 25000); S_grip = samples(grip, 12000)
def plane_min(W): return float((W @ n_pl + d_pl).min() / np.linalg.norm(n_pl))

# ------------------------------------------------------------- render setup
R_W, R_H = 640, 480
renderer = mujoco.Renderer(model, R_H, R_W)
look = 0.5 * (mug_open + np.array([0.5, 0.0, mug_open[2]]))
cams = []
for az, el, dist in ((150, -25, 1.25), (60, -15, 0.75)):
    c = mujoco.MjvCamera(); c.lookat[:] = look if dist > 1 else mug_open + [0, 0, 0.08]
    c.azimuth, c.elevation, c.distance = az, el, dist; cams.append(c)
def render():
    ims = []
    for c in cams:
        renderer.update_scene(data, camera=c); ims.append(renderer.render())
    return np.concatenate(ims, axis=1)

# ------------------------------------------------------------- replay
summary, sheet = [], []
for pdir in map(Path, a.plans):
    st = json.loads((pdir / "stages.json").read_text())
    T_ee = np.asarray(st["T_ee_canon"], float)
    # teapot geom frame = T_ee in tool0 (canonical vertices)
    model.body_pos[tp_bid] = T_ee[:3, 3]; qb = np.zeros(4); mujoco.mju_mat2Quat(qb, T_ee[:3, :3].ravel())
    model.body_quat[tp_bid] = qb
    S_tp_t0 = xf(T_ee, S_tp_c); tp_tree_t0 = cKDTree(S_tp_t0)
    spout_t0, body_t0 = T_ee[:3, :3] @ tp_spout + T_ee[:3, 3], T_ee[:3, :3] @ tp_body + T_ee[:3, 3]
    mins = {}; stream = {"frames": 0, "in": 0, "worst_r_cm": 0.0, "min_h_cm": 1e9}
    def upd(key, v, where):
        if key not in mins or v < mins[key][0]: mins[key] = (v, where)
    do_video = a.video is None or pdir.name in a.video
    vw = None; keys = []
    for sname in ("transport", "pour"):
        S = next(s for s in st["stages"] if s["name"] == sname)
        t, Qs, _ = um.time_optimal(um.densify(np.asarray(S["positions"], float)),
                                   um.UR5E_VMAX * a.speed, um.UR5E_AMAX * a.speed)
        n = len(Qs); import time as _t; _t0 = _t.time()
        print(f"  {pdir.name} {sname}: {n} samples, {t[-1]:.1f} s", flush=True)
        for i, qq in enumerate(Qs):
            if i % 100 == 0: print(f"    {i}/{n} {_t.time() - _t0:.1f}s", flush=True)
            last = i == n - 1
            chk, vid = (i % a.check_every == 0) or last, do_video and ((i % a.video_every == 0) or last)
            if not (chk or vid or i in (0, n // 2)): continue
            data.qpos[qadr] = qq; mujoco.mj_forward(model, data)
            T0 = np.eye(4); T0[:3, :3] = data.body("tool0").xmat.reshape(3, 3); T0[:3, 3] = data.body("tool0").xpos
            where = f"{sname} {100 * i / max(n - 1, 1):.0f}%"
            sp, bd = xf(T0, spout_t0), xf(T0, body_t0)
            pouring = sname == "pour" and sp[2] < bd[2]
            if chk:
                W_tp = xf(T0, S_tp_t0); W_gr = xf(T0, S_grip)
                upd("teapot-mug", float(mug_tree.query(W_tp)[0].min()), where)
                upd("teapot-table", plane_min(W_tp), where)
                upd("gripper-mug", float(mug_tree.query(W_gr)[0].min()), where)
                upd("gripper-table", plane_min(W_gr), where)
                Tinv = np.linalg.inv(T0)
                for gi, (bname, Sg) in ARM.items():
                    Wg = Sg @ data.geom_xmat[gi].reshape(3, 3).T + data.geom_xpos[gi]
                    upd(f"arm-mug ({bname})", float(mug_tree.query(Wg)[0].min()), where)
                    if bname not in NO_TABLE: upd(f"arm-table ({bname})", plane_min(Wg), where)
                    if bname not in NO_TEAPOT:
                        upd(f"teapot-arm ({bname})", float(tp_tree_t0.query(xf(Tinv, Wg))[0].min()), where)
                if pouring:
                    d_ = sp - mug_open; h = d_ @ mug_up
                    # vertical line from the spout down to the opening plane
                    tt = h / max(mug_up @ np.array([0, 0, 1.0]), 1e-6)
                    hit = sp - tt * np.array([0, 0, 1.0]); r = np.linalg.norm((hit - mug_open) - ((hit - mug_open) @ mug_up) * mug_up)
                    stream["frames"] += 1; stream["in"] += int(r < rim)
                    stream["worst_r_cm"] = max(stream["worst_r_cm"], 100 * r); stream["min_h_cm"] = min(stream["min_h_cm"], 100 * h)
            if vid or i in (0, n // 2) or last:
                if pouring:
                    data.mocap_pos[0] = sp - np.array([0, 0, 0.05]); model.geom_rgba[SG][3] = 0.9
                else:
                    model.geom_rgba[SG][3] = 0.0
                mujoco.mj_forward(model, data)
                img = render()
                cv2.putText(img, f"{pdir.name}  {where}  t={t[i]:.1f}s", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if vid:
                    if vw is None:
                        vw = cv2.VideoWriter(str(OUT / f"{pdir.name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                             10, (img.shape[1], img.shape[0]))
                    vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                if i in (0, n // 2) or last:
                    keys.append(cv2.resize(img[:, :R_W], (320, 240)))
        if sname == "transport": t_tr = float(t[-1])
        else: t_pour = float(t[-1])
    if vw is not None: vw.release()
    keys = [keys[k] for k in (0, 2, 4, 5)] if len(keys) >= 6 else keys
    sheet.append(np.concatenate(keys, axis=1))
    rec = {"plan": pdir.name, "transport_s": round(t_tr, 1), "pour_s": round(t_pour, 1),
           "min_clearance_m": {k: [round(v, 4), w] for k, (v, w) in sorted(mins.items(), key=lambda kv: kv[1][0])},
           "stream": stream,
           "flags": [f"{k} {100 * v:.1f} cm at {w}" for k, (v, w) in mins.items() if v < FLAG_M]}
    summary.append(rec)
    worst = list(rec["min_clearance_m"].items())[:3]
    print(f"{pdir.name}: tr {t_tr:.1f}s pour {t_pour:.1f}s | "
          + "; ".join(f"{k} {100 * v[0]:.1f}cm@{v[1]}" for k, v in worst)
          + f" | stream in {stream['in']}/{stream['frames']} (worst r {stream['worst_r_cm']:.1f} cm, min h {stream['min_h_cm']:.1f} cm)"
          + (f" | FLAGS {rec['flags']}" if rec["flags"] else ""))

json.dump({"fk_check_max_pos_err_m": fk_err, "flag_threshold_m": FLAG_M, "rim_radius_m": rim,
           "plans": summary}, open(OUT / "summary.json", "w"), indent=1)
cv2.imwrite(str(OUT / "contact_sheet.png"), cv2.cvtColor(np.concatenate(sheet, axis=0), cv2.COLOR_RGB2BGR))
print("->", OUT)
