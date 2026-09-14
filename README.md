# Perception wing: FoundationPose + PointSO + ROS2 bridge

Fixed ports: grasp `5666`, pose `5667`, pointso `5668`, trellis2 `5669`,
sam3 `5670`, curobo `5671`, any6d `5672`, oriany `5673`. Everything is
host-networked *except* `grasp`, which is bridged with a pinned MAC so the
AnyGrasp license fingerprint can't drift — see
[AnyGrasp sidecar](#anygrasp-sidecar-port-5666).

```
ur5e-manip-sim/
├── docker-compose.yml                  # existing (unchanged)
├── docker-compose.perception.yml       # NEW overlay
├── docker/
│   ├── Dockerfile.foundationpose       # NEW
│   ├── Dockerfile.pointso              # NEW
│   └── Dockerfile.ros2                 # NEW
├── pose_server/server.py               # NEW  (mounted into pose)
├── pointso_server/server.py            # NEW  (mounted into pointso)
├── ros2_bridge/
│   ├── pose_bridge_node.py             # NEW  (mounted into ros2-bridge)
│   └── pointso_bridge_node.py          # NEW
├── foundationpose_runtime/
│   ├── weights/                        # refiner 2023-10-28-18-33-37,
│   │                                   # scorer  2024-01-11-20-02-45
│   └── meshes/                         # object CADs (.obj)
└── pointso_runtime/
    └── checkpoints/                    # small.pth / base_finetune.pth
```

## One-time setup

```bash
# PointSO checkpoint (small; use base_finetune.pth for Open6DOR-style tasks)
mkdir -p pointso_runtime/checkpoints
wget -c https://huggingface.co/qizekun/PointSO/resolve/main/small.pth \
     -P pointso_runtime/checkpoints/

# FoundationPose weights: from the repo's Google Drive link (README), place
# both run folders under foundationpose_runtime/weights/
mkdir -p foundationpose_runtime/weights foundationpose_runtime/meshes
```

## Build & run

```bash
docker compose -f docker-compose.yml -f docker-compose.perception.yml \
    --profile perception build pose pointso

docker compose -f docker-compose.yml -f docker-compose.perception.yml \
    --profile perception up -d pose pointso

docker compose -f docker-compose.yml -f docker-compose.perception.yml \
    --profile ros2 up -d ros2-bridge
```

Smoke test from the host or `sim`:

```python
import zmq, msgpack, msgpack_numpy; msgpack_numpy.patch()
s = zmq.Context().socket(zmq.REQ); s.connect("tcp://127.0.0.1:5668")
s.send(msgpack.packb({"cmd": "ping"})); print(msgpack.unpackb(s.recv()))
```

## Notes / sharp edges


- **FoundationPose build drift**: `build_all.sh` occasionally breaks against
  upstream kaolin/repo changes. If the image build fails there, set
  `--build-arg FP_COMMIT=<known-good SHA>`.
- **SoFar deps**: the Dockerfile does a full `pip install -e .` (pulls the
  whole SoFar dep tree, including VLM-side extras you don't need for PointSO
  alone). If it fights the base image's torch, retry with `--no-deps` and
  install `easydict pyyaml timm` manually.
- **ROS2 bridge**: host network + `ipc: host` so DDS finds the robot and any
  other Humble machines. Set `ROS_DOMAIN_ID` in your shell env to match.
  The bridge nodes are ZMQ clients only — no CUDA, no model code — so the
  perception environments never see ROS's Python 3.10 pin.

## ROS2 bridge layer (`ros2_ws/`)

Two kinds of node, matching how the sidecars are used in a task:

| kind | node | interface | sidecar |
|---|---|---|---|
| service (once per scene) | `sam3_bridge` | `/sam3/segment` `Segment` | sam3 :5670 |
| service (once per scene) | `trellis2_bridge` | `/trellis2/generate_mesh` `GenerateMesh` | trellis2 :5669 |
| service (once per scene) | `oriany_bridge` | `/oriany/orient` `Orient` | oriany :5673 |
| service + stream | `any6d_bridge` | `/any6d/estimate` `/any6d/release` → `/any6d/<obj>/pose` + TF | any6d :5672 |
| service + stream | `pose_bridge` | `/pose/estimate` `/pose/release` → `/pose/<obj>/pose` + TF | pose :5667 |
| service (per stage) | `constrained_planner` | `/planner/plan_constrained` `PlanConstrained` → `/planner/trajectory` `/planner/{ee,body}_path` | curobo :5671 (`plan_constrained` cmd) |
| topic filter (per object) | `grasp_filter` | `/grasp/<obj>/grasps` × `/tsr/<obj>/grasp` → `/grasp/<obj>/filtered` + report + markers | none (pure `manip_tsr`) |
| latched publisher | `tsr_from_yaml` | YAML → `/tsr/<obj>/grasp` (hand arm / test harness) | none |
| legacy topic-JSON | `ros2_bridge/pointso_bridge_node.py` | `/pointso_bridge/*` | pointso :5668 |

`EstimatePose` registers once from a caller-supplied RGB-D + mask; on success
the node starts tracking that object on every synced camera frame until
`Release`. The two trackers share `manip_bridge/tracker_bridge.py` (two
callback groups, two ZMQ sockets, drop-if-busy, frames skipped while an
estimate is in flight) so FoundationPose and Any6D have identical interfaces
and can be compared head-to-head.

`oriany_bridge` is a service and not a tracker on purpose: the sidecar holds
no per-object state, and semantic orientation is a scene-time label. Once a
body frame is registered its orientation is carried per frame by the pose
tracker, so nothing is gained by re-asking which way the front is at 30 Hz.
It publishes no TF — the model returns a rotation, and a frame needs an
origin; compose the `orientation` quaternion (`R_cam`, see
[Orient Anything V2](#orient-anything-v2-sidecar-port-5673)) with a pose
sidecar translation client-side.

`Orient` reuses `GenerateMesh`'s mask convention: an **empty (0x0)** mask
means the model mattes the full frame itself (rembg, upstream `app.py`); a
**non-empty** mask means the bridge crops to the mask bbox, fills the
background, and square-pads to `fg_ratio` (0.85) before sending with
`remove_bkg=false`. That padding is not cosmetic — upstream only runs
`resize_foreground` inside the rembg path, so a tight bbox crop is a framing
the model never trained on. `run_scene --oriany-matting` switches to the
matting path so the two are directly comparable. `bg_fill` defaults to 255,
which is what upstream `preprocess_images` composites the rembg RGBA onto.

`mesh` in `EstimatePose` is a filename under the sidecar's `/opt/meshes` **or**
an absolute path. Mounts: `./trellis2_runtime/outputs` at `/data/meshes`
(pose, any6d) and `./any6d_runtime/outputs` at `/data/any6d` (pose,
ros2-bridge).

**Perception flow (what `run_scene` does by default).** TRELLIS.2 produces
the canonical unit-box GLB and replaces InstantMesh as Any6D's input mesh;
Any6D does the metric scaling itself and exports `final_mesh_<obj>.obj`;
FoundationPose registers and tracks on that file and is the tracker of
record (Any6D's own `track` is the same refiner on the same mesh -- the
second registration buys process isolation, a `release`-able Any6D
session, and a same-mesh consistency check that `run_scene` prints). The
TRELLIS sidecar's `_metric.glb` / `metric_scale.py` branch is recorded in
`summary.json` when computed but nothing consumes it.

Two facts about `final_mesh` that the rest of the stack relies on
(`taeyeopl/Any6D` at `80eb486`): the returned `cam_T_obj` is in
`final_mesh`'s frame -- the last `reset_object` is on the already-centred
scaled mesh, so the centring compensation is the identity and the export is
that mesh (its bbox centre is 0; assert it, it is numerical not structural);
and Any6D only ever replaces `mesh.vertices`, so vertex order and faces are
identical to the input (`final = diag(s)·(canonical − c)`, measured residual
1e-8 m on the mug run). A textured GLB into Any6D used to crash in
`reset_object` (`material.image` on a `PBRMaterial`); `any6d_server` now
swaps in a `SimpleMaterial`, verified to carry an `EXT_texture_webp` texture
under trimesh 4.2.2.

### Build

```bash
docker compose up -d --build ros2-bridge
docker exec -it Ros2Bridge bash
colcon build --symlink-install && exit
docker compose restart ros2-bridge       # launches bridges.launch.py
```

### Online test from a rosbag

Put the recording under `./bags` (or set `BAG_DIR` in `.env`); it is mounted
at `/bags` in `Ros2Bridge`. Everything runs inside that container so
`ROS_DOMAIN_ID` and DDS config are automatically consistent.

```bash
docker compose up -d sam3 trellis2 any6d pose        # sidecars you want
docker exec -it Ros2Bridge bash

ros2 bag info /bags/<name>        # read the ACTUAL topic names + depth encoding
ros2 bag play /bags/<name> --clock --loop &

ros2 launch manip_bridge bridges.launch.py use_sim_time:=true \
    rgb_topic:=/camera/camera/color/image_raw \
    depth_topic:=/camera/camera/aligned_depth_to_color/image_raw \
    info_topic:=/camera/camera/color/camera_info
```

Then, in a second shell in the container:

```bash
ros2 run manip_bridge run_scene --ros-args -p use_sim_time:=true -- \
    --prompts teapot mug --bg "robot arm" --watch 30
```

`run_scene` grabs one synced frame, segments, runs TRELLIS.2 (canonical +
metric), Any6D (`img_to_3d`, or `--any6d-mesh trellis`), FoundationPose on
the TRELLIS metric mesh, prints a per-object pose table and then reports
tracking rate / position std / rotation drift for the streaming nodes.
Artifacts (rgb, depth, masks, overlay, `summary.json`) land in
`./outputs/runs/<stamp>/`; GLBs in `./trellis2_runtime/outputs`, scaled Any6D
meshes in `./any6d_runtime/outputs`. `--skip trellis2,any6d` etc. for a
partial stack.

Things that silently yield *no callbacks* on replay:

- `--clock` without `use_sim_time:=true` (or vice versa): stamps are
  bag-epoch, sync/TF compare against wall-clock and drop everything.
- QoS: RealSense topics are best-effort; the bridges subscribe with
  `qos_profile_sensor_data`. If the bag's `metadata.yaml` recorded reliable
  QoS that is still compatible, but a reliable subscriber on a best-effort
  publisher is not.
- Topic names: the realsense-ros default is `/camera/camera/...` (node name
  repeated). Older bags may be `/camera/...`. Pass the launch args.
- No aligned depth in the bag → the pipeline needs depth→color reprojection
  first; the bridges assume aligned RGB-D with one `K`.

For the hand-eye transform on hardware, add a `static_transform_publisher`
(from the easy_handeye2 YAML) to the launch file; `rviz2` with
`use_sim_time` on the host then shows `pose_<obj>` / `any6d_<obj>` frames
relative to `base_link`.

## Render asset from the tracked mesh (`render_asset`)

The sim repo's grounding renderers (`ground_parts.py`, `render_candidates.py`,
`render_stage_frames.py`) read an object asset dir -- `<name>.xml` with a
body `object` carrying the visual geom, plus `meshes/<name>_visual.obj` --
and every `frames.json` symbol is expressed in that mesh's body frame. On
hardware the body frame of record is **Any6D's `final_mesh_<obj>.obj`**:
FoundationPose registers and tracks on it and Any6D's own pose is in it (see
the perception-flow paragraph above). `manip_bridge/render_asset.py` builds
the asset dir as that file's geometry under the identity transform:

```bash
ros2 run manip_bridge render_asset -- \
    --from-summary /data/runs/<stamp>/summary.json --object mug \
    --out /data/runs/<stamp>/assets --check
```

Do **not** use `ur5e-manip-sim/scripts/convert_asset.py` for this. It
recentres at the bbox centroid unconditionally, drops the texture on OBJ
export, and runs CoACD (nothing on hardware consumes hulls; attached-object
collision is cuRobo spheres). A recentred asset offsets every symbol from
the tracked frame silently -- it looks like a bad VLM pick or tracker drift.
And `final_mesh` is not `canonical GLB x one scale` either (Any6D bbox-centres
and scales each axis separately), so nothing is rebuilt from a scale factor.

How the build stays honest:

- **Geometry is `final_mesh`, verbatim.** Vertices and faces are parsed from
  the OBJ text with the sim's own `load_obj` rule (`v` lines, first face
  index) -- no library reprocessing in between.
- **Texture comes from the canonical GLB by vertex index.** Any6D only ever
  replaces `mesh.vertices`, so order and faces are the input's. The builder
  loads the GLB exactly as `any6d_server` does, then **refuses** unless the
  correspondence is proven: equal vertex counts, identical face arrays, and
  `final == diag(s)·(canonical − c)` with residual ≤ `--tol` (1 µm) and
  `s > 0` -- a scrambled order, a rotated copy, or a different TRELLIS run
  all fail this. `s` and `c` are recorded in `render_asset.json`.
- **`final_mesh` must be bbox-centred** (`--centre-tol`, 0.1 mm). That is the
  condition under which Any6D's centring compensation is the identity and
  the pose it returns -- and FoundationPose's, registered on the same file
  -- is in this file's frame.
- MuJoCo ignores MTL. The OBJ carries `vt` per vertex, and the MJCF declares
  `<texture type="2d">` + `<material>` on the geom explicitly. The texture
  path is **absolute** (the sim's `build_model` loads via `from_xml_string`
  with an injected `meshdir` and no `texturedir`), so an asset dir is bound
  to the path it was built at -- rebuild, don't move.
- `--check` renders one view to `<name>_check.png`. A flat grey object means
  the texture is not wired and SAM3 on the canonical renders will see the
  same grey; fix that before grounding.

`--from-summary` refuses an object whose Any6D ran with `source: img_to_3d`
(InstantMesh geometry has no correspondence to the TRELLIS GLB).

The builder also writes `meshes/<name>_fp.obj` (+ `.mtl`, same PNG): the same
geometry with `mtllib`, so trimesh -- and therefore the pose sidecar -- loads
it as a textured mesh. FoundationPose's scorer takes RGB + XYZ; on a
yaw-symmetric body (a mug) XYZ cannot break the yaw tie and the texture is the
only cue it has, so register on `<name>_fp.obj` rather than on the untextured
`final_mesh_<obj>.obj`. `./outputs` is mounted read-only at `/data/runs` in
the pose container for this. The `register` reply reports `texture:` so a
gray scorer input is visible in the log.

**Yaw ambiguity diagnosis (2026-09-13, saved run `20260903_203531`).** Both
Any6D and standalone FoundationPose registered the mug ~140° wrong in yaw
(handle behind the body); the sidecar's 252 scores spanned two points with the
top thirty within 0.4. `outputs/runs/reproject_check.py` (symmetry-axis
search, yaw sweep against the SAM mask) located the error; `--pre-rotate body
140` put the handle on the mask with the body in place. `fp_from_mesh.py
--all` returns every refined hypothesis (`return_all`); `rank_hypotheses.py`
scores each against the mask. Findings: a correct refined hypothesis EXISTS
(ranks 17–30, 155° from the pick) but no global silhouette statistic separates
it from the wrong family -- the undersized handle explains only ~26% of the
mask's handle region, the same size as body-fit jitter.

**Fix: mask-conditioned re-rank (`pose_server/rerank.py`, `"rerank": true` on
`register`).** The scorer's top-10 agree on the body and disagree only about
the part, so their consensus silhouette (dilated by 1% of the mask's bbox
diagonal) IS the body, with no part detector. `U = mask ∖ consensus`, minus
pixels farther than half the mesh diameter from the body's depth (SAM leak
guard), is what the body cannot explain. Per hypothesis `expl = |sil ∩ U| /
|U|`; survivors have `expl ≥ 0.6·max`, silhouette precision ≥ 0.9 (a
tumbled body can cover `U` and score 1.0 -- precision catches it), at most
10% of their silhouette pixels with rendered depth more than 15% of the mesh
diameter off the measured depth (an **inverted** cup has the same outline and
puts its handle in the same place, but a flat base where the visible cavity
is: measured 0.02 upright vs 0.44 inverted), and a scorer score within 1.0 of
the pick's (the scorer is overridden only where it is flat: correct family
0.25 below the pick, inverted family 1.4 below). The relative `expl`
threshold is taken over the gated set, so a disqualified tumbled body at
`expl` 1.0 cannot set it. The scorer picks among survivors. Declines with a recorded reason when `|U| < 2%` of the
mask (nothing unexplained / symmetric object), when `max(expl) < 0.10` (no
hypothesis reaches `U`: part missing from the mesh or set collapsed), or when
`u_frac > 0.35` (the top-10 do not agree on the body -- **treat as a failed
registration upstream**). Off by default; the reply carries the `rerank`
record. Validated offline on the 252 saved hypotheses: `U` = 570 px at half
resolution vs a 569 px geometric handle blob, picks rank 18 (155°, handle on
the mask, scorer 71.17 vs 71.42) -- `rank_hypotheses.py --rerank` runs the
identical code on a saved `.npz`; `test/test_rerank.py` covers the positive
path, each decline, the depth guard and the scaling of the constants on a
synthetic tapered mug. Any6D's own registration is not re-ranked (its scale
loop is interleaved with scoring); FoundationPose on `final_mesh` is the
tracker of record, so this is where the fix reaches the pipeline.

Tests (offline, no sidecars): `python -m pytest test/test_render_asset.py -v`
-- the fixture replays Any6D's chain on a textured off-centre GLB (any6d-style
load, bbox-centre, per-axis scale, trimesh OBJ export); checks vertex
identity, recovered `s`/`c`, refusal of wrong reconstruction / scrambled
order / face mismatch / uncentred / flipped, texture wiring, and a MuJoCo
render that must show both checker colours.

## Orient Anything V2 sidecar (port 5673)

Category-free canonical **up / front** from one RGB crop. It is the
semantic init for `refine_frame.py` (the geometry refinement owns the
sub-degree regime); its `alpha` head is why V2 is here: `alpha != 1` means
the front axis is only defined up to a symmetry group and the corresponding
TSR rotational bound should widen or drop.

### Frame convention (read before trusting an overlay)

The model emits `(azimuth, elevation, rotation)` in 1° bins. Upstream never
states a handedness in text; the only place it commits to one is the demo
overlay, `utils/axis_renderer.py`, which sets a Blender gizmo to
`R_blender = Rx(rot) Ry(ele) Rz(-azi)` and renders it from the camera in
`assets/axis_render.blend` (world `(100, 0, 0)`, Euler XYZ `(90°, 0, 90°)`:
looking `-X`, `Z` up, so image-right = `+Y_w`, image-up = `+Z_w`). Gizmo
canonical is `x = front`, `y = lateral`, `z = up`, with front facing that
camera at `az = 0`.

The sidecar therefore serves

```
R_cam = BLENDER_TO_CAM @ R_blender,   BLENDER_TO_CAM = [[0, 1, 0], [0, 0, -1], [-1, 0, 0]]
up_cam / front_cam / lateral_cam = R_cam[:, 2] / R_cam[:, 0] / R_cam[:, 1]
```

in the OpenCV frame of the input image (x right, y down, z forward). That
is a fixed change of basis, not a fitted sign; there is no `AZ_SIGN` knob.
(The first cut of the server hand-rolled these vectors and had the image-x
sign of both `az` and `ro` mirrored w.r.t. the gizmo, i.e. front drawn on
the wrong side of the object whenever `az != 0`. If you have overlays from
before that fix, mirror front/lateral about the image's vertical axis to
read them.)

What "front" *names* on a given category (handle or anti-handle on a mug,
spout on a teapot) is the model's Objaverse-side labelling, not something
this repo defines. Pin it once per category on the `compile_tsr` side
(canonical up/front/lateral symbols are caller-supplied) after looking at
an `alpha == 1` overlay; do not flip signs in the sidecar to make one
object look right.

### Run the bag demo (no ROS needed)

```bash
docker compose up -d oriany sam3          # sam3 optional: without it pass --bbox
python3 -m venv .venv && . .venv/bin/activate     # host: `python` may be 2.7
pip install mcap-ros2-support pyzmq msgpack msgpack-numpy pillow numpy

python test/oriany_bag_demo.py ros2bags/mug/mug_0.mcap --out outputs/oriany
python test/oriany_bag_demo.py ros2bags/mug/mug_0.mcap --frame 40 --prompt mug
python test/oriany_bag_demo.py ros2bags/mug/mug_0.mcap --bbox 300 200 420 340  # no sam3
```

Writes `frame.png`, `crop.png`, `crop_masked.png` (sam3 path, square-padded
to `--fg-ratio 0.85` exactly like `oriany_bridge_node.square_crop`),
`overlay.png` and `result.json` under `--out`, and prints `az/el/ro/alpha`
for both the rembg and the sam3-mask path. The overlay draws `up` (blue),
`front` (red), `lateral` (green) at the mask's 3D centroid with the bag's
`K` and aligned depth; the `az= el= ro= alpha=` line top-left is part of the
evidence — keep it in screenshots. Read `alpha` first: `0` → judge only
`up`; `2`/`4` → front is one of that many equivalent modes; `1` → front is a
committed prediction.

### Watch the axes in rviz2 (bag replay)

`ros2 run manip_bridge oriany_viz` is the ROS twin of the demo: same sam3 →
square-pad → oriany path, but publishes instead of drawing. It talks to the
sidecars directly, so `bridges.launch.py` need not be up. Inside
`Ros2Bridge` (`xhost +local:docker` on the host first; `colcon build` once
after pulling):

```bash
docker compose up -d sam3 oriany
docker exec -it Ros2Bridge bash
colcon build --symlink-install && . install/setup.bash
ros2 bag play /bags/mug --clock --loop &
ros2 run manip_bridge oriany_viz --ros-args -p use_sim_time:=true -- --prompt mug --once &
rviz2 --ros-args -p use_sim_time:=true
```

In rviz2: Fixed Frame = `camera_color_optical_frame` (the rgb `frame_id`),
add **PointCloud2** on `/camera/camera/depth/color/points` (Reliability →
Best Effort; the bag replays sensor QoS), **MarkerArray** on `/oriany/axes`
(Durability → Transient Local; the publisher is latched and a volatile
subscriber never sees the single `--once` message), and **TF** with
`oriany_mug` ticked. Arrows are front
(red) / lateral (green) / up (blue) from the mask's 3D centroid; the text
marker carries `az/el/ro/alpha`. When `alpha == 0` the front/lateral arrows
are drawn half-length as a visual flag that they are not committed. The TF
frame `oriany_<prompt>` is `R_cam` (x = front, y = lateral, z = up) at the
centroid — static under `--once` so it survives bag loops; otherwise
re-broadcast every `--every` rgb frames. The centroid is on the visible
surface (median mask depth), not the body centre.

Sharp edges:

- Top-down views (you can see into the mug) are the ill-conditioned regime
  for azimuth by construction — the front face is foreshortened to a rim.
  Prefer a frame with `el` around 30–50° when deciding the category
  convention.
- The two background paths should agree within a few bins. If they don't,
  the crop framing is the suspect, not the model: compare `crop_masked.png`
  against upstream's `resize_foreground(0.85)` output before anything else.
- `/oriany/orient` over ROS (`ros2 run manip_bridge run_scene ...`) goes
  through the same `R_cam`; `summary.json` now carries `R_cam`, not `R_obj`.

## AnyGrasp sidecar (port 5666)

Moved over from `ur5e-manip-sim`. Same ZMQ/pickle contract, so
`manip_sim/perception/grasp_client.py` works against it unchanged — point
the sim wing's `ANYGRASP_ADDR` at this box instead of at its own `grasp`
profile.

### One-time setup

Order matters: the feature id you register has to be the one the *pinned*
config produces, so pin first, register second, and only bring the service
up once the license is in place.

```bash
mkdir -p anygrasp_runtime/{checkpoints,license}

# 1. pin the MAC. Note the sed rather than `echo >>`: .env.example ships an
#    empty ANYGRASP_MAC=, and appending leaves two lines. Last-wins saves
#    you until something reorders them, and then you're debugging a silent
#    placeholder. One line, filled in.
cp .env.example .env
sed -i "s|^ANYGRASP_MAC=$|ANYGRASP_MAC=$(cat /sys/class/net/eno1/address)|" .env
bash scripts/preflight_anygrasp.sh eno1        # verifies the whole chain

# 2. build and read the id (no license needed for this)
docker compose build grasp
docker compose run --rm grasp feature-id

# 3. register that id, wait ~5 working days, then unpack the reply:
#      license zip      -> anygrasp_runtime/license/
#      checkpoint .tar  -> anygrasp_runtime/checkpoints/
docker compose run --rm grasp check
docker compose up -d grasp
docker compose logs -f grasp        # wait for "model ready, listening on :5666"
```

Record the id you submitted somewhere outside this tree. `anygrasp_runtime/`
is gitignored, so a `git clean -xdf` takes the license with it, and until
`licenseCfg.json` exists there is nothing in-repo saying what you registered.

**The weights are not a public download.** The SDK README only says "put
model weights under `log/`" and links nothing; `checkpoint_detection.tar`
arrives in the same approval email as the license. Put it directly in
`anygrasp_runtime/checkpoints/` — no `log/` nesting — since compose maps
that directory to `/opt/anygrasp/checkpoints`, which is where
`server.py --checkpoint_path` defaults.

#### If `.env` goes missing

`mac_address: ${ANYGRASP_MAC:-02:00:00:00:00:00}` degrades quietly, and
`:-` treats an *empty* value the same as unset — so replacing `.env` with
`.env.example`, or running compose from the wrong directory, silently
swaps in the placeholder. The entrypoint refuses to serve or print a
feature id under that MAC for exactly this reason: a placeholder-derived id
is a constant any machine would reproduce, useless to register and wrong to
serve with. `scan` is exempt, since rewriting the MAC is its job.

That guard only catches the placeholder. A *different* wrong MAC — a stale
one from another machine — passes it. `scripts/preflight_anygrasp.sh`
catches that case by comparing what compose resolved against the NIC
itself; once the license is mounted, the entrypoint's own
`feature_id`-vs-`licenseCfg.json` check covers it permanently.

### The feature-id problem, and why this service isn't host-networked

AnyGrasp's license is bound to a *feature id*, and that id is not a stable
machine identifier. From `gsnet.license_tools`:

```
macs       = sorted(set of MACs matched by /(?:ether|HWaddr)\s+([0-9A-Fa-f:.-]{12,17})/
                    in the output of `ifconfig`)
feature_id = "N" + f(sha256("mac=" + ",".join(macs)))
```

It hashes **the whole set of MAC addresses `ifconfig` reports**, and
net-tools `ifconfig` without `-a` reports every interface that is *UP*. So:

- On `network_mode: host` — what the sim repo's stanza uses — the container
  sees `eno1` **and** `docker0` **and** a `br-<id>` + `veth*` pair for every
  other running container. Bring up a different set of sidecars, create or
  tear down a compose network, attach a dock or a VPN, and the set changes,
  so the id changes, so the license stops validating. That's the drift you
  hit moving between the two repos: the sim box ran the `grasp` profile
  more or less alone; this box runs seven sidecars.
- On default bridge with no pin, Docker hands out a fresh random MAC per
  container, so the id changes on literally every `up`.

The fix here: the `grasp` service is the one non-host-networked service in
this compose file. It sits on its own `grasp_net` bridge with
`mac_address: ${ANYGRASP_MAC}` and publishes `5666`. `ifconfig` inside then
reports exactly one `ether` line (`lo` prints `loop`, not `ether`, so it
contributes nothing to the hash), always the same one. Set `ANYGRASP_MAC`
to the workstation's permanent NIC MAC and the fingerprint stays bound to
this physical machine while ignoring whatever else Docker is doing.

Second, independent source of drift: the SDK commit is now **pinned**
(`ANYGRASP_COMMIT=b8eaafc9…`). The 2026-07-04 SDK release replaced the
license tool outright and changed feature-id generation, so an unpinned
`main` can invalidate a working license on a rebuild. `docker/Dockerfile.anygrasp`
in the sim repo still floats `--branch main`; if you rebuild there, pin it too.

### Diagnosing

The entrypoint validates the license *before* loading the model, so a
failure is a fast restart loop with a legible message rather than a silent
`create_detector -> None`.

```bash
docker compose run --rm grasp feature-id     # what this container hashes to
docker compose run --rm grasp check          # validate the mounted license
```

If you have a working license but don't know which MAC produced it, sweep
the host's interfaces — `scan` rewrites `eth0`'s MAC in place, asks the SDK
for the resulting id, and flags the match:

```bash
docker compose run --rm grasp scan $(cat /sys/class/net/*/address | tr '\n' ' ')
```

If nothing matches, the id was hashed from a multi-MAC set (the host-network
case) and can't be reproduced from a single pinned interface. Re-register:
pin `ANYGRASP_MAC` first, take the id from `feature-id`, and submit it at
the SDK's [registration form](https://forms.gle/XVV3Eip8njTYJEBo6) (~5
working days). Because the MAC is pinned, that id will not drift again.

### Talking to other services

`grasp` is the only bridged service here, which makes its networking
asymmetric. Inbound is unaffected: `ros2-bridge` is host-networked so
`GRASP_ADDR=tcp://127.0.0.1:5666` hits the published port, and the sim wing
still reaches it via `grasp:host-gateway`.

Outbound is the part that changed. Inside a bridged container `127.0.0.1`
is the container's own loopback, *not* the host's — so the
`tcp://127.0.0.1:566x` pattern every other sidecar uses will fail from
inside `grasp`. Nothing needs it today (`server.py` is a pure REP loop that
never dials out), but if that changes, add
`extra_hosts: ["host.docker.internal:host-gateway"]` and address peers
through that name.

`grasp_bridge_node.py`, when it exists, belongs in `ros2_bridge/` running
inside `ros2-bridge` alongside the pose and pointso nodes — thin ZMQ
clients, no CUDA. Don't run ROS inside the grasp container; DDS discovery
across a NAT bridge isn't worth it.

Port `5666` is published on `0.0.0.0` so the sim wing can reach it from
another box. `server.py` does `pickle.loads()` on whatever arrives, which
is arbitrary code execution on deserialization, and Docker's DNAT rules
bypass `ufw`. Inherited from the sim design, not a regression from the
bridge move — but if the sim wing runs on this same workstation, narrow the
mapping to `172.17.0.1:5666:5666` (not `127.0.0.1`, which the
`host-gateway` path doesn't use).

### Smoke test

Pickle, not msgpack, unlike the other sidecars here:

```python
import pickle, zmq
s = zmq.Context().socket(zmq.REQ); s.connect("tcp://127.0.0.1:5666")
s.send(pickle.dumps({"cmd": "ping"})); print(pickle.loads(s.recv()))
```

### Still missing

No `grasp_bridge_node.py` in `ros2_bridge/` yet — `GRASP_ADDR` is already
plumbed into the `ros2-bridge` service, but nothing consumes it. The sidecar
also expects points in the **camera** optical frame, metres, float32; the
base-frame composition stays client-side.

## PointSO semantic orientation from meshes (no camera)

[#pointso-semantic-orientation-from-meshes-no-camera](#pointso-semantic-orientation-from-meshes-no-camera)

Camera-free harness for the PointSO sidecar: sample the object's visual
mesh into an Nx6 cloud, ship it over ZMQ, score the predicted semantic
directions against the calibrated `frames.json` symbols, and render the
predictions as arrow montages next to the interaction-point montages.
Meshes are written in the body frame of mjcf body `object` — the same
frame `frames.json` declares — so predictions come back directly
comparable to `pour_axis` / `up_axis` / handle geometry. PointSO
normalizes xyz internally (`pc_norm`), so only the frame's orientation
matters, not scale.

### Prerequisites

- `assets/objects/{mug,teapot}/meshes/` present (gitignored — run
  `scripts/convert_asset.py` on the machine first)
- the `pointso` sidecar from **ur5e-manip-hardware** serving on host
  `:5668`, checkpoint in place per that repo's README:

```
cd ../ur5e-manip-hardware
docker compose up -d pointso
docker compose logs -f pointso        # wait for "model ready, listening on :5668"
```

The sim services reach it as `tcp://pointso:5668` via the same
`host-gateway` alias the grasp sidecar uses (`POINTSO_ADDR` in compose).

### Run the scoring harness

```
docker compose run --rm sim python scripts/pointso_mesh_test.py
docker compose run --rm sim python scripts/pointso_mesh_test.py --camera-frame
docker compose run --rm sim python scripts/pointso_mesh_test.py --camera-frame --partial
docker compose run --rm sim python scripts/pointso_mesh_test.py --save-npz outputs/pointso
```

The three rows are an input-distribution ablation, weakest to most
faithful:

| flags                      | cloud                | frame                      |
| -------------------------- | -------------------- | -------------------------- |
| (none)                     | full surface         | z-up body frame            |
| `--camera-frame`           | full surface         | OpenCV camera (y down, z fwd) |
| `--camera-frame --partial` | single-view (HPR)    | OpenCV camera              |

PointSO's training input is the last row — a segmented single-view cloud
in the camera's frame — so that is the number that predicts hardware
behavior; the deltas between rows attribute error to frame convention vs
partiality vs the model itself. `--camera-frame` rotates predictions back
to the body frame before scoring, so all rows score against the same GT.

Per instruction the table prints the predicted unit vector (body frame),
the GT symbol it is scored against, and the angular error; teapot scores
against `pour_axis` / `up_axis` / the `handle_center` bearing, mug
against `up_axis` plus a handle bearing derived from the mesh (farthest
radial band about `opening_center`).

### Render the arrow montages

```
docker compose run --rm sim python scripts/render_pointso.py
docker compose run --rm sim python scripts/render_pointso.py --camera-frame --partial
docker compose run --rm sim python scripts/render_pointso.py --npz outputs/pointso
```

Output: `outputs/pointso/<object>_montage.png` — the eight canonical
views from `render_candidates.py` (same cameras, same projection) with,
per instruction, a **solid** arrow for the PointSO prediction and a
**dashed** arrow for the `frames.json` ground truth in the same color,
labeled with the angular error. Matching colors diverging is the whole
readout. Arrows pointing into a camera foreshorten to a small circle —
read those axes from the orthogonal views. Live mode takes the same
flags as the harness so the montage shows exactly what was scored;
`--npz` re-renders a `--save-npz` dump without touching the server.

### Sharp edges (verified the hard way)

1. **Sidecar edits need a restart, never a rebuild.** All ZMQ server code
   is volume-mounted; Python reads it once at process start. Symptom of a
   stale process: tracebacks whose line numbers match the old code but
   whose source lines are printed from the new file.
2. **Instructions are a LIST upstream.** SoFar's `pred_orientation` takes
   `n = len(instruction)`; a bare string is iterated per-character
   (`"opening"` → `n = 7` → `reshape(84, 1, 512)` crash). The server
   wraps `[ins]` per query; upstream's own n≥2 batching interleaves
   instructions across the 12-vote mean, so one-instruction-per-call is
   the only unambiguous pairing.
3. **`--camera-frame` results depend on the virtual viewpoint** — the
   model reasons in view frames, so sweep `--azimuth` before trusting a
   single number.
4. **`--partial` on dense real meshes keeps few points** (HPR hull size
   does not scale with cloud density); raise `--n-points` to ~50000 for
   partial runs so PointSO's 10k-point votes aren't resampling a
   700-point cloud.
5. **`small.pth` is the weak checkpoint.** If camera-frame + partial does
   not close the gap, swap the cfg/checkpoint pair at the top of
   `serve/pointso.py` to `base.yaml` + `base_finetune.pth`
   (Open6DOR-finetuned) and restart the sidecar — see edge #1.
6. Host `python` is 2.7 on the workstation; these scripts are
   container-first (`docker compose run --rm sim ...`). Bare-metal runs
   need the venv from the quick-start section and `PYTHONPATH=.`.