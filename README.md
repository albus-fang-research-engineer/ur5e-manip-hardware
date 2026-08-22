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
origin; compose `R_obj` with a pose sidecar translation client-side.

`Orient` reuses `GenerateMesh`'s mask convention: an **empty (0x0)** mask
means the model mattes the full frame itself (rembg, upstream `app.py`); a
**non-empty** mask means the bridge crops to the mask bbox, fills the
background, and square-pads to `fg_ratio` (0.85) before sending with
`remove_bkg=false`. That padding is not cosmetic — upstream only runs
`resize_foreground` inside the rembg path, so a tight bbox crop is a framing
the model never trained on. `run_scene --oriany-matting` switches to the
matting path so the two are directly comparable; the `bg_fill` parameter
(the fill colour under the mask) is *not* verified against upstream and is
worth sweeping.

`mesh` in `EstimatePose` is a filename under the sidecar's `/opt/meshes` **or**
an absolute path: `./trellis2_runtime/outputs` is mounted read-only at
`/data/meshes` in the pose and any6d containers, so a TRELLIS.2 metric GLB
from `/trellis2/generate_mesh` can be passed straight in.

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