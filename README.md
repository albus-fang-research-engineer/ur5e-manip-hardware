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
- **Registration masks**: the pose bridge expects an instance mask on
  `/pose_bridge/mask`. In sim, publish the ground-truth instance mask; on
  hardware, front it with your SAM/Florence segmenter (which SoFar already
  bundles if you later widen the pointso image to full SoFar).

## AnyGrasp sidecar (port 5666)

Moved over from `ur5e-manip-sim`. Same ZMQ/pickle contract, so
`manip_sim/perception/grasp_client.py` works against it unchanged — point
the sim wing's `ANYGRASP_ADDR` at this box instead of at its own `grasp`
profile.

### One-time setup

```bash
mkdir -p anygrasp_runtime/checkpoints anygrasp_runtime/license
# checkpoint_detection.tar from the AnyGrasp SDK's Google Drive link
# license/ = the zip the authors mail back (see below)

cp .env.example .env
echo "ANYGRASP_MAC=$(cat /sys/class/net/eno1/address)" >> .env   # your real NIC
docker compose up -d --build grasp
docker compose logs -f grasp        # wait for "model ready, listening on :5666"
```

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