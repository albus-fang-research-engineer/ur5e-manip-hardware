# Perception wing: FoundationPose + PointSO + ROS2 bridge

Three new services following the existing `grasp` sidecar pattern. Everything
is host-networked; fixed ports: grasp `5666` (existing), pose `5667`,
pointso `5668`.

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