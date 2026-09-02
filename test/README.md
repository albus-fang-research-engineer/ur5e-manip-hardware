# Sidecar test suite

Integration tests for the perception/mapping sidecars, driven by a real RGB-D
frame captured from the `ur5e-manip-sim` TableTop scene (UR5e + teapot + mug).
No ROS2 anywhere: tests speak msgpack-ZMQ to the sidecars on localhost — the
same wire protocol the ROS bridge nodes use — so passing here means the exact
deployment interface works. The sim ground truth (rendered instance masks,
body poses, joint state) rides along in the packet and is what the outputs are
scored against.

## One-time: capture a frame packet

In `ur5e-manip-sim` (container or venv, headless):

```bash
MUJOCO_GL=egl PYTHONPATH=. python scripts/capture_rgbd_packet.py \
    --out outputs/frame_packets/packet
```

The script self-validates raster/intrinsics/extrinsics consistency by
projecting GT poses into the GT masks, and writes debug overlay PNGs — glance
at `overlay_robot.png` before trusting a new packet.

Copy the whole output dir (packet.npz + `*.obj` + `*_frames.json`) here:

```bash
cp -r ../ur5e-manip-sim/outputs/frame_packets/packet test/data/packet
# or: export FRAME_PACKET=/path/to/packet.npz
```

The `.obj` meshes are gitignored in the sim repo, so capture on the machine
that has them (the FoundationPose / TRELLIS-metric / sphere-fit tests skip
without them).

## Host-side tests (ZMQ, from the repo root)

```bash
docker compose up -d sam3 pose trellis2 curobo   # whichever subset you want
python -m pytest test/ -v                        # down sidecars SKIP, not fail
python -m pytest test/ -v -m "not slow"          # skip TRELLIS generation (~min)
```

Host deps: `pip install pytest numpy pyzmq msgpack msgpack-numpy scipy trimesh`
(scipy/trimesh only needed by a couple of tests; they importorskip).

- `test_sam3.py` — open-vocab masks vs GT (IoU), score ordering, absent-concept
  N=0 contract, segment/segment_multi consistency, threshold monotonicity.
- `test_foundationpose.py` — register vs GT `cam_T_obj` (3 cm / 15°), static
  track consistency, session lifecycle errors. Copies the packet teapot mesh
  into `foundationpose_runtime/meshes/` automatically.
- `test_trellis2.py` — canonical unit-box geometry contract, GLB on the shared
  mount, metric-scale branch vs the GT mesh extent (±35%).
- `test_curobo_esdf.py` — configure/integrate/esdf over ZMQ; ESDF sign checks
  from a backprojected table pixel (free probe positive, behind-surface probe
  below it), voxel-size override, reset semantics, save/load round-trip.
  Restores the server's env-default config afterwards.

## cuRobo library tests (inside the curobo container)

Robot self-masking, mesh attachment, and sphere approximation are cuRoboV2
library features, not mapper-server commands, so they run in-container:

```bash
docker compose run --rm -v $PWD/test:/opt/test curobo \
    bash -lc "pip install -q pytest trimesh scipy && \
              python -m pytest /opt/test/curobo_incontainer -v -s"
```

`ur5e_curobo_config.py` generates a UR5e config on first run (cuRoboV2 ships
UR5e meshes but only a ur10e config): ur10e.urdf with the six UR5e kinematic
lengths substituted **and** the five link mesh origins (`UR5E_LINK_ORIGINS` —
ur_description places each link's mesh via `shoulder_offset`/`elbow_offset`
and the wrist `visual_offset`s, which are separate constants from the DH
table and do not follow from the joint origins), then
`RobotBuilder.fit_collision_spheres()` on the real
link meshes — which is itself the robot-sphere-approximation test. Cached in
`/tmp/ur5e_curobo` (`CUROBO_TEST_CACHE` to move it, `CUROBO_TEST_FIT_TYPE=
morphit` for the deploy-quality fit; default `voxel` for test speed). Do not
reuse this config for inverse-dynamics/torque features — inertials are still
ur10e values.

To eyeball the fit rather than assert on it, `visualize_ur5e_fit.py` serves
the saved config in viser (the container is host-net, so viser lands on
`localhost:8080`; over ssh, `ssh -L 8080:localhost:8080 <box>`):

```bash
docker exec -it CuroboServer bash -lc \
    "CUROBO_TEST_CACHE=/data/robot python /opt/robot_builder/visualize_ur5e_fit.py"
```

`--refit` ignores the saved yml and fits fresh (also skips the ~1 min
link-pair pruning pass, so prefer it when iterating). Note that both cached
artifacts are reused unconditionally — `build_ur5e_config` returns early if
`ur5e.yml` exists and `--refit` regenerates `ur5e.urdf` only when missing —
so after changing the generator, delete both or you will keep viewing the
old fit.

The self-masking test evaluates the base yaw at 0 and π and reports which
wins: robosuite's UR5e MJCF and ur_description can disagree by the
`base_link`/`base_link_inertia` π flip, and the printed answer is the
calibration fact to bake into the real camera extrinsics.
