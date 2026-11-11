# Hardware runbook: pour tea, VLM method

One trial, from a fresh scene to the pour on the UR5e. The trial is scored
at the end of the `pour` step; the steps after it (unpour, return, lower,
release, retreat) are cleanup.

Checkouts (nothing to switch; all are already on these branches):

| What | Where | Branch |
|---|---|---|
| Drivers, planner, executor | `~/ur5e-manip-hardware` | `vlm-port` |
| Sim code for VLM calls and TSR compiling | `~/ur5e-manip-sim-critic` (worktree) | `critic-repair` |
| TSR core, CBiRRT | `~/manip-tsr`, `~/manip-cbirrt` | `main` |
| Mounted in Ros2Bridge as `/opt/manip-sim` | `~/ur5e-manip-sim` | `insert-marker` (leave it) |

The container's sim is NOT the critic branch, so every step that compiles
TSRs (call #3, call #4 and the planner) runs on the HOST against
`~/ur5e-manip-sim-critic` (`VLM_HOST_SIM=$SIM` for the paid ones). Calls
#1 and #2 compile nothing and run in the container as on 11/05. The
executor imports no sim code, so it runs in the container as before.

Status: steps 1-3 are the 11/05 procedure. Step 4 in host mode is new:
its `--dry-run` (free) is the check. Steps 5-7 ran offline on the 11/05
run (planner on the host, critic in host mode, replay, `--dry`). No trial
has run end to end on the robot with this code.

## 0. Setup (every terminal)

```bash
cd ~/ur5e-manip-hardware
STAMP=<run stamp>                       # set after step 1
R=outputs/runs/$STAMP                   # host path
RC=/data/runs/$STAMP                    # the same dir inside Ros2Bridge
SIM=~/ur5e-manip-sim-critic
HOSTPY="env SIM_DIR=$SIM PYTHONPATH=$PWD/../manip-tsr:$SIM:$PWD/../manip-cbirrt:$PWD/ros2_ws/src/manip_bridge $HOME/miniconda3/envs/manip-tsr/bin/python"
```

Paid VLM calls go through `scripts/vlm_run.sh` (the key comes from `.env`;
every call is appended to `outputs/vlm_calls.log`). Run each paid step
with `--dry-run` first where it has one; that is free.

Before anything moves: the robot is free (no teleop container driving it),
the laptop driver stack and gripper node are up, the CuroboServer sidecar
answers on tcp://127.0.0.1:5671, and the e-stop is in reach.

## 1. Scene capture and registration (README)

Objects on the table, robot at its capture pose. README sections, in order:
"Online test from a rosbag" (`run_scene` live: marks, then `--register`
teapot and mug), "Render asset from the tracked mesh", "Canonical render
asset", "Grounding seam". Output: `outputs/runs/<stamp>/` with
`summary.json`. Look at `marks/marked.png` and the reproject images before
going on.

## 2. Grasp (as 11/05: taught grasp)

In Ros2Bridge:

```bash
export PYTHONPATH=/root/ros2_ws/src/manip_bridge:$PYTHONPATH
python3 /data/runs/taught_grasp.py --run $RC --key m2 \
    --grasps /data/runs/library/lib_teapot.taught_grasps.json
python3 /data/runs/grasp_reach.py --run $RC --obj m2_taught      # -> m3/reach.json
```

(`m2` = the teapot's mark on 11/05; use this run's key.)

## 3. VLM calls #1 and #2 (paid)

Container mode (no `VLM_HOST_SIM`), container paths:

```bash
scripts/vlm_run.sh plan_scene.py --run $RC --task "pour tea into the mug"
# run the grounding commands it prints (in Ros2Bridge), then bind the roles (no API call):
scripts/vlm_run.sh plan_scene.py --run $RC --task "pour tea into the mug" \
    --replay $RC/stage_plan/plan.log.json --grounding $RC/grounding_vlm \
    --object teapot=<id> --object mug=<id>
scripts/vlm_run.sh select_scene.py --run $RC                     # -> call2/selections.json
```

Keep `--task` identical in every step (it is part of the plan's sha).

## 4. VLM call #3 (paid: 3 calls, more on compile retries)

```bash
VLM_HOST_SIM=$SIM scripts/vlm_run.sh emit_scene.py --run $R --dry-run   # free: renders + request
VLM_HOST_SIM=$SIM scripts/vlm_run.sh emit_scene.py --run $R             # -> call3/emissions.json
```

## 5. Plan (+ optional critic, call #4)

The planner's defaults: clearance margin 0.01 m, densify 0.005 rad, 5
transport seeds. Without the critic (no API call):

```bash
$HOSTPY outputs/runs/plan_stages_hw.py --run $R --method vlm \
    --emissions $R/call3/emissions.json                          # -> m4_vlm/
```

With the critic (paid: 1-3 calls). Export the gripper mesh once per run:

```bash
$HOSTPY outputs/runs/critic_scene.py export-gripper --run $R
VLM_HOST_SIM=$SIM scripts/vlm_run.sh plan_stages_hw.py --run $R --method vlm \
    --emissions $R/call3/emissions.json --critic preview --critic-dry-run   # free
VLM_HOST_SIM=$SIM scripts/vlm_run.sh plan_stages_hw.py --run $R --method vlm \
    --emissions $R/call3/emissions.json --critic preview
```

Read `m4_vlm/report.json`: both stages planned, and no `dense.refused`
(a TOPP-RA refusal; deferred H8 -- re-plan with `--seed-start 5`).

## 6. Replay check (offline, about 2 minutes)

```bash
$HOSTPY outputs/runs/critic_scene.py export-gripper --run $R     # if not done in step 5
MUJOCO_GL=glfw OMP_NUM_THREADS=4 $HOSTPY scripts/mujoco_replay/replay_mujoco.py \
    --run $R --emissions $R/call3/emissions.json --selections $R/call2/selections.json \
    --plans $R/m4_vlm --out $R/m4_vlm/replay --video
```

Look at `m4_vlm/replay/m4_vlm.mp4`. Go on when the printed line shows no
FLAGS (every clearance >= 5 mm) and the stream lands inside the opening in
every pour frame. Log only: this decides nothing for you.

## 7. Dry build (in Ros2Bridge; nothing moves)

```bash
export PYTHONPATH=/root/ros2_ws/src/manip_bridge:$PYTHONPATH
python3 /data/runs/exec_plan.py $RC/m4_vlm/plan.json --dry       # rviz: /exec/markers
```

Every step must build (no MotionRefusal).

## 8. Run on the robot

The robot must be where the scene was captured (the plan starts there;
`exec_plan` refuses a start more than 0.01 rad away).

```bash
python3 /data/runs/exec_plan.py $RC/m4_vlm/plan.json             # --speed 0.10 default
```

Every motion waits at a gate: Enter = go, anything else = stop. Ctrl-C
cancels a motion; the e-stop stays with the operator. Score the trial at
the end of `pour`. `m4_vlm/exec.json` records every step.
