"""Probe 2: the modules probe 1 mis-named -- inverse_kinematics,
collision_checking, motion_planner internals, Scene/VoxelGrid, the
self-collision config and RobotCollisionGeometry. Class-level introspection
(signatures + docstring first lines), no solver instantiation.

    docker compose run --rm -v $PWD/test:/opt/test curobo \\
        bash -lc "python /opt/test/curobo_incontainer/probe_v2_api2.py"
"""

import importlib
import inspect
import sys

import torch

sys.path.insert(0, "/opt/robot_builder")


def first_line(doc):
    return (doc or "").strip().split("\n")[0][:110]


def show_class(cls, keys=None, depth=1):
    print(f"\n== {cls.__module__}.{cls.__name__}: {first_line(cls.__doc__)}")
    for n, m in inspect.getmembers(cls):
        if n.startswith("_") and n != "__init__":
            continue
        if keys and not any(k in n.lower() for k in keys):
            continue
        if inspect.isfunction(m) or inspect.ismethod(m):
            try:
                sig = str(inspect.signature(m))
            except (TypeError, ValueError):
                sig = "(...)"
            print(f"   {n}{sig}")
            if depth > 1 and m.__doc__:
                print(f"       {first_line(m.__doc__)}")
        elif isinstance(m, property):
            print(f"   {n}  [property]")
    # dataclass fields
    f = getattr(cls, "__dataclass_fields__", None)
    if f:
        print("   fields:", ", ".join(f"{k}: {getattr(v.type, '__name__', v.type)}" for k, v in f.items())[:600])


def show_module(name):
    try:
        mod = importlib.import_module(name)
    except Exception as e:
        print(f"\n== {name}: not importable ({type(e).__name__}: {e})")
        return None
    pub = [n for n in dir(mod) if not n.startswith("_")]
    print(f"\n== {name}: {pub}")
    return mod


ik = show_module("curobo.inverse_kinematics")
if ik:
    for n in dir(ik):
        o = getattr(ik, n)
        if inspect.isclass(o) and not n.startswith("_"):
            show_class(o, depth=2)

cc = show_module("curobo.collision_checking")
if cc:
    for n in dir(cc):
        o = getattr(cc, n)
        if inspect.isclass(o) and not n.startswith("_"):
            show_class(o, keys=("__init__", "collision", "distance", "sdf", "check", "update",
                                "sphere", "world", "voxel", "attach", "self", "cfg", "create",
                                "from"), depth=2)

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg  # noqa: E402
show_class(MotionPlanner, depth=2)
show_class(MotionPlannerCfg, depth=1)

from curobo.scene import Scene, VoxelGrid  # noqa: E402
show_class(Scene, depth=2)
show_class(VoxelGrid, depth=1)

from curobo.kinematics import Kinematics, KinematicsCfg  # noqa: E402
from curobo.types import JointState  # noqa: E402
from ur5e_curobo_config import build_ur5e_config  # noqa: E402
cfg = build_ur5e_config()
kin = Kinematics(KinematicsCfg.from_data_dict(cfg["kinematics"]))
sc = kin.get_self_collision_config()
show_class(type(sc), depth=1)
q = torch.zeros((1, 6), device="cuda")
st = kin.compute_kinematics(JointState.from_position(q, joint_names=list(kin.joint_names)))
show_class(type(st.robot_collision_geometry), depth=2)
print("\n   tool_jacobians shape", tuple(st.tool_jacobians.shape),
      "-- rows 0:3 linear, 3:6 angular? check:", st.tool_jacobians[0, 0, 0].detach().cpu().numpy().round(3).tolist())
print("   kin.compute_jacobian =", kin.compute_jacobian)
print("   self_collision_config attrs:", [a for a in dir(sc) if not a.startswith("_")][:40])
