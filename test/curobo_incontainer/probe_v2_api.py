"""Print the cuRoboV2 API surface the CBiRRT backend needs, so the backend is
written against what THIS image has rather than remembered from cuRobo v0.7.
Read-only: builds the robot config (cached yml), a Kinematics, and lists
attributes. No ESDF, no planner build (those take minutes).

    docker compose run --rm -v $PWD/test:/opt/test curobo \\
        bash -lc "python /opt/test/curobo_incontainer/probe_v2_api.py"

Paste the output back; the backend's collision + IK calls get written from it.
"""

import importlib
import inspect
import pkgutil
import sys

import torch

sys.path.insert(0, "/opt/robot_builder")
import curobo  # noqa: E402
from curobo.kinematics import Kinematics, KinematicsCfg  # noqa: E402
from curobo.types import JointState  # noqa: E402
from ur5e_curobo_config import build_ur5e_config  # noqa: E402

KEYS = ("collision", "distance", "sdf", "esdf", "self", "sphere", "ik", "solve",
        "jacobian", "fk", "kinematic", "check", "valid", "constraint", "attach",
        "world", "scene", "voxel", "tool", "link")


def show(obj, title, keys=KEYS):
    names = [n for n in dir(obj) if not n.startswith("__")
             and any(k in n.lower() for k in keys)]
    print(f"\n== {title}: {type(obj).__module__}.{type(obj).__name__}")
    for n in names:
        try:
            a = getattr(obj, n)
        except Exception as e:  # properties that need state
            print(f"   {n}: <{type(e).__name__}>")
            continue
        if callable(a):
            try:
                sig = str(inspect.signature(a))
            except (TypeError, ValueError):
                sig = "(...)"
            print(f"   {n}{sig}")
        else:
            t = type(a).__name__
            shape = getattr(a, "shape", None)
            print(f"   {n}: {t}{'' if shape is None else ' ' + str(tuple(shape))}")


print("curobo", getattr(curobo, "__version__", "?"), "from", curobo.__file__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())

print("\n== curobo submodules")
for m in pkgutil.iter_modules(curobo.__path__):
    print("  ", m.name, "(pkg)" if m.ispkg else "")
for cand in ("curobo.ik", "curobo.motion_planner", "curobo.scene", "curobo.collision",
             "curobo.rollout", "curobo.perception", "curobo._src.collision",
             "curobo._src.ik", "curobo.types"):
    try:
        mod = importlib.import_module(cand)
        pub = [n for n in dir(mod) if not n.startswith("_")]
        print(f"\n== {cand}: {pub}")
    except Exception as e:
        print(f"\n== {cand}: not importable ({type(e).__name__}: {e})")

cfg = build_ur5e_config()
kin = Kinematics(KinematicsCfg.from_data_dict(cfg["kinematics"]))
show(kin, "Kinematics")
show(kin.config, "KinematicsCfg")
q = torch.zeros((1, len(kin.joint_names)), device="cuda")
st = kin.compute_kinematics(JointState.from_position(q, joint_names=list(kin.joint_names)))
show(st, "KinematicsState", keys=("",))          # everything on the state
print("\n   joint_names", list(kin.joint_names))
print("   tool_frames", list(kin.tool_frames))
tp = st.tool_poses.get_link_pose(kin.tool_frames[0])
print("   tool0 @ q=0  pos", tp.position.detach().cpu().numpy().ravel().round(4),
      " quat(wxyz?)", tp.quaternion.detach().cpu().numpy().ravel().round(4))
print("   robot_spheres", tuple(st.robot_spheres.shape))
print("\n== Pose class:", type(tp).__module__, type(tp).__name__)
show(tp, "Pose", keys=("",))
