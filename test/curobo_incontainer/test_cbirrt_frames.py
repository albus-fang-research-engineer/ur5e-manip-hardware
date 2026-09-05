"""manip_cbirrt's numpy UR5e DH chain vs cuRoboV2's FK: discover the fixed
T_base / T_tool that make them agree, and demand agreement to sub-0.1 mm.

Why this test exists: CBiRRT's inner loop runs on the DH chain (microsecond
FK), collision and IK run on cuRobo. They MUST describe the same tool0
frame or every plan is silently off by the discrepancy. UR's DH base frame
vs ur_description's base_link (pi about z) and DH frame 6 vs tool0 are
fixed rigid offsets; rather than hard-code a convention, this solves for
them: for each candidate T_base, T_tool(q) = inv(T_base @ T_dh6(q)) @
T_curobo_tool0(q) must be the SAME matrix for every q. The one that is
constant wins, and its value is printed -- that is what the backend uses.

Runs inside the curobo container (needs curobo + manip_cbirrt on PYTHONPATH,
see docker-compose curobo volumes):

    docker compose run --rm -v $PWD/test:/opt/test curobo \\
        bash -lc "pip install -q pytest scipy && \\
                  python -m pytest /opt/test/curobo_incontainer/test_cbirrt_frames.py -v -s"
"""

import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("curobo")
pytest.importorskip("manip_cbirrt", reason="mount ../manip-cbirrt at /opt/manip-cbirrt (compose)")

sys.path.insert(0, "/opt/robot_builder")
from curobo.kinematics import Kinematics, KinematicsCfg  # noqa: E402
from curobo.types import JointState  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402
from ur5e_curobo_config import build_ur5e_config  # noqa: E402

from manip_cbirrt import ur5e_chain  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("cuRobo needs CUDA", allow_module_level=True)

# UR DH joint order; cuRobo's joint_names come from the URDF
DH_ORDER = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]


@pytest.fixture(scope="module")
def kin():
    cfg = build_ur5e_config()
    return Kinematics(KinematicsCfg.from_data_dict(cfg["kinematics"]))


def curobo_fk(kin, q_dh: np.ndarray, quat_order: str = "wxyz") -> np.ndarray:
    """tool0 pose (4x4) from cuRobo for a config given in DH joint order.
    cuRobo Pose quaternions are wxyz; `quat_order` exists so the test can
    show that the OTHER reading fails, not just that this one is orthonormal."""
    names = list(kin.joint_names)
    q = torch.zeros((1, len(names)), device="cuda", dtype=torch.float32)
    for i, n in enumerate(DH_ORDER):
        q[0, names.index(n)] = float(q_dh[i])
    st = kin.compute_kinematics(JointState.from_position(q, joint_names=names))
    p = st.tool_poses.get_link_pose(kin.tool_frames[0])
    pos = p.position.detach().cpu().numpy().reshape(3).astype(float)
    quat = p.quaternion.detach().cpu().numpy().reshape(4).astype(float)
    xyzw = [quat[1], quat[2], quat[3], quat[0]] if quat_order == "wxyz" else list(quat)
    T = np.eye(4)
    T[:3, :3] = R.from_quat(xyzw).as_matrix()
    T[:3, 3] = pos
    return T


def _rz(a):
    T = np.eye(4); T[:3, :3] = R.from_euler("z", a).as_matrix(); return T


def discover(kin, n=12, seed=0, quat_order="wxyz"):
    """-> (T_base, T_tool, max_dev, label) for the T_base candidate whose
    implied T_tool is constant over n random configs."""
    rng = np.random.default_rng(seed)
    qs = [rng.uniform(-np.pi, np.pi, 6) for _ in range(n)]
    Tc = [curobo_fk(kin, q, quat_order) for q in qs]
    chain = ur5e_chain()
    best = None
    for label, Tb in (("identity", np.eye(4)), ("Rz(pi)", _rz(np.pi))):
        Tt = [np.linalg.inv(Tb @ chain.fk(q)) @ T for q, T in zip(qs, Tc)]
        Tt0 = Tt[0]
        dev = max(np.abs(T - Tt0).max() for T in Tt[1:])
        print(f"  T_base={label:8s} implied T_tool deviation across {n} configs: {dev:.2e}")
        if best is None or dev < best[2]:
            best = (Tb, Tt0, dev, label)
    return best


def test_dh_chain_matches_curobo_tool0(kin):
    Tb, Tt, dev, label = discover(kin)
    print(f"\n  winner: T_base = {label}")
    print("  T_tool (DH frame 6 -> tool0) =\n", np.array2string(Tt, precision=6, suppress_small=True))
    print("  T_tool rotation as rpy_deg:",
          np.rad2deg(R.from_matrix(Tt[:3, :3]).as_euler("xyz")).round(3),
          " translation:", Tt[:3, 3].round(6))
    assert dev < 1e-4, ("no constant T_tool for either base convention: the DH "
                        "parameters in manip_cbirrt.UR5E_DH disagree with cuRobo's URDF")
    np.testing.assert_allclose(Tt[:3, :3].T @ Tt[:3, :3], np.eye(3), atol=1e-6)

    # and the composed chain reproduces cuRobo on fresh configs to < 0.1 mm / 1e-4 rad
    chain = ur5e_chain(T_base=Tb, T_tool=Tt)
    rng = np.random.default_rng(1)
    worst_p, worst_r = 0.0, 0.0
    for _ in range(20):
        q = rng.uniform(-np.pi, np.pi, 6)
        A, B = chain.fk(q), curobo_fk(kin, q)
        worst_p = max(worst_p, np.linalg.norm(A[:3, 3] - B[:3, 3]))
        worst_r = max(worst_r, np.linalg.norm(R.from_matrix(A[:3, :3].T @ B[:3, :3]).as_rotvec()))
    print(f"  fresh-config agreement: pos {worst_p*1e3:.4f} mm, rot {worst_r:.2e} rad")
    assert worst_p < 1e-4 and worst_r < 1e-4


def test_curobo_quaternion_is_wxyz(kin):
    """A misread quaternion order still yields an orthonormal matrix, so
    orthonormality proves nothing. Constancy of the implied T_tool does:
    under the right decoding one base convention gives a constant T_tool;
    under the wrong decoding neither does."""
    _, _, dev_ok, _ = discover(kin, quat_order="wxyz")
    _, _, dev_bad, _ = discover(kin, quat_order="xyzw")
    print(f"\n  implied-T_tool deviation: wxyz {dev_ok:.2e}  xyzw {dev_bad:.2e}")
    assert dev_ok < 1e-4 < dev_bad
