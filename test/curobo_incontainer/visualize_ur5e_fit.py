"""Render the UR5e collision-sphere fit in viser (meshes + spheres).

Runs INSIDE the curobo container (this dir is mounted at /opt/robot_builder):

    docker exec -it CuroboServer bash -lc \
        "CUROBO_TEST_CACHE=/data/robot python /opt/robot_builder/visualize_ur5e_fit.py"

then open http://localhost:8080 (container is host-net; over ssh:
`ssh -L 8080:localhost:8080 <box>`). Ctrl+C to stop.

By default this loads the SAVED config (CUROBO_TEST_CACHE/ur5e.yml, i.e. the
exact fit the server/segmenter/planner consume) via RobotBuilder.from_config.
If that trips on the injected attached_object extra-link fields, or --refit
is given, it re-fits from the generated URDF with the same parameters as
ur5e_curobo_config (~3 s; the slow collision-matrix pass is skipped -- it
does not affect the spheres).

What to look for: spheres SHOULD protrude (over-approximation is the point).
The failure mode is the opposite -- link mesh poking out un-sphered,
especially wrist_3/tool0, where both the self-mask and self-collision
checking care most. Toggle sphere visibility in the viser GUI.

The arm you see is drawn by ViserVisualizer.__init__, which loads the visual
meshes through ViserUrdf and poses every link by forward kinematics. That is
the view you want and it is always on.

RobotBuilder.visualize(show_meshes=True) adds a SECOND, independent set of
geometry on top: the collision STLs, added with transform_with_pose=True,
which applies only each mesh's origin within its own link frame and runs no
FK. All of them therefore land stacked at the world origin -- a white mass
of unposed arm solids beside the base, easily mistaken for a stray object in
the scene. It is a duplicate of geometry ViserUrdf already drew correctly,
so it is off by default here; --collision-overlay opts back in.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from curobo._src.util_file import get_assets_path  # noqa: E402
from curobo.robot_builder import RobotBuilder  # noqa: E402

from ur5e_curobo_config import CACHE, _generate_ur5e_urdf  # noqa: E402


def _builder_from_saved(yml: Path) -> RobotBuilder:
    return RobotBuilder.from_config(str(yml))


def _builder_refit(fit_type: str, density: float) -> RobotBuilder:
    from curobo._src.geom.sphere_fit.types import SphereFitType

    urdf = CACHE / "ur5e.urdf"
    if not urdf.exists():
        _generate_ur5e_urdf(urdf)
    asset_root = str(Path(get_assets_path()) / "robot" / "ur_description")
    b = RobotBuilder(str(urdf), asset_path=asset_root, tool_frames=["tool0"])
    b.fit_collision_spheres(
        fit_type=SphereFitType[fit_type.upper()],
        sphere_density=density,
        use_collision_mesh=True,
        compute_metrics=True,
    )
    for link, m in sorted(b.link_metrics.items()):
        vals = {k: v for k, v in vars(m).items() if isinstance(v, (int, float))}
        print(f"  {link:20} " + " ".join(f"{k}={v:.3f}" for k, v in vals.items()))
    return b


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(CACHE / "ur5e.yml"),
                    help="saved yml to view (default: the one the stack uses)")
    ap.add_argument("--refit", action="store_true",
                    help="ignore the saved yml; fit fresh and view that")
    ap.add_argument("--fit-type", default=os.environ.get(
        "CUROBO_TEST_FIT_TYPE", "voxel"), help="refit only")
    ap.add_argument("--density", type=float, default=1.0, help="refit only")
    ap.add_argument("--collision-overlay", action="store_true",
                    help="also add the collision STLs at the world origin "
                         "(unposed -- see module docstring; off by default)")
    ap.add_argument("--hide-spheres", action="store_true")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    yml = Path(args.config)
    builder = None
    if not args.refit and yml.exists():
        try:
            builder = _builder_from_saved(yml)
            print(f"loaded saved fit: {yml} ({builder.num_spheres} spheres)")
        except Exception as e:
            print(f"from_config({yml}) failed ({type(e).__name__}: {e}); "
                  f"falling back to a fresh fit with the module's parameters")
    if builder is None:
        builder = _builder_refit(args.fit_type, args.density)
        print(f"fresh fit: {builder.num_spheres} spheres "
              f"({args.fit_type}, density {args.density})")

    builder.visualize(
        port=args.port,
        show_meshes=args.collision_overlay,
        show_spheres=not args.hide_spheres,
    )
    print(f"viser: http://localhost:{args.port}  (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
