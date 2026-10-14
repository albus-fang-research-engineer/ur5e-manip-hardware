"""The ONE place hardware imports the sim's stage-1 grasp TSR
(manip_sim.grasping.handle_grasp_tsr).

Rule: every sim module hardware uses gets its own *_compat.py, and nothing
else in this repo imports manip_sim (test/test_sim_imports.py enforces it).

What grasp_tsr (M3 step 2) uses, numpy/scipy only:
  handle_grasp_tsr   (T0_body, Frame, approach_h, elevation) -> sim TSR
                     with the sim's default bounds, unchanged: the hardware
                     grasp region is the sim's, so results stay comparable.

The returned TSR is the sim's class (manip_sim.tsr.TSR), not manip_tsr's.
Hardware crosses over through the TSR's numbers only (T0_w, Tw_e, Bw),
written as a tsr_spec file and read back with manip_tsr by tsr_spec.
"""

try:
    import manip_sim.grasping as _grasping
    from manip_sim.grasping import handle_grasp_tsr
except ImportError as e:
    raise ImportError(
        "grasping_compat: cannot import manip_sim.grasping. The sim checkout must be "
        "bind-mounted at /opt/manip-sim and on PYTHONPATH (docker-compose: "
        "${SIM_DIR:-../ur5e-manip-sim}:/opt/manip-sim:ro) -- check inside the container:\n"
        "    ls /opt/manip-sim/manip_sim/grasping.py\n"
        "    echo $PYTHONPATH") from e

from manip_bridge.sim_provenance import provenance as _provenance  # noqa: E402

SOURCES = (_grasping.__file__,)

__all__ = ["handle_grasp_tsr", "provenance", "SOURCES"]


def provenance():
    return _provenance(SOURCES)
