"""The ONE place hardware imports the sim's hand-authored pour arm:
manip_sim.pour_stages (TaskFrames, the transport and pour TSR pairs), the
Frame and TSR types it builds on, and the mesh calibration that derives its
symbols (scripts/calibrate_frames_from_mesh.py: spout tip, pour/tilt axes,
mug opening and rim radius).

This is the ground-truth arm of the emission ablation in the sim: hand
compilers over hand/mesh-calibrated symbols. On hardware it is the
comparison arm for the VLM-authored stages, and TaskFrames switches to
compile_stage when given emissions, so the planner driver is shared.

Frame convention the sim assumes and hardware must supply: the object's
"body" for these functions has +z = up (sim assets are authored that way).
A tracked hardware mesh's body frame is arbitrary, so callers pass poses and
symbols in the object's CANONICAL frame ([x=front, y=up x front, z=up], the
frame grounding already uses); plan_stages_hw.py does the change of frame.

scripts/ has no __init__.py: `scripts.calibrate_frames_from_mesh` resolves
as a namespace package off /opt/manip-sim, as in stageplan_compat. It
imports numpy only.
"""
try:
    import manip_sim.pour_stages as _ps
    from manip_sim.frames import Frame
    from manip_sim.pour_stages import TaskFrames, pour_pair, transport_pair
    from manip_sim.tsr import TSR
    from scripts.calibrate_frames_from_mesh import calibrate_mug, calibrate_teapot, mesh_pour_axis
except ImportError as e:
    raise ImportError(
        "pour_stages_compat: cannot import manip_sim.pour_stages / "
        "scripts.calibrate_frames_from_mesh. The sim checkout must be bind-mounted at "
        "/opt/manip-sim and on PYTHONPATH (docker-compose: "
        "${SIM_DIR:-../ur5e-manip-sim}:/opt/manip-sim:ro) -- check inside the container:\n"
        "    ls /opt/manip-sim/manip_sim/pour_stages.py "
        "/opt/manip-sim/scripts/calibrate_frames_from_mesh.py") from e

SIM_MODULE = _ps.__file__

__all__ = ["Frame", "TSR", "TaskFrames", "transport_pair", "pour_pair",
           "calibrate_teapot", "calibrate_mug", "mesh_pour_axis", "SIM_MODULE"]
