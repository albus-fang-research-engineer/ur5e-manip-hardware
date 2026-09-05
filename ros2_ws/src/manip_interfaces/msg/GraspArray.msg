# Parallel-jaw grasp proposals with the per-grasp scalars a PoseArray drops.
#
# Two conventions are carried on this one type; which one applies is the
# publisher's contract and is stated on the topic:
#   /grasp/<obj>/grasps     raw AnyGrasp output, camera optical frame,
#                           AnyGrasp axes (x approach, y closing, z normal),
#                           position = surface grasp centre (graspnetAPI).
#   /grasp/<obj>/filtered   grasp_filter output, TSR reference frame (base),
#                           e axes (+z approach, +x closing), position at the
#                           pad centre (pad_offset applied).
# Arrays are parallel and index-aligned with `poses`; order is the
# publisher's ranking (AnyGrasp: score desc; filtered: see route).
std_msgs/Header header
geometry_msgs/Pose[] poses
float32[] scores          # AnyGrasp score (grasp score x (1 - stable score))
float32[] widths          # m, gripper opening the proposal was scored at
float32[] depths          # m, fingertip reach past the grasp centre along approach
