# Ranked parallel-jaw grasp proposals. A struct array rather than parallel
# pose/score/width/depth arrays so a publisher cannot misalign them.
#
# Two conventions ride on this one type; which one applies is the
# publishing topic's contract:
#   /grasp/<obj>/grasps     raw AnyGrasp output, camera optical frame,
#                           AnyGrasp axes (x approach, y closing, z normal),
#                           position = surface grasp centre (graspnetAPI).
#   /grasp/<obj>/filtered   grasp_filter output, TSR reference frame (base),
#                           e axes (+z approach, +x closing), position at the
#                           pad centre (pad_offset applied).
# Order is the publisher's ranking: AnyGrasp = score desc (grasp_server does
# nms().sort_by_score() before replying); filtered = per its route.
std_msgs/Header header
Grasp[] grasps
