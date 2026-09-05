# One parallel-jaw grasp proposal with the scalars a bare Pose drops.
# Frame and axis convention belong to the GraspArray topic this rides on
# (see GraspArray.msg); this type is deliberately convention-free.
geometry_msgs/Pose pose
float32 score             # AnyGrasp score (grasp score x (1 - stable score))
float32 width             # m, gripper opening the proposal was scored at
float32 depth             # m, fingertip reach past the grasp centre along approach
