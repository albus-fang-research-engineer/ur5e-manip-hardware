# A Task Space Region (Berenson et al., IJRR 2011) on the wire. The math
# lives in the manip-tsr library (github.com/albus-fang-research-engineer/
# manip-tsr); this is only how a TSR crosses a node boundary. Consumers
# rebuild manip_tsr.TSR(T0_w, Bw, Tw_e, name) from it and evaluate
# displacement / containment / distance locally -- never a service per pose.
#
# header.frame_id  the reference frame T0_w is expressed in (base_link for
#                  everything downstream of the eye-to-base calibration).
#                  Poses evaluated against this TSR must be in that frame.
# header.stamp     when T0_w was frozen (stage entry). A TSR is a snapshot,
#                  not a tracked quantity: the producer publishes it latched
#                  once per stage and re-publishes on stage re-entry.
#
# All matrices row-major, metres and radians.
std_msgs/Header header
float64[16] t0_w      # 4x4 world pose of the reference frame w
float64[16] tw_e      # 4x4 offset w -> constrained frame e at zero displacement
float64[12] bw        # 6x2 bounds [lo, hi] over (x, y, z, roll, pitch, yaw) in w
string name           # e.g. "grasp/handle", "transport/path" -- appears in reports
