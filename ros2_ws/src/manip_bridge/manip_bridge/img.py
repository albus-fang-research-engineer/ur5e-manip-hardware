"""sensor_msgs <-> numpy and pose <-> msg conversions shared by every bridge.

No cv_bridge: it drags in a system OpenCV ABI and the sidecars' contracts are
plain numpy anyway. Step-aware so row-padded images decode correctly.
"""

import numpy as np
from geometry_msgs.msg import PoseStamped, TransformStamped
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image


# -------------------------------------------------------------- msg -> numpy
def image_to_rgb(msg: Image) -> np.ndarray:
    """rgb8/bgr8/rgba8/bgra8 -> HxWx3 uint8 RGB."""
    h, w = msg.height, msg.width
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if msg.encoding in ("rgb8", "bgr8"):
        img = buf.reshape(h, msg.step // 3, 3)[:, :w, :]
        return img[..., ::-1].copy() if msg.encoding == "bgr8" else img.copy()
    if msg.encoding in ("rgba8", "bgra8"):
        img = buf.reshape(h, msg.step // 4, 4)[:, :w, :3]
        return img[..., ::-1].copy() if msg.encoding == "bgra8" else img.copy()
    raise ValueError(f"unsupported rgb encoding: {msg.encoding}")


def image_to_mono(msg: Image) -> np.ndarray:
    """mono8 -> HxW uint8."""
    if msg.encoding != "mono8":
        raise ValueError(f"mask must be mono8, got: {msg.encoding}")
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    return buf.reshape(msg.height, msg.step)[:, : msg.width].copy()


def image_to_depth_m(msg: Image) -> np.ndarray:
    """16UC1 (mm) or 32FC1 (m) -> HxW float32 meters, invalid = 0."""
    h, w = msg.height, msg.width
    if msg.encoding == "16UC1":
        d = np.frombuffer(bytes(msg.data), dtype=np.uint16)
        return d.reshape(h, msg.step // 2)[:, :w].astype(np.float32) * 1e-3
    if msg.encoding == "32FC1":
        d = np.frombuffer(bytes(msg.data), dtype=np.float32)
        d = d.reshape(h, msg.step // 4)[:, :w].copy()
        return np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    raise ValueError(f"unsupported depth encoding: {msg.encoding}")


def camera_info_to_K(msg) -> np.ndarray:
    # .k not .p: raw intrinsics of the (already rectified) RealSense stream.
    return np.asarray(msg.k, dtype=np.float64).reshape(3, 3)


# -------------------------------------------------------------- numpy -> msg
def mono_to_image(mask: np.ndarray, header) -> Image:
    """bool / uint8 HxW -> mono8 Image (255 = foreground)."""
    m = np.ascontiguousarray(mask)
    if m.dtype == bool:
        m = m.astype(np.uint8) * 255
    elif m.max(initial=0) <= 1:
        m = (m * 255).astype(np.uint8)
    msg = Image()
    msg.header = header
    msg.height, msg.width = m.shape
    msg.encoding = "mono8"
    msg.is_bigendian = 0
    msg.step = m.shape[1]
    msg.data = m.tobytes()
    return msg


def T_to_pose(T, header) -> PoseStamped:
    T = np.asarray(T, dtype=np.float64)
    q = Rotation.from_matrix(T[:3, :3]).as_quat()  # xyzw
    msg = PoseStamped()
    msg.header = header
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, T[:3, 3])
    (msg.pose.orientation.x, msg.pose.orientation.y,
     msg.pose.orientation.z, msg.pose.orientation.w) = map(float, q)
    return msg


def T_to_tf(T, header, child) -> TransformStamped:
    T = np.asarray(T, dtype=np.float64)
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    tf = TransformStamped()
    tf.header = header
    tf.child_frame_id = child
    (tf.transform.translation.x, tf.transform.translation.y,
     tf.transform.translation.z) = map(float, T[:3, 3])
    (tf.transform.rotation.x, tf.transform.rotation.y,
     tf.transform.rotation.z, tf.transform.rotation.w) = map(float, q)
    return tf
