"""Shared fixtures for the sidecar test suite.

Philosophy: the sidecars are exercised over their real interface — msgpack-ZMQ
on localhost — with a sim-captured RGB-D frame packet standing in for the
RealSense. The packet carries ground truth (masks, poses, joint state), so
these are integration tests with quantitative checks, not smoke pings.

Setup:
  1. In ur5e-manip-sim:
       MUJOCO_GL=egl PYTHONPATH=. python scripts/capture_rgbd_packet.py \
           --out outputs/frame_packets/packet
  2. Copy the OUTPUT DIR (packet.npz + *.obj + *_frames.json) to
       ur5e-manip-hardware/test/data/packet/
     or export FRAME_PACKET=/path/to/packet.npz
  3. Bring up the sidecars you want to test:
       docker compose up -d sam3 pose trellis2 curobo
  4. From the repo root:  python -m pytest test/ -v
     (skip slow TRELLIS generation with:  -m "not slow")

Tests for a sidecar that isn't running SKIP (with the compose command in the
reason) rather than fail, so partial stacks are fine.

The cuRobo library tests (self-masking, mesh attachment, sphere fitting) live
in test/curobo_incontainer/ and import curobo directly, so they run inside the
curobo container — see that directory's docstrings.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent

ADDRS = {
    "sam3":    os.environ.get("SAM3_ADDR",    "tcp://127.0.0.1:5670"),
    "pose":    os.environ.get("POSE_ADDR",    "tcp://127.0.0.1:5667"),
    "trellis2": os.environ.get("TRELLIS_ADDR", "tcp://127.0.0.1:5669"),
    "curobo":  os.environ.get("CUROBO_ADDR",  "tcp://127.0.0.1:5671"),
    "any6d":   os.environ.get("ANY6D_ADDR",   "tcp://127.0.0.1:5672"),
}
COMPOSE_HINT = {
    "sam3": "docker compose up -d sam3",
    "pose": "docker compose up -d pose",
    "trellis2": "docker compose up -d trellis2",
    "curobo": "docker compose up -d curobo",
    "any6d": "docker compose up -d any6d",
}


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: multi-minute generative tests")


# ------------------------------------------------------------------ ZMQ client
class ZmqService:
    """Minimal REQ client, msgpack + msgpack_numpy, rebuild-on-timeout —
    identical semantics to the manip_sim clients and the ROS bridge nodes."""

    def __init__(self, addr, timeout_ms=60_000):
        import msgpack_numpy
        msgpack_numpy.patch()
        self.addr = addr
        self.timeout_ms = timeout_ms
        self._connect()

    def _connect(self):
        import zmq
        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self.addr)

    def call(self, payload, timeout_ms=None):
        import msgpack
        import zmq
        if timeout_ms is not None:
            self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        try:
            self.sock.send(msgpack.packb(payload, use_bin_type=True))
            rep = msgpack.unpackb(self.sock.recv(), raw=False)
        except zmq.error.Again:
            self._connect()
            raise TimeoutError(f"no reply from {self.addr} for {payload.get('cmd', payload.get('op'))}")
        finally:
            if timeout_ms is not None:
                self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        return rep

    def ok(self, payload, timeout_ms=None):
        rep = self.call(payload, timeout_ms=timeout_ms)
        assert rep.get("ok"), f"{self.addr} error: {rep.get('error')}"
        return rep

    def err(self, payload, timeout_ms=None):
        """Send a request that MUST fail server-side; return the error reply.
        The sidecars' REP loops catch handler exceptions and answer
        {"ok": False, "error": ...}, so an err() call still gets a reply —
        a timeout here means the server crashed, which is its own failure."""
        rep = self.call(payload, timeout_ms=timeout_ms)
        assert not rep.get("ok"), \
            f"{self.addr}: expected error reply for {payload.get('cmd', payload.get('op'))}, got {rep.keys() if isinstance(rep, dict) else rep}"
        assert rep.get("error"), f"{self.addr}: error reply without error message: {rep}"
        return rep


def _service(name, ping):
    addr = ADDRS[name]
    svc = ZmqService(addr)
    try:
        rep = svc.call(ping, timeout_ms=5_000)
    except TimeoutError:
        pytest.skip(f"{name} sidecar not responding at {addr} — {COMPOSE_HINT[name]}")
    if not rep.get("ok"):
        pytest.skip(f"{name} sidecar unhealthy at {addr}: {rep.get('error')}")
    return svc


@pytest.fixture(scope="session")
def sam3():
    return _service("sam3", {"cmd": "ping"})


@pytest.fixture(scope="session")
def pose():
    return _service("pose", {"cmd": "ping"})


@pytest.fixture(scope="session")
def trellis2():
    return _service("trellis2", {"op": "ping"})


@pytest.fixture(scope="session")
def curobo():
    return _service("curobo", {"cmd": "ping"})


@pytest.fixture(scope="session")
def any6d():
    return _service("any6d", {"cmd": "ping"})


# --------------------------------------------------------------- frame packet
class Packet:
    def __init__(self, npz_path: Path):
        self.path = npz_path
        self.dir = npz_path.parent
        data = np.load(npz_path, allow_pickle=False)
        self.arrays = {k: data[k] for k in data.files if k != "meta"}
        self.meta = json.loads(str(data["meta"]))

    def __getitem__(self, k):
        return self.arrays[k]

    @property
    def objects(self):
        return self.meta["objects"]

    def mesh_path(self, name):
        p = self.dir / f"{name}.obj"
        return p if p.exists() else None


def _find_packet() -> Path:
    env = os.environ.get("FRAME_PACKET")
    if env:
        p = Path(env)
        if p.is_file():
            return p
        pytest.skip(f"FRAME_PACKET={env} not found")
    default = TEST_DIR / "data" / "packet" / "packet.npz"
    if default.is_file():
        return default
    pytest.skip(
        "no frame packet: capture one in ur5e-manip-sim with "
        "scripts/capture_rgbd_packet.py and copy its output dir to "
        f"{default.parent} (or set FRAME_PACKET)")


@pytest.fixture(scope="session")
def packet() -> Packet:
    return Packet(_find_packet())


# ------------------------------------------------------------------- geometry
def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


def rot_err_deg(Ra, Rb) -> float:
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))