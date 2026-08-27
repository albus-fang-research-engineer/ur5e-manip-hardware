"""Minimal msgpack-numpy REQ client with rebuild-on-timeout.

Same semantics as test/conftest.py::ZmqService and the manip_sim clients.
One instance per *concurrent call site*: a REQ socket strictly alternates
send/recv, so a service callback and a streaming subscriber in the same
node must each own their own SidecarClient, not share one.
"""

import pickle
import threading

import msgpack
import msgpack_numpy
import zmq

msgpack_numpy.patch()


class SidecarError(RuntimeError):
    pass


class SidecarClient:
    def __init__(self, addr: str, timeout_ms: int, codec: str = "msgpack"):
        """codec: "msgpack" (every sidecar) or "pickle" (grasp_server only --
        it predates the msgpack contract). Same socket semantics either way."""
        if codec == "msgpack":
            self._enc = lambda o: msgpack.packb(o, use_bin_type=True)
            self._dec = lambda b: msgpack.unpackb(b, raw=False)
        elif codec == "pickle":
            self._enc, self._dec = pickle.dumps, pickle.loads
        else:
            raise ValueError(f"unknown codec {codec!r}")
        self.addr = addr
        self.timeout_ms = int(timeout_ms)
        self._ctx = zmq.Context.instance()
        self._sock = None
        self._lock = threading.Lock()
        self._connect()

    def _connect(self):
        if self._sock is not None:
            self._sock.close(linger=0)
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, 5000)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(self.addr)

    def call(self, payload: dict, timeout_ms: int | None = None) -> dict:
        """Send, wait for reply. Raises TimeoutError (socket rebuilt) or
        SidecarError (server answered {"ok": False})."""
        with self._lock:
            if timeout_ms is not None:
                self._sock.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
            try:
                self._sock.send(self._enc(payload))
                rep = self._dec(self._sock.recv())
            except zmq.error.Again:
                self._connect()
                name = payload.get("cmd", payload.get("op"))
                raise TimeoutError(f"{self.addr}: no reply to {name!r}")
            finally:
                if timeout_ms is not None:
                    self._sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        if not rep.get("ok", False):
            raise SidecarError(rep.get("error", "unknown sidecar error"))
        return rep

    def ping(self, cmd_key="cmd", timeout_ms=3000) -> bool:
        try:
            self.call({cmd_key: "ping"}, timeout_ms=timeout_ms)
            return True
        except (TimeoutError, SidecarError):
            return False
