"""Record / replay transports for the sim VLM client (vlm_compat.Client).

The client's transport is a plain callable payload -> raw text, so every
hardware VLM driver (plan_scene: call #1, select_scene: call #2) records
each raw response into its log and can feed a log back with no API call.
Shared here so both drivers replay the same way.

This module imports no sim code itself (vlm_compat is the import site), so
it is not a compat module.

Paid-call accounting: every RecordingTransport counts the requests it sends
to the live API. At interpreter exit the process's total is written as JSON
to $VLM_CALLS_FILE when that is set (scripts/vlm_run.sh sets it and appends
the count to outputs/vlm_calls.log), so the log records the calls a driver
actually made, not a guess from its flags. Replay and dry-run transports
never reach the API and count nothing.
"""
from __future__ import annotations

import atexit
import json
import os
import sys

from manip_bridge import vlm_compat

_RECORDERS: list = []


def calls_made() -> int:
    """Requests sent to the live API by this process (all recorders)."""
    return sum(r.sent for r in _RECORDERS)


def write_calls_file(path: str | None = None) -> str | None:
    """Write {"calls", "responses", "argv"} to `path` (default
    $VLM_CALLS_FILE); called at exit. Returns the path written, or None."""
    path = path or os.environ.get("VLM_CALLS_FILE")
    if not path:
        return None
    with open(path, "w") as f:
        json.dump({"calls": calls_made(), "responses": sum(len(r.raw) for r in _RECORDERS),
                   "argv": sys.argv}, f)
        f.write("\n")
    return path


atexit.register(write_calls_file)


class RecordingTransport:
    """Wraps the live transport so every raw response survives into the log
    (Client.logs keeps the accepted text and the rejections, not each raw
    attempt). The live transport is looked up at CALL time, so tests that
    monkeypatch vlm_compat.live_transport reach it."""

    def __init__(self, inner=None):
        self.inner = inner
        self.raw: list[str] = []
        self.sent = 0                     # requests sent, answered or not
        _RECORDERS.append(self)

    def __call__(self, payload: dict) -> str:
        self.sent += 1
        text = (self.inner or vlm_compat.live_transport)(payload)
        self.raw.append(text)
        return text


class ReplayTransport:
    """Returns recorded responses in order; never reaches the network."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.raw: list[str] = []

    def __call__(self, payload: dict) -> str:
        if len(self.raw) >= len(self.responses):
            raise vlm_compat.VLMError(
                f"replay exhausted: the log holds {len(self.responses)} response(s) and the "
                "client asked for another -- the prompt or parser changed since recording")
        text = self.responses[len(self.raw)]
        self.raw.append(text)
        return text
