"""Record / replay transports for the sim VLM client (vlm_compat.Client).

The client's transport is a plain callable payload -> raw text, so every
hardware VLM driver (plan_scene: call #1, select_scene: call #2) records
each raw response into its log and can feed a log back with no API call.
Shared here so both drivers replay the same way.

This module imports no sim code itself (vlm_compat is the import site), so
it is not a compat module.
"""
from manip_bridge import vlm_compat


class RecordingTransport:
    """Wraps the live transport so every raw response survives into the log
    (Client.logs keeps the accepted text and the rejections, not each raw
    attempt). The live transport is looked up at CALL time, so tests that
    monkeypatch vlm_compat.live_transport reach it."""

    def __init__(self, inner=None):
        self.inner = inner
        self.raw: list[str] = []

    def __call__(self, payload: dict) -> str:
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
