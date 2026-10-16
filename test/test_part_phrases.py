"""outputs/runs/part_phrases.py -- the segmentation-phrase call per object.

Offline: the real sim client and ask loop (vlm_compat, so parse rejections
are fed back exactly as in calls #1-#3) with CANNED model responses, and a
stub renderer in place of render_compat (no GL). Needs the sim checkout at
$SIM_DIR or the sibling ../ur5e-manip-sim; skips without it.
"""
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE = REPO_ROOT / "ros2_ws" / "src" / "manip_bridge"
SIM_DIR = Path(os.environ.get("SIM_DIR", REPO_ROOT.parent / "ur5e-manip-sim")).resolve()
HAVE_SIM = (SIM_DIR / "manip_sim" / "vlm.py").is_file()
pytestmark = pytest.mark.skipif(not HAVE_SIM, reason=f"no sim checkout at {SIM_DIR} (set SIM_DIR)")
PIL_Image = pytest.importorskip("PIL.Image")

VNAMES = ["x+", "x-", "top"]
RC = "manip_bridge.render_compat"


def _stub_render():
    rc = types.ModuleType(RC)
    rc.VIEW_PX = 32
    rc.load_obj = lambda path: (np.zeros((3, 3)), np.array([[0, 1, 2]]))

    def render_depth_views(name, obj_dir, V, out_views, px=32):
        out_views = Path(out_views)
        out_views.mkdir(parents=True, exist_ok=True)
        paths = {}
        for v in VNAMES:
            p = out_views / f"{v.replace('+', 'p').replace('-', 'n')}.png"
            PIL_Image.fromarray(np.full((px, px, 3), 200, np.uint8)).save(p)
            paths[v] = p
        return {v: {} for v in VNAMES}, (lambda P: ({}, {})), paths

    rc.render_depth_views = render_depth_views
    rc.provenance = lambda: {"commit": "stub-render"}
    return rc


@pytest.fixture
def drv():
    saved = sys.modules.get(RC)
    sys.path.insert(0, str(BRIDGE))
    sys.path.insert(0, str(SIM_DIR))
    sys.modules[RC] = _stub_render()
    spec = importlib.util.spec_from_file_location(
        "part_phrases", REPO_ROOT / "outputs" / "runs" / "part_phrases.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    yield m
    if saved is None:
        sys.modules.pop(RC, None)
    else:
        sys.modules[RC] = saved
    sys.path.remove(str(SIM_DIR))
    sys.path.remove(str(BRIDGE))


class Canned:
    def __init__(self, *responses):
        self.responses, self.payloads = list(responses), []

    def __call__(self, payload):
        self.payloads.append(payload)
        if len(self.payloads) > len(self.responses):
            raise AssertionError("more calls than canned responses")
        r = self.responses[len(self.payloads) - 1]
        if isinstance(r, Exception):
            raise r
        return r


def _asset(tmp):
    a = tmp / "asset"
    (a / "meshes").mkdir(parents=True)
    (a / "meshes" / "obj_visual.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    return a


def _ph(**kv):
    return json.dumps({"phrases": kv})


def _run(drv, tmp, transport, parts=("barrel", "tip"), replay=None):
    out = tmp / "masks" / "obj" / "phrases.json"
    code, doc = drv.part_phrases(_asset(tmp), "obj", list(parts), out,
                                 transport=transport, replay=replay)
    return code, doc, out


def test_inputs_are_parts_and_views_only(drv, tmp_path):
    t = Canned(_ph(barrel="white tube", tip="black cap"))
    code, doc, out = _run(drv, tmp_path, t)
    assert code == 0 and doc["status"] == "ok"
    assert doc["phrases"] == {"barrel": {"prompt": "white tube", "source": "vlm"},
                              "tip": {"prompt": "black cap", "source": "vlm"}}
    content = t.payloads[0]["messages"][0]["content"]
    texts = [b["text"] for b in content if b["type"] == "text"]
    assert texts == ["Parts: barrel, tip"]                  # no task text, no stage plan
    assert sum(b["type"] == "image" for b in content) == len(VNAMES)   # every canonical view
    assert len(doc["views"]) == len(VNAMES) and doc["views_px"] == 32
    assert json.loads(out.read_text()) == json.loads(json.dumps(doc))
    log = json.loads(out.with_suffix(".log.json").read_text())
    assert log["raw_responses"] == [t.responses[0]] and log["parts"] == ["barrel", "tip"]


def test_prompt_carries_no_example_from_an_evaluated_object(drv):
    """The answers measured on the insert-marker assets must not appear in
    the prompt that is then measured on them."""
    for leaked in ("black cap", "white tube", "inner wall", "marker", "cup", "mug", "teapot"):
        assert leaked not in drv.SYSTEM.lower()


def test_missing_part_falls_back_to_its_name(drv, tmp_path):
    code, doc, _ = _run(drv, tmp_path, Canned(_ph(barrel="white tube")))
    assert code == 0
    assert doc["phrases"]["tip"] == {"prompt": "tip", "source": "bare"}


@pytest.mark.parametrize("bad, why", [
    (_ph(barrel="white, tube"), "contains [','"),
    (_ph(barrel="tube=white"), "contains ['='"),
    (_ph(handle="red grip"), "not a listed part"),
    (json.dumps({"barrel": "white tube"}), 'missing object "phrases"'),
])
def test_rejections_are_fed_back_and_retried(drv, tmp_path, bad, why):
    t = Canned(bad, _ph(barrel="white tube", tip="black cap"))
    code, doc, out = _run(drv, tmp_path, t)
    assert code == 0 and doc["phrases"]["barrel"]["prompt"] == "white tube"
    assert len(t.payloads) == 2
    retry_turn = t.payloads[1]["messages"][-1]["content"][0]["text"]
    assert why in retry_turn
    log = json.loads(out.with_suffix(".log.json").read_text())
    assert any(why in r for r in log["client_logs"][0]["rejections"])


def test_duplicate_phrase_later_part_takes_its_name(drv, tmp_path):
    code, doc, _ = _run(drv, tmp_path, Canned(_ph(cap="Black Cap", tip="black cap")),
                        parts=("cap", "tip"))
    assert doc["phrases"]["cap"] == {"prompt": "Black Cap", "source": "vlm"}
    assert doc["phrases"]["tip"]["source"] == "bare_duplicate"
    assert doc["phrases"]["tip"]["prompt"] == "tip" and "'cap'" in doc["phrases"]["tip"]["note"]


def test_call_failure_exits_2_with_bare_names_written(drv, tmp_path):
    from manip_bridge import vlm_compat
    code, doc, out = _run(drv, tmp_path, Canned(vlm_compat.VLMError("HTTP 529 overloaded")))
    assert code == 2 and doc["status"] == "failed" and "529" in doc["error"]
    assert all(r["source"] == "bare" and r["prompt"] == p for p, r in doc["phrases"].items())
    assert json.loads(out.read_text())["status"] == "failed"     # still written, for the operator


def test_retry_budget_exhausted_is_a_failure(drv, tmp_path):
    bad = _ph(barrel="a,b")
    code, doc, _ = _run(drv, tmp_path, Canned(bad, bad, bad))
    assert code == 2 and doc["status"] == "failed"


def test_replay_is_offline_and_identical(drv, tmp_path, monkeypatch):
    t = Canned(_ph(barrel="white tube", tip="black cap"))
    _, live, out = _run(drv, tmp_path / "live", t)
    log = out.with_suffix(".log.json")
    from manip_bridge import vlm_compat

    def boom(payload):
        raise AssertionError("replay reached the live transport")
    monkeypatch.setattr(vlm_compat, "live_transport", boom)
    code, rep, _ = _run(drv, tmp_path / "rep", None, replay=str(log))
    assert code == 0 and rep["phrases"] == live["phrases"]
    assert rep["replayed_from"] == str(log.resolve())


def test_refusals(drv, tmp_path):
    with pytest.raises(drv.PhraseRefusal, match="empty"):
        drv.part_phrases(_asset(tmp_path), "obj", [], tmp_path / "p.json", transport=Canned())
    with pytest.raises(drv.PhraseRefusal, match="no .*obj_visual"):
        drv.part_phrases(tmp_path / "nowhere", "obj", ["a"], tmp_path / "p.json",
                         transport=Canned())
