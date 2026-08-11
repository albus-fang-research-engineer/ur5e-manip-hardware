"""SAM 3 sidecar tests: open-vocab segmentation on the sim frame packet,
scored against the renderer's ground-truth instance masks.

The packet masks are occlusion-aware (rendered element segmentation), so IoU
against SAM 3 output is a fair comparison. Thresholds are deliberately loose:
this is a "the deployment path works and points at the right pixels" test,
not a benchmark of SAM 3.
"""

import numpy as np
import pytest

from conftest import iou

IOU_MIN = 0.45  # per-object floor; sim textures are bland, keep it honest but loose


def _best_mask_iou(rep, gt):
    """Best IoU over returned instances (SAM may split/duplicate)."""
    masks = np.asarray(rep["masks"])
    if masks.size == 0:
        return 0.0, None
    scores = [iou(m, gt) for m in masks]
    return max(scores), int(np.argmax(scores))


def test_ping(sam3):
    assert sam3.ok({"cmd": "ping"})["ok"]


def test_reply_schema_and_score_order(sam3, packet):
    rep = sam3.ok({"cmd": "segment", "rgb": packet["rgb"],
                   "prompt": packet.objects[0]})
    masks = np.asarray(rep["masks"])
    boxes = np.asarray(rep["boxes"])
    scores = np.asarray(rep["scores"])
    h, w = packet["rgb"].shape[:2]

    assert masks.ndim == 3 and masks.shape[1:] == (h, w)
    assert masks.dtype == bool
    assert boxes.shape == (masks.shape[0], 4)
    assert scores.shape == (masks.shape[0],)
    # server contract: sorted by score desc
    assert np.all(np.diff(scores) <= 1e-6)
    # boxes are xyxy pixels inside the image
    if len(boxes):
        assert np.all(boxes[:, 0] <= boxes[:, 2]) and np.all(boxes[:, 1] <= boxes[:, 3])
        assert boxes.min() >= -1 and boxes[:, 2].max() <= w + 1 and boxes[:, 3].max() <= h + 1


@pytest.mark.parametrize("obj", ["teapot", "mug"])
def test_open_vocab_matches_gt_mask(sam3, packet, obj):
    if obj not in packet.objects:
        pytest.skip(f"{obj} not in packet")
    gt = packet[f"mask_{obj}"]
    rep = sam3.ok({"cmd": "segment", "rgb": packet["rgb"], "prompt": obj})
    best, idx = _best_mask_iou(rep, gt)
    assert best >= IOU_MIN, (
        f"best IoU {best:.3f} < {IOU_MIN} for prompt '{obj}' "
        f"({len(np.asarray(rep['masks']))} instances returned)")
    # box of the best instance should contain the GT centroid
    vs, us = np.nonzero(gt)
    cx, cy = us.mean(), vs.mean()
    x0, y0, x1, y1 = np.asarray(rep["boxes"])[idx]
    assert x0 - 5 <= cx <= x1 + 5 and y0 - 5 <= cy <= y1 + 5


def test_absent_concept_returns_empty_not_error(sam3, packet):
    # SAM 3's presence token suppresses absent concepts; N == 0 is a valid
    # answer per the server docstring — this is the regression guard for it.
    rep = sam3.ok({"cmd": "segment", "rgb": packet["rgb"], "prompt": "zebra"})
    assert np.asarray(rep["masks"]).shape[0] <= 1  # ideally 0; never an error
    if np.asarray(rep["masks"]).shape[0] == 1:
        # if something fired, it must be low-confidence junk, not a real hit
        assert float(np.asarray(rep["scores"])[0]) < 0.9


def test_segment_multi_amortizes_consistently(sam3, packet):
    """segment_multi shares one image encode across prompts; per-prompt output
    must match the single-prompt path (deterministic model, same threshold)."""
    prompts = list(packet.objects)
    multi = sam3.ok({"cmd": "segment_multi", "rgb": packet["rgb"],
                     "prompts": prompts})
    assert len(multi["results"]) == len(prompts)
    for p, res in zip(prompts, multi["results"]):
        single = sam3.ok({"cmd": "segment", "rgb": packet["rgb"], "prompt": p})
        sm, mm = np.asarray(single["masks"]), np.asarray(res["masks"])
        assert sm.shape == mm.shape, f"prompt '{p}': {sm.shape} vs {mm.shape}"
        if sm.size:
            assert iou(sm[0], mm[0]) > 0.98, f"prompt '{p}' diverges between paths"


def test_threshold_monotonicity(sam3, packet):
    obj = packet.objects[0]
    lo = sam3.ok({"cmd": "segment", "rgb": packet["rgb"], "prompt": obj,
                  "threshold": 0.2})
    hi = sam3.ok({"cmd": "segment", "rgb": packet["rgb"], "prompt": obj,
                  "threshold": 0.8})
    assert np.asarray(hi["masks"]).shape[0] <= np.asarray(lo["masks"]).shape[0]


def test_mask_is_pose_bridge_ready(sam3, packet):
    """masks[0] is what gets published on /pose_bridge/mask for FoundationPose
    registration — it must be a clean single-instance foreground blob."""
    obj = packet.objects[0]
    rep = sam3.ok({"cmd": "segment", "rgb": packet["rgb"], "prompt": obj})
    masks = np.asarray(rep["masks"])
    assert masks.shape[0] >= 1, f"no instance for '{obj}'"
    m = masks[0]
    frac = m.mean()
    assert 0.001 < frac < 0.5, f"top mask covers {frac:.1%} of image — implausible"
