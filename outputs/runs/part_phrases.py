#!/usr/bin/env python3
"""Segmentation phrases for an object's parts: one VLM call per object that
turns the planner's part names (function: "barrel", "interior") into the
words a text-prompted segmenter answers to (appearance: a thing noun, with a
visible attribute when that distinguishes it).

    <asset_dir> (render asset) + part names
            |  the eight canonical views (render_compat.render_depth_views --
            |  the SAME renderer and cameras make_part_masks segments, so the
            |  model chooses each phrase against the images SAM3 will get)
            v  one VLM call (vlm_compat.ask: the client's ask loop + retry)
    <out>             phrases.json  {part: {prompt, source}} + provenance
    <out>.log.json    raw model responses (replayable, no API call)

then  make_part_masks.py ... --phrases <out>  sends SAM3 the phrase for each
part (and falls back to the bare part name per view when the phrase finds
nothing). Masks, grounding, the contract and call #2 keep the part NAMES;
the phrase is only ever a segmenter prompt.

Why this is its own call (2026-10-15): SAM3 is an object-concept segmenter.
On the insert-marker assets it scored function words low (marker `body` <=
0.22, `barrel` <= 0.31 at any view) and appearance phrases high, and the
masks from appearance phrases grounded to the authored frames (shaft axis
1.4 deg, rim radius 0.6 mm). Call #1 is left untouched: its prompt is part
of what the planning ablations measure, and it sees the scene, not the
object close up.

Input is the part list and the views, NOTHING else -- no task text, no
stage plan -- so this stays a perception stage, not a second planner.

Outcomes per part (phrases.json `phrases[part].source`):
  vlm             the model's phrase
  bare            the model gave none for this part -> the part name itself
  bare_duplicate  the model gave a phrase an EARLIER part already has; two
                  parts with one prompt get one mask, and one would lose
                  every vote in grounding. The later part (in --parts
                  order) falls back to its bare name, recorded here rather
                  than discovered two stages later. Two names for one
                  region is usually the planner naming it twice, not an
                  error.
A phrase containing ',' or '=' (it would break --parts syntax) or a key that
is not a listed part is a parse rejection: fed back, bounded retry.

Exit: 0 phrases written; 2 the call failed outright (transport error or
retry budget exhausted) -- phrases.json is STILL written with every part at
its bare name and status "failed", so the operator can choose to continue
on bare names (today's behaviour) knowing that is what they get.

In the ros2 container (ANTHROPIC_API_KEY set; MANIP_VLM_MODEL optional):

    export PYTHONPATH=/root/ros2_ws/src/manip_bridge:$PYTHONPATH
    python3 /data/runs/part_phrases.py /data/runs/<stamp>/assets_canon/mug --name mug \\
        --parts handle,rim --out /data/runs/<stamp>/masks_vlm_<sha>/mug/phrases.json
    python3 /data/runs/part_phrases.py ... --replay <out>.log.json      # no API call
"""
import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

try:
    from manip_bridge import vlm_compat
    from manip_bridge.render_compat import VIEW_PX, load_obj, render_depth_views
    from manip_bridge.render_compat import provenance as render_provenance
    from manip_bridge.vlm_record import RecordingTransport, ReplayTransport
except ImportError as e:
    sys.exit(f"part_phrases: cannot import manip_bridge ({e}). In the ros2 container:\n"
             "    export PYTHONPATH=/root/ros2_ws/src/manip_bridge:$PYTHONPATH")

TOUCHPOINT = "part_phrases"
FORBIDDEN = (",", "=")

# Examples are deliberately from objects in NO evaluation task, so the
# prompt never carries the answer for an object it is measured on.
SYSTEM = (
    "You are the part-naming module for a text-prompted image segmenter. "
    "The images are renders of ONE object, alone, from several viewpoints. "
    "For each listed part, give the 1-3 word phrase the segmenter would "
    "recognise on these images: a thing noun, with a visible attribute if "
    "that distinguishes it (e.g. \"metal blade\", \"rubber grip\"). "
    "Output schema: {\"phrases\": {\"<part>\": \"<phrase>\", ...}}, keyed by "
    "exactly the listed part names. ")


class PhraseRefusal(Exception):
    """An input the driver will not run on; the message names the fix."""


def build_prompt(parts: list[str], view_paths: list[Path]) -> tuple[str, list]:
    content = [vlm_compat.text_block("Parts: " + ", ".join(parts))]
    content += [vlm_compat.image_block(p) for p in view_paths]
    return SYSTEM + vlm_compat.JSON_ONLY, [{"role": "user", "content": content}]


def parse(raw: str, parts: list[str]) -> dict[str, str]:
    """{part: phrase} for the parts the model answered; rejections name the
    offending key/value and the accept set."""
    doc = vlm_compat.load_json(raw)
    ph = doc.get("phrases") if isinstance(doc, dict) else None
    if not isinstance(ph, dict):
        raise vlm_compat.ParseRejection('missing object "phrases": {"<part>": "<phrase>"}')
    out = {}
    for k, v in ph.items():
        if k not in parts:
            raise vlm_compat.ParseRejection(
                f"phrases key {k!r} is not a listed part; listed: {parts}")
        if not isinstance(v, str) or not v.strip():
            raise vlm_compat.ParseRejection(f"phrases[{k!r}] must be a non-empty string")
        bad = [c for c in FORBIDDEN if c in v]
        if bad:
            raise vlm_compat.ParseRejection(
                f"phrases[{k!r}] = {v!r} contains {bad}; a phrase may not contain "
                f"{' or '.join(repr(c) for c in FORBIDDEN)}")
        out[k] = " ".join(v.split())
    return out


def resolve(parts: list[str], answered: dict[str, str]) -> dict[str, dict]:
    """Per part, the prompt and where it came from (module docstring)."""
    res, used = {}, {}
    for p in parts:
        ph = answered.get(p)
        if ph is None:
            res[p] = {"prompt": p, "source": "bare"}
        elif ph.casefold() in used:
            res[p] = {"prompt": p, "source": "bare_duplicate",
                      "note": f"phrase {ph!r} already given to {used[ph.casefold()]!r}"}
        else:
            res[p] = {"prompt": ph, "source": "vlm"}
            used[ph.casefold()] = p
    return res


def _asset_record(asset_dir: Path):
    ra = asset_dir / "render_asset.json"
    if not ra.is_file():
        return None
    d = json.loads(ra.read_text())
    return {"obj_sha256": d.get("obj_sha256"),
            "rotated": d.get("rotation_body_from_canon") is not None}


def part_phrases(asset_dir, name: str, parts: list[str], out, views_out=None, px=VIEW_PX,
                 replay: str | None = None, transport=None) -> tuple[int, dict]:
    asset_dir, out = Path(asset_dir), Path(out)
    if not parts:
        raise PhraseRefusal("--parts is empty: nothing to phrase")
    bad = [p for p in parts if any(c in p for c in FORBIDDEN)]
    if bad:
        raise PhraseRefusal(f"part names {bad} contain ',' or '=' and cannot be passed as --parts")
    mesh = asset_dir / "meshes" / f"{name}_visual.obj"
    if not mesh.exists():
        raise PhraseRefusal(f"no {mesh} -- asset_dir must be a render asset for --name {name}")
    views_out = Path(views_out) if views_out else out.parent / "_views"
    V, _F = load_obj(mesh)
    _cams, _proj, view_paths = render_depth_views(name, asset_dir, V, out_views=views_out, px=px)
    views = [Path(view_paths[v]) for v in sorted(view_paths)]

    replayed_from, model = None, vlm_compat.MODEL
    if replay:
        try:
            log = json.loads(Path(replay).read_text())
        except (OSError, ValueError) as e:
            raise PhraseRefusal(f"--replay {replay}: unreadable ({e})") from e
        if not log.get("raw_responses"):
            raise PhraseRefusal(f"--replay {replay}: no raw_responses recorded")
        rec, model = ReplayTransport(log["raw_responses"]), log.get("model") or model
        replayed_from = str(Path(replay).resolve())
    else:
        rec = RecordingTransport(transport)
    client = vlm_compat.Client(transport=rec, model=model)
    system, messages = build_prompt(parts, views)
    status, error = "ok", None
    try:
        answered = vlm_compat.ask(client, TOUCHPOINT, system, messages,
                                  lambda raw: parse(raw, parts))
    except vlm_compat.VLMError as e:
        status, error, answered = "failed", str(e), {}
    phrases = resolve(parts, answered)

    log_path = out.with_suffix(".log.json")
    doc = {"name": name, "parts": parts, "phrases": phrases, "status": status, "error": error,
           "model": client.model, "views": [str(v) for v in views], "views_px": px,
           "asset_dir": str(asset_dir.resolve()), "asset": _asset_record(asset_dir),
           "log": str(log_path.resolve()), "replayed_from": replayed_from,
           "prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
           "sim": vlm_compat.provenance(), "render_sim": render_provenance()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    log_path.write_text(json.dumps({
        "model": client.model, "name": name, "parts": parts,
        "client_logs": [asdict(log) for log in client.logs],
        "raw_responses": rec.raw}, indent=2) + "\n")
    return (2 if status == "failed" else 0), doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("asset_dir", help="render asset dir (<name>.xml + meshes/)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--parts", required=True, help="comma-separated part names")
    ap.add_argument("--out", required=True, help="phrases.json to write (log beside it)")
    ap.add_argument("--views-out", default=None, help="default <out dir>/_views")
    ap.add_argument("--px", type=int, default=VIEW_PX)
    ap.add_argument("--replay", default=None, help="a <out>.log.json to replay (no API call)")
    a = ap.parse_args(argv)
    parts = [p.strip() for p in a.parts.split(",") if p.strip()]
    try:
        code, doc = part_phrases(a.asset_dir, a.name, parts, a.out, a.views_out, a.px, a.replay)
    except PhraseRefusal as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    for p, r in doc["phrases"].items():
        print(f"  {p:16s} -> {r['prompt']!r:28s} ({r['source']}{'; ' + r['note'] if 'note' in r else ''})")
    print(f"[phrases] {a.out}  model {doc['model']}  views {len(doc['views'])}")
    if code:
        print(f"FAILED: the phrase call did not return ({doc['error']}); every part is at its "
              "bare name in the file. make_part_masks --phrases will run on bare names "
              "(today's behaviour) -- continue only if that is what you want.", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
