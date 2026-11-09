#!/usr/bin/env python3
"""Three-case check of the critic (VLM call #4, critic v2) on hardware run
20261105_035449: one critic call per case on the specification as given,
verdict only (no edit applied, no second round, no planning).

    case            specification                                   expected
    handle_anchor   11/05 as recorded: call #2 anchored the carry   reject
                    on teapot.handle_center (handle over the mug)
    spout_repick    the same, after the round-1 re-pick             accept
                    reselect_point teapot.spout_center (hand-built)
    spout_in_mug    spout anchor, but call #3's carry subgoal       reject
                    hand-edited from "above" to "below"
                    mug.opening_center: the spout goes down into
                    the mug, so the teapot body intersects it

All three use call #3's sim stand-in emissions (call3_simreplay), the
run's measured poses and the taught grasp. The two hand-built cases live in
<run>/critic_cases/<case>/ (selections.json, selections.log.json,
emissions.json, README.md), written by `build` from the run's own inputs
with real tokens only; nothing else in them differs from the run.

    python3 critic_cases.py build --run /data/runs/20261105_035449
    python3 critic_cases.py run --run /data/runs/20261105_035449            # prints the count, stops
    python3 critic_cases.py run --run /data/runs/20261105_035449 --dry-run  # requests, no call
    scripts/vlm_run.sh critic_cases.py run --run /data/runs/20261105_035449 --live   # PAID: 3 calls

Each case runs plan_stages_hw.py --method vlm --critic preview
--critic-verdict-only into <out>/<case>/ and keeps there: the request
(critic/round_1/prompt.json: system text, then the content blocks in order,
images by path; prompt.txt; images/), the raw response
(critic/round_1/raw_response.txt, and critic/responses/ for a paid run) and
the parsed verdict (critic/round_1/verdict.json). <out>/summary.json and
summary.md tabulate case, expected and actual verdict, edits, and the
critic's expected / observed text per stage, plus the request sizes.

Without --live, --dry-run or --responses nothing runs: the paid call count
is printed and the script stops. Exit 1 when any case errors (did not reach
its verdict, or its request in a dry run); the summary is written either
way. A verdict that differs from the expected one is a result, recorded as
a mismatch, and does not change the exit code. --responses DIR replays DIR/<case>/critic_*
files (tests). Default --out: <run>/critic_cases/results/<UTC stamp>_<mode>.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from manip_bridge import vlm_compat
except ImportError as e:
    sys.exit(f"critic_cases: cannot import manip_bridge ({e}). In the ros2 container:\n"
             "    export PYTHONPATH=/root/ros2_ws/src/manip_bridge:$PYTHONPATH")

HERE = Path(__file__).resolve().parent
EMISSIONS = "call3_simreplay/emissions.json"
SELECTIONS = "call2/selections.json"
CARRY_ROLE = "transport_active"
SPOUT = "spout_center"


@dataclass(frozen=True)
class Case:
    name: str
    expected: str
    selections: str          # relative to the run
    emissions: str
    about: str


CASES = (
    Case("handle_anchor", "reject", SELECTIONS, EMISSIONS,
         "11/05 as recorded: call #2 anchored the carry on teapot.handle_center"),
    Case("spout_repick", "accept", "critic_cases/spout_repick/selections.json",
         "critic_cases/spout_repick/emissions.json",
         "11/05 after the round-1 re-pick (reselect_point teapot.spout_center); hand-built"),
    Case("spout_in_mug", "reject", "critic_cases/spout_in_mug/selections.json",
         "critic_cases/spout_in_mug/emissions.json",
         "spout anchor, carry subgoal 'below' mug.opening_center instead of 'above': the spout "
         "goes into the mug and the body intersects it; hand-built"),
)


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _w(p: Path, doc) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2) + "\n")


# ------------------------------------------------------------------ build the hand-built cases

def _spout_candidate(sel_log: dict) -> int:
    """The teapot candidate whose call #2 menu tag names spout_center."""
    menu = (sel_log.get("menus") or {}).get("teapot") or {}
    ids = [int(k) for k, tag in menu.items() if f" {SPOUT} " in f" {tag} "]
    if not ids:
        raise SystemExit(f"[cases] no teapot candidate tagged {SPOUT} in call #2's menu: {menu}")
    return min(ids)


def _write_case(run: Path, name: str, sels: dict, sel_log: dict, em_doc: dict, readme: str,
                edit: dict) -> Path:
    d = run / "critic_cases" / name
    d.mkdir(parents=True, exist_ok=True)
    sel_path = d / "selections.json"
    sel_path.write_text(json.dumps(sels, indent=2) + "\n")
    _w(d / "selections.log.json", {
        "hand_built": True, "plan_sha": sel_log.get("plan_sha"), "menus": sel_log.get("menus"),
        "note": "menus copied from call2/selections.log.json (the re-pick must stay inside "
                "what call #2 was shown); 'resolved' is not recorded for a hand-built case"})
    em = copy.deepcopy(em_doc)
    em["hardware"]["selections_sha256"] = _sha(sel_path)
    em["hand_built"] = {"by": "critic_cases.py build", "from": [SELECTIONS, EMISSIONS], **edit}
    _w(d / "emissions.json", em)
    (d / "README.md").write_text(readme)
    return d


def build_cases(run: Path) -> list[Path]:
    """critic_cases/spout_repick and critic_cases/spout_in_mug from the run's
    call #2 selections and call #3 (stand-in) emissions."""
    run = Path(run)
    sels = json.loads((run / SELECTIONS).read_text())
    sel_log = json.loads((run / SELECTIONS).with_name("selections.log.json").read_text())
    em_doc = json.loads((run / EMISSIONS).read_text())
    spout = _spout_candidate(sel_log)
    old = sels[CARRY_ROLE]["candidate_id"]
    repick = copy.deepcopy(sels)
    repick[CARRY_ROLE]["candidate_id"] = spout
    repick[CARRY_ROLE]["rationale"] = (
        f"HAND-BUILT (critic_cases.py): reselect_point teapot.{SPOUT} applied to call #2's "
        f"{CARRY_ROLE} pick (was candidate {old}), as the critic's round-1 edit does")
    pick = {"selection_edit": {"role": CARRY_ROLE, "candidate_id": [old, spout],
                               "point": f"teapot.{SPOUT}"}}
    out = [_write_case(run, "spout_repick", repick, sel_log, em_doc, (
        "# spout_repick (HAND-BUILT)\n\n"
        "Hardware run 20261105_035449 with one change: call #2's transport_active pick moved "
        f"from candidate {old} (teapot.handle_center) to candidate {spout} "
        f"(teapot.{SPOUT}), i.e. the critic's round-1 edit `reselect_point teapot.{SPOUT}` "
        "already applied. Call #3's emissions are the run's (sim stand-in), unchanged except "
        "the recorded selections sha256. Written by `critic_cases.py build`.\n\n"
        "Expected critic verdict: accept.\n"), pick)]

    bad = copy.deepcopy(em_doc)
    roles = bad.get("roles") or []
    carry = bad["emissions"][roles.index(CARRY_ROLE)]
    terms = carry["subgoal_tsr"]["trans"]
    above = [i for i, t in enumerate(terms) if t.get("term") == "above"]
    if len(above) != 1:
        raise SystemExit(f"[cases] the carry subgoal has {len(above)} 'above' terms; expected 1")
    i = above[0]
    anchor = terms[i]["anchor"]
    terms[i]["term"] = "below"
    carry["goal"] = (f"Lower teapot so that teapot.{SPOUT} is centered on {anchor}, a small "
                     "clearance below it, with teapot.+up pointing up throughout.")
    carry["verify"] = (f"teapot.+up is vertical; teapot.{SPOUT} is horizontally centered on "
                       f"{anchor} and a small clearance below it.")
    edit = {**pick, "emission_edit": {
        "stage": carry["name"], "slot": f"{carry['name']}.subgoal.trans[{i}]",
        "term": ["above", "below"], "goal_and_verify_text": "rewritten to match"}}
    out.append(_write_case(run, "spout_in_mug", repick, sel_log, bad, (
        "# spout_in_mug (HAND-BUILT negative case)\n\n"
        "A plausible-looking specification that means the wrong thing. Hardware run "
        f"20261105_035449 with the carry anchored on teapot.{SPOUT} (as in spout_repick), and "
        f"call #3's carry subgoal term `{carry['name']}.subgoal.trans[{i}]` hand-edited from "
        f"`above {anchor}` to `below {anchor}` (same clearance and slack; the goal and verify "
        "text rewritten to match). The spout goes down into the mug, so the teapot body "
        "intersects it. Real tokens only; it compiles through the normal path. Written by "
        "`critic_cases.py build`.\n\n"
        "Expected critic verdict: reject.\n"), edit))
    return out


# ------------------------------------------------------------------ run

def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def request_stats(prompt_json: Path) -> dict:
    """Sizes of the logged request: text chars (system + every text block),
    images and their bytes (base64 adds a third on the wire)."""
    req = json.loads(Path(prompt_json).read_text())
    blocks = [b for m in req["messages"] for b in m["content"]]
    texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
    imgs = [b["path"] for b in blocks if b.get("type") == "image"]
    img_bytes = sum(Path(p).stat().st_size for p in imgs if Path(p).is_file())
    return {"system_chars": len(req["system"]), "text_block_chars": sum(map(len, texts)),
            "text_chars": len(req["system"]) + sum(map(len, texts)),
            "text_blocks": len(texts), "images": len(imgs), "image_bytes": img_bytes,
            "image_base64_chars": 4 * ((img_bytes + 2) // 3) if imgs else 0}


EXPECTED_STATUS = {"live": "verdict", "replay": "verdict", "dry_run": "dry_run"}


def case_record(case: Case, code: int | None, case_out: Path, mode: str,
                error: str | None = None) -> dict:
    """One case's row. `error` (a case that did not run to its verdict, or
    to its request in a dry run) is set from the exit code, an exception,
    a missing record or a failed critic call; a verdict that differs from
    the expected one is a mismatch (`match` False), not an error."""
    rec = {"case": case.name, "expected": case.expected, "about": case.about, "exit": code,
           "out": str(case_out), "error": error, "actual": None, "match": None}
    summ_p = case_out / "critic" / "critic.json"
    if not summ_p.is_file():
        rec["error"] = error or f"exit {code}: no critic record"
        rec["status"] = "error"
        return rec
    summ = json.loads(summ_p.read_text())
    rd = case_out / "critic" / "round_1"
    v = summ.get("verdict") or {}
    rec.update(status=summ.get("status"), mode=summ.get("mode"), model=summ.get("model"),
               calls=summ.get("calls"), actual=v.get("verdict"),
               edits=[f"{e['action']} {e['target']}" + (f" {e['token']}" if e.get("token") else "")
                      for e in v.get("edits") or []],
               diagnosis=v.get("diagnosis"), stages=v.get("stages") or [],
               rejections=(summ.get("rounds") or [{}])[0].get("rejections"),
               request=str(rd / "prompt.json"),
               raw_response=str(rd / "raw_response.txt") if (rd / "raw_response.txt").is_file()
               else None,
               verdict_file=str(rd / "verdict.json") if (rd / "verdict.json").is_file() else None,
               sizes=request_stats(rd / "prompt.json") if (rd / "prompt.json").is_file() else None,
               failure=summ.get("failure"))
    rec["match"] = None if rec["actual"] is None else rec["actual"] == case.expected
    if rec["error"] is None and (code != 0 or rec["status"] != EXPECTED_STATUS[mode]):
        rec["error"] = (f"exit {code}, critic status {rec['status']}"
                        + (f": {rec['failure'].get('reason')}" if rec.get("failure") else ""))
    return rec


def summary_md(recs: list[dict], mode: str) -> str:
    lines = [f"# Critic cases ({mode})", "",
             "| case | expected | actual | match | edits | text chars | images | error |",
             "|---|---|---|---|---|---|---|---|"]
    for r in recs:
        sz = r.get("sizes") or {}
        lines.append(f"| {r['case']} | {r['expected']} | {r.get('actual') or r.get('status')} | "
                     f"{'' if r.get('match') is None else ('yes' if r['match'] else 'NO')} | "
                     f"{'; '.join(r.get('edits') or []) or '-'} | {sz.get('text_chars', '-')} | "
                     f"{sz.get('images', '-')} | {r.get('error') or ''} |")
    for r in recs:
        lines += ["", f"## {r['case']}: {r['about']}", ""]
        if r.get("diagnosis"):
            lines += [f"Diagnosis: {r['diagnosis']}", ""]
        for s in r.get("stages") or []:
            lines += [f"- **{s['stage']}**", f"  - expected: {s['expected']}",
                      f"  - observed: {s['observed']}"]
        lines += ["", f"Request: `{r.get('request')}`"]
    return "\n".join(lines) + "\n"


def run_cases(run: Path, out: Path | None, mode: str | None, responses: Path | None,
              names: list[str] | None) -> int:
    run = Path(run)
    cases = [c for c in CASES if not names or c.name in names]
    n = len(cases)
    print(f"[cases] {n} case(s) x 1 critic call = {n} paid call(s) if every response parses, "
          f"at most {n * (1 + vlm_compat.MAX_PARSE_RETRIES)} with parse retries; verdict only "
          "(no edit, no second round, no planning)")
    if mode is None:
        print("[cases] not run: pass --live to make the paid calls (through scripts/vlm_run.sh), "
              "or --dry-run to write the requests without a call")
        return 0
    for c in cases:
        for f in (c.selections, c.emissions):
            if not (run / f).is_file():
                raise SystemExit(f"[cases] {run / f} not found -- run `critic_cases.py build "
                                 f"--run {run}` first")
    out = Path(out) if out else (run / "critic_cases" / "results"
                                 / f"{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_{mode}")
    psh = _load("plan_stages_hw")
    recs = []
    for c in cases:
        case_out = out / c.name
        args = ["--run", str(run), "--method", "vlm", "--emissions", str(run / c.emissions),
                "--selections", str(run / c.selections), "--critic", "preview",
                "--critic-verdict-only", "--out", str(case_out)]
        if mode == "dry_run":
            args.append("--critic-dry-run")
        elif mode == "replay":
            args += ["--critic-responses", str(Path(responses) / c.name)]
        print(f"[cases] {c.name} (expected {c.expected})")
        code, err = None, None
        try:
            code = psh.main(args)
        except (Exception, SystemExit) as e:        # one case's failure must not end the loop
            err = f"{type(e).__name__}: {e}"
            print(f"[cases] {c.name}: {err}")
        recs.append(case_record(c, code, case_out, mode, err))
    _w(out / "summary.json", {"run": str(run), "mode": mode, "model": (
        vlm_compat.MODEL if mode == "live" else None), "cases": recs})
    (out / "summary.md").write_text(summary_md(recs, mode))
    for r in recs:
        sz = r.get("sizes") or {}
        print(f"[cases] {r['case']}: expected {r['expected']}, got "
              f"{r.get('actual') or r.get('status')}; request {sz.get('text_chars')} text chars, "
              f"{sz.get('images')} image(s), {sz.get('image_bytes')} image bytes"
              + (f"; ERROR {r['error']}" if r.get("error") else ""))
    errors = [r["case"] for r in recs if r.get("error")]
    mism = [r["case"] for r in recs if r.get("match") is False]
    print(f"[cases] -> {out / 'summary.md'}"
          + (f"; verdict mismatch: {mism}" if mism else "")
          + (f"; ERRORS: {errors}" if errors else ""))
    return 1 if errors else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write the hand-built cases into <run>/critic_cases/")
    b.add_argument("--run", required=True)
    r = sub.add_parser("run", help="one critic call per case (verdict only)")
    r.add_argument("--run", required=True)
    r.add_argument("--out", default=None)
    r.add_argument("--cases", nargs="+", choices=[c.name for c in CASES], default=None)
    g = r.add_mutually_exclusive_group()
    g.add_argument("--live", action="store_true", help="PAID: make the critic calls")
    g.add_argument("--dry-run", action="store_true", help="write each request; no call")
    g.add_argument("--responses", default=None, help="replay DIR/<case>/critic_* (no call)")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        for d in build_cases(Path(a.run)):
            print(f"[cases] wrote {d}")
        return 0
    mode = "live" if a.live else "dry_run" if a.dry_run else "replay" if a.responses else None
    return run_cases(Path(a.run), a.out, mode, Path(a.responses) if a.responses else None,
                     a.cases)


if __name__ == "__main__":
    sys.exit(main())
