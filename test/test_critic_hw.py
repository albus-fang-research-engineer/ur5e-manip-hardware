"""plan_stages_hw.py --method vlm --critic preview (critic_scene.py), offline:
the recorded 11/05 run with call #3's sim stand-in emissions, canned critic
and call #3 responses (--critic-responses), a stub renderer (no GL), the
fake cuRobo sidecar and stub collision spheres from test_plan_stages_hw_vlm.
No network, no API call: the live transport is replaced by one that fails.

Covered: (a) reject with reselect_point teapot.spout_center on the carry ->
recompile -> accept, and the planner gets the re-picked carry; (b) accept on
round 1 plans call #3's regions unchanged; (c) a reject once the 2 rounds of
edits are spent -> preview_rejected, nothing planned; (d) an illegal edit is
refused and goes back to the model, the refusal recorded; (e) --critic off
writes the same report as no flag and never loads the critic; reemit_stage
re-runs call #3 through the compile-retry path; the dry run writes round 1's
prompt and images and stops; the image captions carry the manifest's facts.

Needs Python >= 3.10, a sim checkout WITH the critic at $SIM_DIR (branch
critic-repair) and manip_cbirrt (PYTHONPATH=../manip-cbirrt).
"""
import inspect
import json
import os
import sys
from pathlib import Path

import pytest

if sys.version_info < (3, 10):
    pytest.skip("the drivers and the sim need Python >= 3.10; run with the manip-tsr env",
                allow_module_level=True)
np = pytest.importorskip("numpy")
pytest.importorskip("manip_cbirrt", reason="PYTHONPATH=../manip-cbirrt")
pytest.importorskip("trimesh")

REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_DIR = Path(os.environ.get("SIM_DIR", REPO_ROOT.parent / "ur5e-manip-sim")).resolve()
if not (SIM_DIR / "manip_sim" / "critic.py").is_file():
    pytest.skip(f"no sim checkout with the critic at {SIM_DIR} (set SIM_DIR to critic-repair)",
                allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_plan_stages_hw_vlm import FakeSidecar, _main, hw, vlm_env  # noqa: E402,F401

CARRY = "lift and move teapot above mug"
POUR = "tilt teapot to pour tea into mug"
GRASP = "grasp teapot by handle"
NO_GRIPPER = ["--gripper-urdf", "/nonexistent/ur5e_robotiq.urdf"]


# ------------------------------------------------------------------ helpers

def _verdict(verdict="accept", edits=(), diagnosis=""):
    return json.dumps({"stages": [{"stage": s, "expected": f"{s}: e", "observed": f"{s}: o"}
                                  for s in (GRASP, CARRY, POUR)],
                       "verdict": verdict, "edits": list(edits), "diagnosis": diagnosis})


REPICK = _verdict("reject", [{"target": f"{CARRY}.point.teapot", "action": "reselect_point",
                              "token": "teapot.spout_center"}],
                  "the handle, not the spout, is over the mug")


def _responses(tmp_path, critic, emit=()):
    d = tmp_path / "responses"
    d.mkdir(exist_ok=True)
    for i, r in enumerate(critic):
        (d / f"critic_{i + 1:02d}.txt").write_text(r)
    for i, r in enumerate(emit):
        (d / f"emit_{i + 1:02d}.txt").write_text(r)
    return d


def _stub_manifest(rp, initial, stages, out_dir, grasp, collisions):
    """render_preview's manifest, in its schema, from the renderer's own
    GL-free geometry (chain_states, representative_pose, boundary_samples),
    with tiny placeholder PNGs. With `collisions`, the facts the renderer's
    collision check would report on 11/05: the carry goal intersecting the
    mug, x -10 cm dropped (it collides while the center pose is free), the
    roll panels kept but in collision. Without, the check is off (the
    manifest's collision fields are None), as for collisions=False."""
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    states, skipped = rp.chain_states(initial, stages, grasp)
    on = bool(collisions)
    m = {"style": "panels", "images": [], "skipped": skipped, "path_text_only": [],
         "gripper": "stub 2F-85",
         "collision_check": ({"bodies": "scene objects, gripper, table (no arm)", "tol_mm": 2.0,
                              "min_points": 3} if on else None)}
    hit = [{"a": "teapot", "b": "mug", "depth_mm": 6.0}]

    def png(name):
        Image.new("RGB", (4, 4)).save(out_dir / name)
        return name

    for n, s in enumerate(states):
        contacts = None
        if on and s.key != "initial":
            contacts = hit if s.key == "goal:transport_active" else []
        m["images"].append({
            "file": png(f"{n:02d}_state_{s.key.replace(':', '_')}.png"), "kind": "state",
            "state": s.key, "title": s.title, "role": s.role, "panels": ["iso", "top"],
            "collisions": contacts,
            "notes": ["teapot intersects mug"] if contacts else []})
    entry = {s.role: states[k - 1] for k, s in enumerate(states)
             if k > 0 and s.key.startswith("goal:")}
    for st in stages:
        if st.path is None or st.role not in entry:
            continue
        d_rep, _ = rp.representative_pose(st.path, st.entry, st.subgoal.nominal())
        bounds, pinned = rp.boundary_samples(st.path, d_rep)
        for group, rows in (("translation", (0, 1, 2)), ("rotation", (3, 4, 5))):
            rec_rows, notes = [], []
            for b in (b for b in bounds if b.row in rows):
                descs = getattr(b, "descs", b.captions)
                panels = []
                for v, d in zip(b.values, descs):
                    h = None
                    if on:
                        h = hit if (rp.ROW_NAMES[b.row] == "x" and v < d_rep[0]) or \
                            rp.ROW_NAMES[b.row] == "roll" else []
                    dropped = bool(h) and rp.ROW_NAMES[b.row] == "x"
                    if dropped:
                        notes.append(f"{d}: teapot collides with mug (the planner avoids this)")
                    panels.append({"value": v, "desc": d, "collides": None if h is None
                                   else bool(h), "collisions": h, "dropped": dropped})
                rec_rows.append({"row": rp.ROW_NAMES[b.row], "label": b.label, "kind": b.kind,
                                 "token": b.token, "values": b.values, "captions": b.captions,
                                 "view": "+x" if b.row >= 3 else "top", "panels": panels})
            if not rec_rows:
                continue
            m["images"].append({
                "file": png(f"{len(m['images']):02d}_path_{st.role}_{group}.png"),
                "title": f"{st.role} path region, {group} rows", "kind": "path_boundary",
                "stage": st.role, "name": st.name, "group": group, "rows": rec_rows,
                "pinned_rows": pinned, "notes": notes,
                "center_collides": False if on else None,
                "representative": {"collisions": [] if on else None}})
    (out_dir / "manifest.json").write_text(json.dumps(m, indent=1) + "\n")
    return m


@pytest.fixture
def critic_env(hw, vlm_env, monkeypatch):
    """vlm_env plus: a stub preview renderer, a stub stage-frame renderer
    for re-emits, and a live transport that fails the test if reached."""
    from manip_bridge import critic_compat as cc
    from manip_bridge import emit_compat as ec
    from manip_bridge import vlm_compat
    run, setup = vlm_env
    opts = {"collisions": False}

    def render_preview(scene, initial, stages, labels, out_dir, grasp=None, gripper=None,
                       style="panels"):
        opts["rendered"] = {"gripper": gripper, "grasp": grasp}
        return _stub_manifest(cc._rp, initial, stages, Path(out_dir), grasp, opts["collisions"])
    monkeypatch.setattr(cc, "render_preview", render_preview)
    monkeypatch.setattr(ec, "stage_frames_renderer", lambda: (
        lambda *a, **k: {a[0][0][1]: None},                 # compute_stage_frames
        lambda *a, **k: {"views": {}},                      # render_role
        lambda scene, poses: (None, 1.0), "stub"))          # scene_extent

    def no_live(payload):
        raise AssertionError("a test reached the live API transport")
    monkeypatch.setattr(vlm_compat, "live_transport", no_live)
    return run, setup, opts


def _report(run):
    return json.loads((run / "m4_vlm" / "report.json").read_text())


def _critic(hw, run, tmp_path, critic, emit=(), *extra):
    d = _responses(tmp_path, critic, emit)
    return _main(hw, run, "--critic", "preview", "--critic-responses", str(d), *NO_GRIPPER,
                 *extra)


# ------------------------------------------------------------------ (a) re-pick, then accept

def test_repick_spout_recompiles_and_accepts(hw, critic_env, tmp_path):
    run, setup, _ = critic_env
    side, calls = setup({})
    assert _critic(hw, run, tmp_path, [REPICK, _verdict()]) == 2      # the pour: O2, as before
    rep = _report(run)
    c = rep["critic"]
    assert c["status"] == "accepted" and c["rounds_to_accept"] == 2
    assert [(e["action"], e["target"], e["token"]) for e in c["edits_applied"]] == [
        ("reselect_point", f"{CARRY}.point.teapot", "teapot.spout_center")]
    assert c["edits_applied"][0]["before"].endswith("handle_center")
    assert c["calls"] == {"critic": 2, "call3_reemit": 0}
    assert rep["method"] == "vlm (call #3 emissions) + critic"
    assert c["model"] is None and c["mode"] == "replay"         # no model ran
    # the spout moved over the mug (logged, not gated)
    key = "teapot.spout_center -> mug.rim_center"
    before = c["point_gaps"]["before"]["transport_active"][key]["horizontal_m"]
    after = c["point_gaps"]["after"]["transport_active"][key]["horizontal_m"]
    assert after < before and after < 0.03
    # round 1 shows the handle-anchored carry, round 2 the re-picked one
    crit = run / "m4_vlm" / "critic"
    r1 = json.loads((crit / "round_1" / "report.json").read_text())
    r2 = json.loads((crit / "round_2" / "report.json").read_text())
    carry = lambda r: next(s for s in r["stages"] if s["role"] == "transport_active")
    assert carry(r1)["point_choice"]["current"] == "teapot.handle_center"
    assert carry(r2)["point_choice"]["current"] == "teapot.spout_center"
    pour = next(s for s in r1["stages"] if s["role"] == "pour")
    assert "compile_error" in pour and pour["path"]["rows"]       # O2: listed, rows as text
    for f in ("prompt.txt", "prompt.json", "verdict.json", "edits.json", "rejections.json",
              "preview_inputs.json", "images/manifest.json"):
        assert (crit / "round_1" / f).is_file(), f
    # the planner got the re-picked carry: its sidecar request carries the new Tw_e
    assert rep["attempts"][0]["stages"][0]["outcome"] == "planned"
    tw_e = np.array(side.requests[0]["path"][0]["tw_e"]).reshape(4, 4)
    inputs = json.loads((crit / "round_2" / "preview_inputs.json").read_text())
    tw_r2 = np.array(next(s for s in inputs["stages"]
                          if s["role"] == "transport_active")["path"]["Tw_e"])
    np.testing.assert_allclose(tw_e, tw_r2, atol=1e-6)
    inputs1 = json.loads((crit / "round_1" / "preview_inputs.json").read_text())
    tw_r1 = np.array(next(s for s in inputs1["stages"]
                          if s["role"] == "transport_active")["path"]["Tw_e"])
    assert not np.allclose(tw_r1, tw_r2, atol=1e-3)            # the anchor moved


# ------------------------------------------------------------------ (b) accept on round 1

def test_accept_on_round_one_plans_call3_unchanged(hw, critic_env, tmp_path):
    run, setup, _ = critic_env
    setup({})
    assert _main(hw, run) == 2
    plain = _report(run)["first_stage_compile"]["compiled"]
    setup({})
    assert _critic(hw, run, tmp_path, [_verdict()]) == 2
    rep = _report(run)
    assert rep["critic"]["status"] == "accepted" and rep["critic"]["rounds_to_accept"] == 1
    assert rep["critic"]["edits_applied"] == []
    assert rep["first_stage_compile"]["compiled"] == plain


# ------------------------------------------------------------------ (c) the cap

def test_reject_after_two_rounds_is_preview_rejected(hw, critic_env, tmp_path):
    run, setup, _ = critic_env
    side, _ = setup({})
    tighten = lambda ptr: _verdict("reject", [{"target": ptr, "action": "tighten_tolerance",
                                               "token": "tight"}], "still too loose")
    code = _critic(hw, run, tmp_path, [REPICK, tighten(f"{CARRY}.subgoal.rot[0]"),
                                       tighten(f"{CARRY}.path.rot[0]")])
    assert code == 2 and side.requests == []                  # nothing planned
    rep = _report(run)
    assert rep["status"] == "failed" and rep["critic"]["status"] == "preview_rejected"
    f = rep["failures"][0]
    assert f["type"] == "preview_rejected" and f["rounds_used"] == 2 and f["budget"] == 2
    from manip_bridge import vlm_compat
    assert f["type"] in vlm_compat.FAILURE_TYPES
    assert len(rep["critic"]["edits_applied"]) == 2 and rep["critic"]["calls"]["critic"] == 3


# ------------------------------------------------------------------ (d) illegal edit

def test_illegal_edit_is_refused_and_recorded(hw, critic_env, tmp_path):
    run, setup, _ = critic_env
    setup({})
    bad = _verdict("reject", [{"target": f"{CARRY}.point.teapot", "action": "reselect_point",
                               "token": "mug.rim_center"}], "aim at the rim")   # not the mover's
    assert _critic(hw, run, tmp_path, [bad, _verdict()]) == 2
    c = _report(run)["critic"]
    assert c["status"] == "accepted" and c["rounds"][0]["attempts"] == 2
    (why,) = c["rounds"][0]["rejections"]
    assert "mug.rim_center" in why
    rec = json.loads((run / "m4_vlm" / "critic" / "round_1" / "rejections.json").read_text())
    assert rec == [why]


# ------------------------------------------------------------------ (e) off is unchanged

def _normalized(run):
    rep = _report(run)
    rep.pop("run", None)
    return json.dumps(rep, sort_keys=True).replace(str(run), "<run>")


def _no_critic(hw, monkeypatch):
    """Make the critic unloadable: plan_stages_hw._load refuses
    critic_scene, and importing manip_bridge.critic_compat raises, as on a
    sim checkout without manip_sim.critic. Returns the names _load saw."""
    import manip_bridge
    loaded, real = [], hw._load

    def load(name):
        loaded.append(name)
        return real(name)
    monkeypatch.setattr(hw, "_load", load)
    monkeypatch.setitem(sys.modules, "manip_bridge.critic_compat", None)
    monkeypatch.delattr(manip_bridge, "critic_compat", raising=False)
    return loaded


def test_critic_off_is_unchanged(hw, critic_env, tmp_path, monkeypatch):
    """--critic off (and no flag) plan as before and never touch the critic:
    with the critic made unimportable, both still run, write the same
    report, and never load critic_scene."""
    run, setup, _ = critic_env
    loaded = _no_critic(hw, monkeypatch)
    setup({})
    assert _main(hw, run) == 2                      # the stand-in pour: O2, as before
    default = _normalized(run)
    setup({})
    assert _main(hw, run, "--critic", "off") == 2
    assert _normalized(run) == default and "critic" not in _report(run)
    assert not (run / "m4_vlm" / "critic").exists()
    assert "critic_scene" not in loaded and "emit_scene" in loaded


def test_sim_without_the_critic_is_a_clear_refusal(hw, critic_env, monkeypatch, capsys):
    """--critic preview on a sim checkout without manip_sim.critic: a typed
    refusal naming the fix (exit 2), not an import crash or a misleading
    PYTHONPATH message."""
    run, setup, _ = critic_env
    import manip_bridge
    setup({})
    monkeypatch.setitem(sys.modules, "manip_bridge.critic_compat", None)
    monkeypatch.delattr(manip_bridge, "critic_compat", raising=False)
    assert _main(hw, run, "--critic", "preview", "--critic-dry-run", *NO_GRIPPER) == 2
    out = capsys.readouterr().out
    assert "REFUSED: sim checkout has no manip_sim.critic" in out
    assert "merge critic-repair or point SIM_DIR at it" in out and "PYTHONPATH" not in out


def test_critic_cases_survive_a_missing_critic(hw, critic_env, tmp_path, monkeypatch):
    """Each case refuses on its own; the loop goes on, the summary is
    written, and the harness exits nonzero (errors, not mismatches)."""
    import manip_bridge
    run, setup, _ = critic_env
    setup({})
    cases = hw._load("critic_cases")
    cases.build_cases(run)
    monkeypatch.setitem(sys.modules, "manip_bridge.critic_compat", None)
    monkeypatch.delattr(manip_bridge, "critic_compat", raising=False)
    out = tmp_path / "results"
    assert cases.run_cases(run, out, "dry_run", None, None) == 1
    summ = json.loads((out / "summary.json").read_text())
    assert [r["case"] for r in summ["cases"]] == ["handle_anchor", "spout_repick", "spout_in_mug"]
    assert all(r["error"] and r["match"] is None for r in summ["cases"])
    assert "| handle_anchor | reject | error |" in (out / "summary.md").read_text()


def test_critic_flags_need_preview(hw):
    for extra in (["--critic-dry-run"], ["--critic-responses", "d"],
                  ["--critic", "preview", "--critic-dry-run", "--critic-responses", "d"]):
        with pytest.raises(SystemExit):
            hw.main(["--run", "x", "--method", "vlm", "--emissions", "e.json", *extra])
    with pytest.raises(SystemExit):
        hw.main(["--run", "x", "--critic", "preview"])                  # hand method


# ------------------------------------------------------------------ re-emit, dry run

@pytest.fixture
def feedback(monkeypatch):
    """Records each Client.emit_constraints call's feedback and frames and
    runs the real method (the sim's feedback= argument; the canned
    transport supplies the response)."""
    from manip_bridge import vlm_compat
    real, calls = vlm_compat.Client.emit_constraints, []
    assert "feedback" in inspect.signature(real).parameters, \
        "sim Client.emit_constraints lacks feedback= (critic-repair predates it)"

    def emit_constraints(self, stage, vocab, selection=None, view_paths=None, rejections=None,
                         frames=None, entry=None, feedback=None):
        calls.append({"stage": stage.name, "feedback": feedback, "views": view_paths,
                      "frames": frames, "entry": entry, "rejections": list(rejections or [])})
        return real(self, stage, vocab, selection=selection, view_paths=view_paths,
                    rejections=rejections, frames=frames, entry=entry, feedback=feedback)
    monkeypatch.setattr(vlm_compat.Client, "emit_constraints", emit_constraints)
    return calls


def _raw_of(run, name):
    log = json.loads((run / "call3_simreplay" / "emissions.log.json").read_text())
    return next(r for r in log["raw_responses"] if json.loads(r)["name"] == name)


def _reemits(run, n):
    return json.loads((run / "m4_vlm" / "critic" / f"round_{n}" / "reemit.json").read_text())


def test_reemit_without_sim_feedback_is_refused(hw, critic_env, tmp_path, monkeypatch):
    """A sim checkout whose emit_constraints has no feedback= (older than
    critic-repair): a re-emit is a typed record and no call #3 request is
    sent with wrong wording."""
    from manip_bridge import vlm_compat
    real = vlm_compat.Client.emit_constraints

    def old_signature(self, stage, vocab, selection=None, view_paths=None, rejections=None,
                      frames=None, entry=None):
        return real(self, stage, vocab, selection=selection, view_paths=view_paths,
                    rejections=rejections, frames=frames, entry=entry)
    monkeypatch.setattr(vlm_compat.Client, "emit_constraints", old_signature)
    run, setup, _ = critic_env
    setup({})
    reemit = _verdict("reject", [{"target": CARRY, "action": "reemit_stage", "token": None}],
                      "the carry needs the teapot kept level")
    assert _critic(hw, run, tmp_path, [reemit, _verdict()], [_raw_of(run, CARRY)]) == 2
    assert _report(run)["critic"]["calls"]["call3_reemit"] == 0
    (rec,) = _reemits(run, 1)
    assert "no feedback= argument" in rec["error"]


def test_reemit_compiled_stage_says_the_critic_rejected_it(hw, critic_env, tmp_path, feedback):
    """A stage that compiled: its recorded raw (still exactly the emission)
    replayed, the follow-up says the preview critic rejected it -- no
    "compiler rejected" claim; the first mover is not chained; the framed
    arm renders the stage frames."""
    run, setup, _ = critic_env
    setup({})
    carry_raw = _raw_of(run, CARRY)
    reemit = _verdict("reject", [{"target": CARRY, "action": "reemit_stage", "token": None}],
                      "the carry needs the teapot kept level")
    assert _critic(hw, run, tmp_path, [reemit, _verdict()], [carry_raw]) == 2
    c = _report(run)["critic"]
    assert c["status"] == "accepted" and c["calls"] == {"critic": 2, "call3_reemit": 1}
    (rec,) = _reemits(run, 1)
    assert rec["role"] == "transport_active" and rec["attempts"] == [
        {"raw": carry_raw, "compiled": True}]
    assert rec["prior"] == "recorded" and rec["chained"] is False and rec["arm"] == "framed-chained"
    (call,) = feedback
    ((prior, text),) = call["feedback"]
    assert prior == carry_raw and call["rejections"] == []
    assert text == ("A preview critic reviewed this stage before planning and rejected it: the "
                    "carry needs the teapot kept level. Re-emit the corrected JSON object only, "
                    "changing as little as needed.")
    assert "compiler" not in text
    assert "mug" in call["entry"]                     # the entry line, anchored as emit_scene


def test_reemit_uncompiled_stage_carries_the_compile_error(hw, critic_env, tmp_path, feedback):
    """The stand-in pour does not compile (O2): the follow-up carries the
    real compile error and the diagnosis; the pour is chained (the carry
    moved the teapot before it)."""
    run, setup, _ = critic_env
    setup({})
    pour_raw = _raw_of(run, POUR)
    reemit = _verdict("reject", [{"target": POUR, "action": "reemit_stage", "token": None}],
                      "the tilt axis is wrong")
    # call #3 answers with the same pour each time: every compile retry fails too
    from manip_bridge import emit_compat as ec
    emits = [pour_raw] * (1 + ec.COMPILE_RETRIES)
    _critic(hw, run, tmp_path, [reemit, _verdict()], emits)
    (rec,) = _reemits(run, 1)
    assert rec["chained"] is True and len(rec["attempts"]) == 1 + ec.COMPILE_RETRIES
    text = feedback[0]["feedback"][0][1]
    assert text.startswith("The compiler rejected that emission: tilt teapot to pour tea into "
                           "mug.path.rot[0]: the constrained tilt axis does not lie along a w "
                           "basis vector")
    assert text.endswith("A preview critic also reviewed the stage before planning: the tilt "
                         "axis is wrong. Re-emit the corrected JSON object only, changing as "
                         "little as needed.")
    # later attempts carry the compiler's own rejections, as emit_scene's retries do
    assert feedback[1]["rejections"] and feedback[1]["feedback"] == feedback[0]["feedback"]


def test_reemit_runs_at_the_entry_of_the_edited_chain(hw, critic_env, tmp_path, feedback):
    """One verdict re-picks the carry's point AND re-emits the pour: the
    re-pick is applied and the chain recompiled first, so the pour is
    re-emitted at the NEW carry goal, not the old one."""
    run, setup, _ = critic_env
    setup({})
    from manip_bridge import emit_compat as ec
    verdict = _verdict("reject", [
        {"target": f"{CARRY}.point.teapot", "action": "reselect_point",
         "token": "teapot.spout_center"},
        {"target": POUR, "action": "reemit_stage", "token": None}], "spout over the mug")
    _critic(hw, run, tmp_path, [verdict, _verdict()],
            [_raw_of(run, POUR)] * (1 + ec.COMPILE_RETRIES))
    (rec,) = _reemits(run, 1)
    crit = run / "m4_vlm" / "critic"
    pour_entry = lambda n: next(s for s in json.loads(
        (crit / f"round_{n}" / "preview_inputs.json").read_text())["stages"]
        if s["role"] == "pour")["entry"]
    np.testing.assert_allclose(rec["entry_pose"], pour_entry(2), atol=1e-6)    # new chain
    assert not np.allclose(rec["entry_pose"], pour_entry(1), atol=1e-3)        # not the old one


def test_reemit_replays_the_edited_emission_and_follows_the_arm(hw, critic_env, tmp_path,
                                                                 feedback, monkeypatch):
    """Round 1 tightens a carry row; round 2 re-emits the carry: the prior
    turn is the CURRENT (edited) emission, not call #3's original raw. The
    run's arm is schema-only here, so no frames are rendered or sent."""
    run, setup, _ = critic_env
    setup({})
    em_path = run / "call3_simreplay" / "emissions.json"
    doc = json.loads(em_path.read_text())
    doc["arm"] = "schema-only"
    em_path.write_text(json.dumps(doc, indent=2) + "\n")
    from manip_bridge import emit_compat as ec
    rendered, stub = [], ec.stage_frames_renderer          # critic_env's stub renderer
    monkeypatch.setattr(ec, "stage_frames_renderer", lambda: rendered.append(1) or stub())
    tighten = _verdict("reject", [{"target": f"{CARRY}.subgoal.rot[0]",
                                   "action": "tighten_tolerance", "token": "tight"}], "level")
    reemit = _verdict("reject", [{"target": CARRY, "action": "reemit_stage", "token": None}],
                      "rewrite the carry")
    _critic(hw, run, tmp_path, [tighten, reemit, _verdict()], [_raw_of(run, CARRY)])
    (rec,) = _reemits(run, 2)
    assert rec["prior"] == "current" and rec["arm"] == "schema-only" and rec["views"] == []
    assert rendered == [] and feedback[0]["views"] is None and feedback[0]["frames"] is None
    from manip_bridge import vlm_compat
    prior = vlm_compat.parse_emission(rec["prior_raw"], vlm_compat.Vocabulary.from_asset_dirs(
        {o: run / "m4_vlm" / "assets" / o for o in ("teapot", "mug")}))
    assert prior.subgoal_tsr.rot[0].tol == "tight"                       # the edited row
    assert prior.path_tsr.rot[0].tol == "moderate"


def test_emission_raw_round_trips(hw, critic_env):
    """emission_raw writes call #3's output schema; parse_emission reads it
    back to the same emission, for every 11/05 stage."""
    run, _, _ = critic_env
    from manip_bridge import vlm_compat
    cs = hw._load("critic_scene")
    vocab = vlm_compat.Vocabulary.from_asset_dirs(
        {"teapot": run / "grounding_vlm" / "canon" / "lib_teapot",
         "mug": run / "grounding_vlm" / "canon" / "lib_mug"})
    doc = json.loads((run / "call3_simreplay" / "emissions.json").read_text())
    for e in doc["emissions"]:
        em = vlm_compat.emission_from_json(e)
        assert vlm_compat.parse_emission(cs.emission_raw(em), vocab) == em


def test_chained_follows_emit_scene(hw, critic_env):
    from types import SimpleNamespace as NS
    cs = hw._load("critic_scene")
    st = lambda n, a, p: NS(name=n, active=a, passive=p)
    em = lambda a, p: NS(active=a, passive=p)
    ctx = NS(stages=[(st("g", "teapot", None), "grasp"), (st("c", "teapot", "mug"), "carry"),
                     (st("p", "teapot", "mug"), "pour"), (st("m", "mug", "tray"), "move_mug")])
    state = NS(emissions={"grasp": em("teapot", None), "carry": em("teapot", "mug"),
                          "pour": em("teapot", "mug"), "move_mug": em("mug", "tray")})
    assert [cs._chained(ctx, state, r) for r in ("carry", "pour", "move_mug")] == [
        False, True, False]


# ------------------------------------------------------------------ accepted = planned

def test_accepted_movers_never_fall_back(hw):
    """The planner gets what the critic accepted; a mover the accepted
    specification cannot supply is a failure, never the original."""
    from types import SimpleNamespace as NS
    movers = [("transport_active", NS(name="c"), "EM0", "W0", "E0", False),
              ("pour", NS(name="p"), "EM1", "W1", "E1", False)]
    fail = {"type": None, "stage": "pour", "phase": "anchor", "reason": "x"}
    critic = {"accepted_spec": ({"transport_active": ("EM0b", "W0b", "E0b")}, [fail])}
    out, failures = hw.accepted_movers(critic, movers)
    assert out == [("transport_active", movers[0][1], "EM0b", "W0b", "E0b", False)]
    assert failures == [fail] and "EM1" not in str(out)


def test_accepted_spec_types_an_unresolvable_anchor(hw, critic_env):
    from dataclasses import replace as dc_replace
    from types import SimpleNamespace as NS
    run, _, _ = critic_env
    cs = hw._load("critic_scene")
    from manip_bridge import selection_compat as sc
    sels = sc.load_selections(run / "call2" / "selections.json")
    objs = sc.selection_objects(run / "call2" / "selections.json", sels)
    bad = {**sels, "pour": dc_replace(sels["pour"], candidate_id=999)}     # not in the pool
    em = json.loads((run / "call3_simreplay" / "emissions.json").read_text())
    from manip_bridge import vlm_compat
    ems = {r: vlm_compat.emission_from_json(e) for r, e in zip(em["roles"], em["emissions"])}
    asset = {"teapot": run / "grounding_vlm" / "canon" / "lib_teapot",
             "mug": run / "grounding_vlm" / "canon" / "lib_mug"}
    syms = {o: sc.load_symbols(d) for o, d in asset.items()}
    stages = [(NS(name=ems[r].name, active=ems[r].active, passive=ems[r].passive), r)
              for r in ("grasp", "transport_active", "pour")]
    ctx = NS(stages=stages, role_index={"grasp": 0, "transport_active": 1, "pour": 2,
                                        "transport_passive": 1},
             asset_dirs=asset, symbols=syms)
    state = NS(emissions=ems, selections=bad, objects=objs)
    spec, failures = cs.accepted_spec(ctx, state, ["transport_active", "pour"])
    assert list(spec) == ["transport_active"] and spec["transport_active"][0] is ems[
        "transport_active"]
    (f,) = failures
    assert f["stage"] == "pour" and f["phase"] == "anchor" and "999" in f["reason"]


def test_an_edited_pour_is_planned_as_edited(hw, critic_env, tmp_path, monkeypatch):
    """The pour does not compile in the preview chain (O2), and the critic
    tightens one of its rows: the planner compiles the EDITED pour (the old
    fallback handed it the original)."""
    run, setup, _ = critic_env
    setup({})
    from manip_bridge import emit_compat as ec
    seen, real = [], ec.compile_stage

    def spy(em, symbols, poses, **k):
        seen.append(em)
        return real(em, symbols, poses, **k)
    monkeypatch.setattr(ec, "compile_stage", spy)
    tighten = _verdict("reject", [{"target": f"{POUR}.path.rot[0]",
                                   "action": "tighten_tolerance", "token": "tight"}], "steady")
    assert _critic(hw, run, tmp_path, [tighten, _verdict()]) == 2
    planned = [em for em in seen if em.name == POUR][-1]
    assert planned.path_tsr.rot[0].tol == "tight"


def test_vlm_record_counts_the_calls_it_sends(tmp_path, monkeypatch):
    from manip_bridge import vlm_record
    monkeypatch.setattr(vlm_record, "_RECORDERS", [])
    t = vlm_record.RecordingTransport(lambda p: "ok")
    t({}), t({})
    vlm_record.ReplayTransport(["x"])({})
    assert vlm_record.calls_made() == 2
    path = vlm_record.write_calls_file(str(tmp_path / "calls.json"))
    assert json.loads(Path(path).read_text())["calls"] == 2
    monkeypatch.delenv("VLM_CALLS_FILE", raising=False)
    assert vlm_record.write_calls_file() is None


def test_vlm_record_keeps_a_cut_off_answer(monkeypatch):
    """A transport error carrying text (the sim's ResponseTruncated) is
    counted, its text recorded, and the error re-raised; a plain VLMError
    records nothing."""
    from manip_bridge import vlm_compat, vlm_record
    monkeypatch.setattr(vlm_record, "_RECORDERS", [])

    class Cut(vlm_compat.VLMError):
        def __init__(self, text):
            super().__init__("cut off")
            self.text = text

    def raiser(err):
        def inner(p):
            raise err
        return inner
    t = vlm_record.RecordingTransport(raiser(Cut('{"a": ')))
    with pytest.raises(Cut):
        t({})
    plain = vlm_record.RecordingTransport(raiser(vlm_compat.VLMError("API error 500")))
    with pytest.raises(vlm_compat.VLMError):
        plain({})
    assert t.raw == ['{"a": '] and plain.raw == [] and vlm_record.calls_made() == 2


def test_dry_run_writes_round_one_and_stops(hw, critic_env):
    run, setup, _ = critic_env
    side, _ = setup({})
    assert _main(hw, run, "--critic", "preview", "--critic-dry-run", *NO_GRIPPER) == 0
    rd = run / "m4_vlm" / "critic" / "round_1"
    text = (rd / "prompt.txt").read_text()
    assert "You are the critic" in text and text.count("[image: ") >= 4
    assert side.requests == [] and not (run / "m4_vlm" / "critic" / "round_2").exists()
    summ = json.loads((run / "m4_vlm" / "critic" / "critic.json").read_text())
    assert summ["status"] == "dry_run" and summ["mode"] == "dry_run" and summ["model"] is None


def test_live_mode_records_the_model(hw, critic_env, monkeypatch):
    """The paid path (no dev flag): the live transport is reached, its
    response recorded as a replayable file, and the model named."""
    from manip_bridge import vlm_compat
    run, setup, _ = critic_env
    setup({})
    sent = []
    monkeypatch.setattr(vlm_compat, "live_transport", lambda payload: sent.append(payload)
                        or _verdict())
    assert _main(hw, run, "--critic", "preview", *NO_GRIPPER) == 2           # the pour: O2
    c = _report(run)["critic"]
    assert len(sent) == 1 and c["mode"] == "live" and c["model"] == vlm_compat.MODEL
    assert c["calls"]["critic"] == 1
    assert (run / "m4_vlm" / "critic" / "responses" / "critic_01.txt").read_text() == _verdict()
    # prompt.json is the request sent: the system text, then every content
    # block in order, each image block replaced by the file it encodes
    import base64
    logged = json.loads((run / "m4_vlm" / "critic" / "round_1" / "prompt.json").read_text())
    req = sent[0]
    assert logged["system"] == req["system"]
    assert [m["role"] for m in logged["messages"]] == [m["role"] for m in req["messages"]]
    for lm, rm in zip(logged["messages"], req["messages"]):
        assert len(lm["content"]) == len(rm["content"])
        for lb, rb in zip(lm["content"], rm["content"]):
            if rb["type"] == "image":
                assert lb == {"type": "image", "path": lb["path"]}
                assert base64.standard_b64decode(rb["source"]["data"]) == \
                    Path(lb["path"]).read_bytes()
            else:
                assert lb == rb
    assert sum(b["type"] == "image" for b in logged["messages"][0]["content"]) == 5


# ------------------------------------------------------------------ captions

def _textures(run):
    """The 11/05 meshes' texture colors (median of the real textures: teal
    teapot, dark red mug), as tiny textures in the fixture's canonical dirs."""
    from PIL import Image
    for lib, rgb in (("lib_teapot", (3, 181, 145)), ("lib_mug", (122, 4, 6))):
        d = run / "grounding_vlm" / "canon" / lib / "meshes"
        d.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), rgb).save(d / f"{lib}_texture.png")


def _captions(run):
    req = json.loads((run / "m4_vlm" / "critic" / "round_1" / "prompt.json").read_text())
    blocks = req["messages"][0]["content"]
    return [blocks[i - 1]["text"] for i, b in enumerate(blocks) if b.get("type") == "image"]


def test_captions_carry_the_manifest_facts(hw, critic_env):
    run, setup, opts = critic_env
    setup({})
    _textures(run)
    opts["collisions"] = True
    assert _main(hw, run, "--critic", "preview", "--critic-dry-run", *NO_GRIPPER) == 0
    caps = _captions(run)
    assert len(caps) == 5        # initial, grasp, carry goal, carry translation, carry rotation
    initial, grasp, goal, trans, rot = caps
    assert "initial scene" in initial and "iso view; top view" in initial
    assert "grasp goal" in grasp and f'"{GRASP}"' in grasp and "role grasp" in grasp
    assert "stage goal" in goal and f'"{CARRY}" (role transport_active)' in goal
    assert "Collisions: teapot intersects mug." in goal
    assert "Collisions: none." in grasp and "Collisions:" not in initial    # initial: unchecked
    assert "path translation rows" in trans and "(top view)" in trans
    assert ("Not drawn: x -10.0 cm (free): teapot collides with mug (the planner avoids this)."
            in trans)
    assert "x free: +10.0 cm shown (top view)" in trans and "x free: -10.0 cm shown (top" not in trans
    assert "Collision checks in these images: scene objects, gripper, table (no arm)" in initial
    assert "path rotation rows" in rot and "side (+x) view" in rot
    for c in (trans, rot):
        assert ("Moving object: teapot (green). No point labels in path images; every object "
                "keeps its color from the state images.") in c
    for c in (initial, grasp, goal):
        assert "Moving object" not in c
    assert "roll at upper bound +15 deg [moderate] (side (+x) view) -- in collision with mug" in rot
    for c in caps:
        assert "Colors: teapot = green, mug = red, gripper = gray." in c


def test_captions_degrade_without_collision_fields(hw, critic_env):
    run, setup, _ = critic_env
    setup({})
    assert _main(hw, run, "--critic", "preview", "--critic-dry-run", *NO_GRIPPER) == 0
    caps = _captions(run)
    text = "\n".join(caps)
    assert "Collisions:" not in text and "Not drawn:" not in text and "in collision" not in text
    assert "roll at upper bound +15 deg [moderate]" in text
    assert all("teapot = textured, mug = textured, gripper = gray" in c for c in caps)


def test_captions_center_collision_and_text_only_groups(hw):
    """The center pose in collision keeps every panel and says so; a path
    group with every panel dropped has no image and goes into the stage
    goal's caption as text."""
    from types import SimpleNamespace as NS
    cs = hw._load("critic_scene")
    chain = [NS(role="transport_active", emission=NS(name=CARRY, active="teapot", passive="mug"))]
    hit = [{"a": "teapot", "b": "mug"}, {"a": "teapot", "b": "table"}]
    path = {"kind": "path_boundary", "stage": "transport_active", "group": "rotation",
            "file": "p.png", "center_collides": True, "representative": {"collisions": hit},
            "rows": [{"row": "roll", "view": "-y", "captions": ["roll at lower bound -15 deg"],
                      "panels": [{"desc": "roll -15 deg (lower bound)", "collides": True,
                                  "collisions": hit, "dropped": False}]}]}
    gone = {"kind": "path_boundary", "stage": "transport_active", "group": "translation",
            "rows": [{"row": "z", "view": None, "captions": ["z free: -10.0 cm shown"],
                      "panels": [{"desc": "z -10.0 cm (free)", "collides": True,
                                  "collisions": hit[:1], "dropped": True}]}]}
    goal = {"kind": "state", "state": "goal:transport_active", "role": "transport_active",
            "file": "g.png", "panels": ["iso", "top"], "collisions": []}
    m = {"images": [goal, path], "path_text_only": [gone], "gripper": None}
    (g_cap, _), (p_cap, _) = cs.views_from_manifest(m, Path("."), chain, {"teapot": "green"})
    assert "Collisions: none." in g_cap
    assert ("Path translation rows of this stage: no image, every panel collides while the "
            "center pose is free -- z -10.0 cm (free): teapot collides with mug (the planner "
            "avoids this).") in g_cap
    assert ("The center pose itself is in collision (teapot collides with mug and table)"
            in p_cap)
    assert "1. roll at lower bound -15 deg (side (-y) view) -- in collision with mug and table" \
        in p_cap
    assert "Colors: teapot = green." in p_cap and "gripper" not in p_cap
    assert "Moving object: teapot (green)." in p_cap


# ------------------------------------------------------------------ the gripper as one mesh

URDF = REPO_ROOT / "curobo_runtime" / "robot" / "ur5e_robotiq.urdf"
MESHES = REPO_ROOT / "test" / "curobo_incontainer" / "robotiq" / "meshes" / "collision"
needs_urdf = pytest.mark.skipif(not URDF.is_file() or not MESHES.is_dir(),
                                reason=f"no 2F-85 URDF at {URDF}")


@needs_urdf
def test_exported_gripper_mesh_is_the_urdf_gripper(hw, tmp_path):
    from manip_bridge import critic_compat as cc
    cs = hw._load("critic_scene")
    path = cs.export_gripper(URDF, MESHES, tmp_path / "critic_assets")
    a = cc.gripper_from_urdf(URDF, "tool0", MESHES)
    b = cs.load_gripper_mesh(path)
    np.testing.assert_array_equal(a.F, b.F)
    np.testing.assert_allclose(a.V, b.V, atol=6e-7)        # the OBJ's 1e-6 m rounding
    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["frame"].startswith("tool0") and meta["n_faces"] == len(a.F)
    assert a.source in b.source


@needs_urdf
def test_mesh_path_renders_the_same_gripper_at_the_same_pose(hw, critic_env):
    """The URDF path and the auto-detected <run>/critic_assets mesh hand the
    renderer the same gripper geometry and the same TCP pose, so its
    collision check (gripper counted with the mover) sees the same thing."""
    run, setup, opts = critic_env
    urdf = ["--gripper-urdf", str(URDF), "--gripper-meshes", str(MESHES)]
    setup({})
    assert _main(hw, run, "--critic", "preview", "--critic-dry-run", *urdf) == 0
    from_urdf = opts["rendered"]
    cs = hw._load("critic_scene")
    cs.export_gripper(URDF, MESHES, run / "critic_assets")
    setup({})
    assert _main(hw, run, "--critic", "preview", "--critic-dry-run", *NO_GRIPPER) == 0
    from_mesh = opts["rendered"]
    np.testing.assert_array_equal(from_urdf["gripper"].F, from_mesh["gripper"].F)
    np.testing.assert_allclose(from_urdf["gripper"].V, from_mesh["gripper"].V, atol=6e-7)
    assert from_urdf["grasp"].obj == from_mesh["grasp"].obj
    np.testing.assert_array_equal(from_urdf["grasp"].tcp_pose, from_mesh["grasp"].tcp_pose)
    summ = json.loads((run / "m4_vlm" / "critic" / "critic.json").read_text())
    assert any(n.startswith("gripper: mesh ") and "critic_assets" in n for n in summ["notes"])
    doc = json.loads((run / "m4_vlm" / "critic" / "round_1" / "preview_inputs.json").read_text())
    assert doc["gripper"]["kind"] == "mesh"


def test_missing_gripper_mesh_is_refused(hw, critic_env):
    run, setup, _ = critic_env
    setup({})
    assert _main(hw, run, "--critic", "preview", "--critic-dry-run",
                 "--gripper-mesh", str(run / "nope.obj")) == 2


def test_captions_carry_the_renderers_search_notes(hw):
    """A colliding goal nominal or path center: the renderer draws the
    nearest collision-free pose (or none is found) and writes search notes
    (render_preview.search_note); the captions carry them verbatim and say
    which pose the image shows."""
    from types import SimpleNamespace as NS
    from manip_bridge import critic_compat as cc
    cs = hw._load("critic_scene")
    chain = [NS(role="transport_active", emission=NS(name=CARRY, active="teapot", passive="mug"))]
    hit = [{"a": "teapot", "b": "mug", "depth_mm": 35.3}]
    free = {"nominal_collides": True, "contacts": hit, "shown": "nearest_free", "n_checked": 7,
            "delta": {"trans_cm": 2.4, "rot_deg": 0.0, "displacement": [0, 0, 0.024, 0, 0, 0]}}
    none = {**free, "shown": "none_free", "n_checked": 64, "delta": None}
    goal_note = cc._rp.search_note(free, "nominal")
    full = {**free, "pass": "full", "delta": {**free["delta"], "rot_deg": 19.1}}
    tr = {**free, "pass": "translation"}
    assert "needed rotation" in cc._rp.search_note(full, "nominal")[0]
    assert "needed rotation" not in cc._rp.search_note(tr, "nominal")[0]
    goal_note, free = cc._rp.search_note(full, "nominal"), full
    center_note = cc._rp.search_note(none, "center")
    assert goal_note[0].startswith("nominal pose collides with mug; nearest collision-free pose")
    assert center_note[0].startswith("no collision-free pose found in the region (64 samples)")
    goal = {"kind": "state", "state": "goal:transport_active", "role": "transport_active",
            "file": "g.png", "panels": ["iso", "top"], "collisions": hit, "search": free,
            "notes": goal_note}
    path = {"kind": "path_boundary", "stage": "transport_active", "group": "rotation",
            "file": "p.png", "center_collides": True,
            "notes": center_note + ["all panels kept; colliding ones are marked"],
            "representative": {"collisions": hit, "search": none},
            "rows": [{"row": "roll", "view": "-y", "captions": ["roll at lower bound -15 deg"],
                      "panels": [{"desc": "roll -15 deg (lower bound)", "collides": True,
                                  "collisions": hit, "dropped": False}]}]}
    m = {"images": [goal, path], "gripper": None}
    (g_cap, _), (p_cap, _) = cs.views_from_manifest(m, Path("."), chain, {})
    assert "at the nearest collision-free pose of its subgoal region (the region's center " \
        "collides)" in g_cap
    assert "Collisions at the subgoal center: teapot intersects mug." in g_cap
    assert f"Renderer: {goal_note[0]}." in g_cap
    assert f"Renderer: {center_note[0]}; all panels kept; colliding ones are marked." in p_cap
    assert "The center pose itself" not in p_cap                 # the renderer's line, not ours
    # a free center found: its note, and no "every panel kept" claim
    path2 = {**path, "center_collides": False,
             "notes": cc._rp.search_note(free, "center")
             + ["x -10.0 cm (free): teapot collides with mug (the planner avoids this)"],
             "rows": [{"row": "x", "view": "top", "captions": ["x free: -10.0 cm shown"],
                       "panels": [{"desc": "x -10.0 cm (free)", "collides": True,
                                   "collisions": hit, "dropped": True}]}]}
    ((cap2, _),) = cs.views_from_manifest({"images": [path2]}, Path("."), chain, {})
    assert "Renderer: center pose collides with mug; nearest collision-free pose shown" in cap2
    assert cap2.count("(the planner avoids this)") == 1 and "every panel" not in cap2


# ------------------------------------------------------------------ verdict only, the three cases

def test_verdict_only_records_the_verdict_and_does_not_plan(hw, critic_env, tmp_path):
    run, setup, _ = critic_env
    side, _ = setup({})
    assert _critic(hw, run, tmp_path, [REPICK], (), "--critic-verdict-only") == 0
    rep = _report(run)
    assert rep["status"] == "verdict_only" and side.requests == []
    c = rep["critic"]
    assert c["status"] == "verdict" and c["calls"]["critic"] == 1 and c["edits_applied"] == []
    assert c["verdict"]["verdict"] == "reject"
    assert c["verdict"]["edits"][0]["token"] == "teapot.spout_center"
    assert not (run / "m4_vlm" / "critic" / "round_2").exists()


def test_critic_cases_build_and_run(hw, critic_env, tmp_path, capsys):
    """build writes the two hand-built cases from the run's own inputs;
    run without --live prints the paid count and runs nothing; a replay
    runs each case once, verdict only, and tabulates the results."""
    run, setup, _ = critic_env
    setup({})
    cases = hw._load("critic_cases")
    repick, in_mug = cases.build_cases(run)
    s = json.loads((repick / "selections.json").read_text())
    assert s["transport_active"]["candidate_id"] == 4                    # teapot.spout_center
    base = json.loads((run / "call2" / "selections.json").read_text())
    assert {k: v for k, v in s.items() if k != "transport_active"} == \
        {k: v for k, v in base.items() if k != "transport_active"}
    em = json.loads((in_mug / "emissions.json").read_text())
    terms = em["emissions"][1]["subgoal_tsr"]["trans"]
    assert [t["term"] for t in terms] == ["centered", "below"]
    assert em["hand_built"]["emission_edit"]["term"] == ["above", "below"]
    assert "HAND-BUILT" in (in_mug / "README.md").read_text()
    # nothing runs without a mode
    assert cases.run_cases(run, tmp_path / "none", None, None, None) == 0
    assert "3 paid call(s)" in capsys.readouterr().out and not (tmp_path / "none").exists()
    # a replay: one canned verdict per case
    resp = tmp_path / "case_responses"
    for name, raw in (("handle_anchor", REPICK), ("spout_repick", _verdict()),
                      ("spout_in_mug", _verdict("reject", [{
                          "target": f"{CARRY}.subgoal.trans[1]", "action": "tighten_slack",
                          "token": "snug"}], "spout in the mug"))):
        (resp / name).mkdir(parents=True)
        (resp / name / "critic_01.txt").write_text(raw)
    out = tmp_path / "results"
    assert cases.run_cases(run, out, "replay", resp, None) == 0
    summ = json.loads((out / "summary.json").read_text())
    assert summ["model"] is None and summ["mode"] == "replay"
    got = {r["case"]: r for r in summ["cases"]}
    assert [got[c]["actual"] for c in ("handle_anchor", "spout_repick", "spout_in_mug")] == [
        "reject", "accept", "reject"]
    assert all(r["match"] for r in got.values())
    assert got["handle_anchor"]["edits"] == [
        f"reselect_point {CARRY}.point.teapot teapot.spout_center"]
    assert got["spout_in_mug"]["stages"][1]["stage"] == CARRY
    for r in got.values():
        assert r["calls"]["critic"] == 1 and r["sizes"]["images"] >= 4
        assert Path(r["request"]).is_file() and Path(r["raw_response"]).is_file()
        assert Path(r["verdict_file"]).is_file()
    # the hand-built carry compiles (no compile error in the report the critic reads)
    rpt = json.loads((out / "spout_in_mug" / "critic" / "round_1" / "report.json").read_text())
    carry = next(s for s in rpt["stages"] if s["role"] == "transport_active")
    assert "compile_error" not in carry and carry["point_choice"]["current"] == "teapot.spout_center"
    md = (out / "summary.md").read_text()
    assert "| spout_in_mug | reject | reject | yes |" in md and "expected: " in md
