"""marks_compat (the sim mark-set import site) and sim_provenance.

The compat tests import the real sim checkout, found at $SIM_DIR or the
sibling ../ur5e-manip-sim, and skip if it isn't there. The provenance tests
build throwaway trees, so they run anywhere git is installed.
"""
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE = REPO_ROOT / "ros2_ws" / "src" / "manip_bridge"
SIM_DIR = Path(os.environ.get("SIM_DIR", REPO_ROOT.parent / "ur5e-manip-sim")).resolve()
HAVE_SIM = (SIM_DIR / "manip_sim" / "perception" / "marks.py").is_file()
needs_sim = pytest.mark.skipif(not HAVE_SIM, reason=f"no sim checkout at {SIM_DIR} (set SIM_DIR)")
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")

sys.path.insert(0, str(BRIDGE))
from manip_bridge.sim_provenance import provenance  # noqa: E402


@pytest.fixture
def compat():
    if not HAVE_SIM:
        pytest.skip(f"no sim checkout at {SIM_DIR} (set SIM_DIR)")
    sys.path.insert(0, str(SIM_DIR))
    try:
        import manip_bridge.marks_compat as mc
        yield mc
    finally:
        sys.path.remove(str(SIM_DIR))


# ------------------------------------------------------------ marks_compat

@needs_sim
def test_contract_is_exactly_the_exported_names(compat):
    assert set(compat.__all__) == {"MIN_AREA_PX", "build_marks", "load_marks", "render_marks",
                                   "provenance", "SOURCES"}
    assert not hasattr(compat, "load_gt"), "marks.gt.json is sim-only; don't export load_gt"
    assert compat.SOURCES[0].endswith(os.path.join("manip_sim", "perception", "marks.py"))


@needs_sim
def test_build_then_load_round_trips(compat, tmp_path):
    """The sim builder, driven from hardware, writes a set the sim loader
    reads back: two same-category blobs stay two marks (the bug run_scene's
    prompt keying had), a speck below MIN_AREA_PX is dropped, and IDs come
    out in reading order regardless of input order."""
    pytest.importorskip("PIL")
    H, W = 120, 200
    rgb = np.full((H, W, 3), 200, np.uint8)
    right = np.zeros((H, W), bool); right[40:80, 130:170] = True
    left = np.zeros((H, W), bool); left[40:80, 30:70] = True
    speck = np.zeros((H, W), bool); speck[5:8, 5:8] = True
    ms = compat.build_marks(rgb, [right, speck, left], "sam", tmp_path)
    back = compat.load_marks(tmp_path)
    assert back.ids() == (1, 2)
    assert back.marks[1].bbox == (30, 40, 69, 79) and back.marks[2].bbox == (130, 40, 169, 79)
    assert back.source == "sam" and (tmp_path / "marked.png").is_file()
    assert not (tmp_path / "marks.gt.json").exists()
    assert np.array_equal(back.load_mask(1), left)
    assert ms.ids() == back.ids()


@needs_sim
def test_provenance_hashes_the_imported_file(compat):
    p = compat.provenance()
    (rel, digest), = p["files"].items()
    assert rel == os.path.join("manip_sim", "perception", "marks.py")
    assert digest == hashlib.sha256(Path(compat.SOURCES[0]).read_bytes()).hexdigest()


def test_missing_mount_fails_loudly(tmp_path):
    """No sim on the path must be an ImportError that says where to look,
    not a skipped step -- Docker mounts an empty dir for a wrong SIM_DIR."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(BRIDGE)
    r = subprocess.run([sys.executable, "-c", "import manip_bridge.marks_compat"],
                       cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r.returncode != 0
    assert "/opt/manip-sim" in r.stderr and "EMPTY dir" in r.stderr


# ----------------------------------------------------------- sim_provenance

def _git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})


def _tree(tmp_path, git):
    root = tmp_path / "sim"
    src = root / "pkg" / "mod.py"
    src.parent.mkdir(parents=True)
    src.write_text("X = 1\n")
    (root / "other.py").write_text("Y = 1\n")
    if git:
        _git(root, "init", "-q")
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "-m", "init")
    return root, src


def test_provenance_without_git_still_hashes(tmp_path):
    root, src = _tree(tmp_path, git=False)
    p = provenance((str(src),))
    assert p["root"] is None and p["commit"] is None and p["dirty"] is None
    assert p["files"] == {"mod.py": hashlib.sha256(b"X = 1\n").hexdigest()}


@needs_git
def test_provenance_clean_then_dirty_elsewhere_then_dirty_source(tmp_path):
    root, src = _tree(tmp_path, git=True)
    clean = provenance((str(src),))
    assert len(clean["commit"]) == 40 and clean["dirty"] is False and clean["dirty_sources"] == []
    assert list(clean["files"]) == [os.path.join("pkg", "mod.py")]

    (root / "other.py").write_text("Y = 2\n")            # dirty, but not in what we import
    provenance.cache_clear()
    elsewhere = provenance((str(src),))
    assert elsewhere["dirty"] is True and elsewhere["dirty_sources"] == []
    assert elsewhere["files"] == clean["files"]

    src.write_text("X = 2\n")                            # now the imported file itself
    provenance.cache_clear()
    touched = provenance((str(src),))
    assert touched["dirty_sources"] == [os.path.join("pkg", "mod.py")]
    assert touched["files"] != clean["files"]
    assert touched["commit"] == clean["commit"]
