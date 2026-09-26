"""Hardware imports sim code ONLY through manip_bridge/*_compat.py.

That rule is what makes extracting the shared modules into a library a
one-line change per compat file instead of a search through this tree, so
it is enforced here rather than left to discipline. Checked via the AST
(comments and docstrings that mention the sim repo are fine):

  - import manip_sim... / from manip_sim... import ...
  - importlib.import_module("manip_sim...") / __import__("manip_sim...")
  - sys.path.insert/append/extend(...) or site.addsitedir(...) with an
    argument naming the mount point /opt/manip-sim: a path hack into the
    sim checkout (strings elsewhere -- error messages, asserts -- are fine)

Pure static check: no sidecars, no sim checkout needed.
"""
import ast
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "__pycache__", "build", "install", "log", "node_modules", ".venv"}
MOUNT = "/opt/manip-sim"


def _is_sim_module(name):
    return name == "manip_sim" or name.startswith("manip_sim.")


def violations(source, filename="<src>"):
    """(lineno, what) for every sim reference in `source`."""
    out = []
    for node in ast.walk(ast.parse(source, filename)):
        if isinstance(node, ast.Import):
            out += [(node.lineno, f"import {a.name}") for a in node.names if _is_sim_module(a.name)]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and _is_sim_module(node.module):
                out.append((node.lineno, f"from {node.module} import ..."))
        elif isinstance(node, ast.Call) and node.args:
            f = node.func
            fname = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            a0 = node.args[0]
            if (fname in ("import_module", "__import__") and isinstance(a0, ast.Constant)
                    and isinstance(a0.value, str) and _is_sim_module(a0.value)):
                out.append((node.lineno, f"{fname}({a0.value!r})"))
            elif _is_path_hook(f) and any(_names_mount(a) for a in node.args):
                out.append((node.lineno, f"{ast.unparse(f)}(... {MOUNT} ...)"))
    return out


def _is_path_hook(f):
    """sys.path.insert/append/extend or site.addsitedir"""
    if not isinstance(f, ast.Attribute):
        return False
    v = f.value
    if f.attr in ("insert", "append", "extend"):
        return (isinstance(v, ast.Attribute) and v.attr == "path"
                and isinstance(v.value, ast.Name) and v.value.id == "sys")
    return f.attr == "addsitedir" and isinstance(v, ast.Name) and v.id == "site"


def _names_mount(arg):
    return any(isinstance(n, ast.Constant) and isinstance(n.value, str) and MOUNT in n.value
               for n in ast.walk(arg))


def _python_files():
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.endswith("_runtime")]
        for fn in filenames:
            if fn.endswith(".py"):
                yield Path(dirpath) / fn


def test_checker_catches_every_form():
    src = ("import manip_sim\n"
           "import manip_sim.perception.marks as m\n"
           "from manip_sim.perception import marks\n"
           "import importlib; importlib.import_module('manip_sim.vlm')\n"
           "__import__('manip_sim')\n"
           "import sys; sys.path.insert(0, '/opt/manip-sim')\n"
           "sys.path.append(os.path.join('/opt/manip-sim', 'scripts'))\n"
           "import site; site.addsitedir(f'/opt/manip-sim/{x}')\n")
    assert [ln for ln, _ in violations(src)] == [1, 2, 3, 4, 5, 6, 7, 8]


def test_checker_ignores_lookalikes_and_prose():
    src = ('"""Mirrors ur5e-manip-sim\'s manip_sim.grasping contract."""\n'
           "# from manip_sim import x   (a comment)\n"
           "import manip_simulator\n"
           "from .manip_sim import y\n"
           "raise ImportError('mount the sim at /opt/manip-sim')\n"
           "sys.path.insert(0, '/opt/other')\n")
    assert violations(src) == []


def test_no_sim_imports_outside_compat_modules():
    bad = []
    for p in _python_files():
        if p.name.endswith("_compat.py"):
            continue
        try:
            found = violations(p.read_text(encoding="utf-8", errors="replace"), str(p))
        except SyntaxError as e:
            bad.append(f"{p.relative_to(REPO_ROOT)}: cannot parse ({e.msg}) -- can't verify")
            continue
        bad += [f"{p.relative_to(REPO_ROOT)}:{ln}: {what}" for ln, what in found]
    assert not bad, ("sim code must be imported only through a manip_bridge/*_compat.py "
                     "module:\n  " + "\n  ".join(bad))
