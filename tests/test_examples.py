"""
The shipped examples must at least be valid, runnable-looking Python.

These examples cannot be executed here because every one of them drives real
hardware (Basel DAC, MFLI/UHFLI, QDAC, Keysight DMM) at import time. None is
behind an ``if __name__ == "__main__"`` guard, so importing one would try to
open instrument connections.

What is still checkable without hardware is that they parse and that their
imports name things Qanary actually exports. That is the class of breakage
that actually happens to examples: they are copied into, renamed around, and
never imported by anything, so nothing notices when a rename leaves them
stale.

"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).resolve().parent.parent / "examples").glob("*.py"))


def example_ids() -> list[str]:
    """Readable test ids -- the file name rather than a full path."""
    return [path.name for path in EXAMPLES]


def test_there_are_examples_to_check():
    """Guards against this whole module silently passing on an empty glob."""
    assert EXAMPLES, "no examples found -- has the directory moved?"


@pytest.mark.parametrize("path", EXAMPLES, ids=example_ids())
def test_the_example_parses(path: Path):
    """
    A syntax or indentation error makes an example useless to copy from, and
    nothing else in the suite would catch it.

    """
    source = path.read_text(encoding="utf-8")

    compile(source, str(path), "exec")


CHECKED_PACKAGES = ("qanary", "qcdrivers")


@pytest.mark.parametrize("path", EXAMPLES, ids=example_ids())
def test_what_the_example_imports_still_exists(path: Path):
    """
    Examples are the most likely thing to be left behind by a rename, since
    nothing imports them. Check every `from qanary... import X` and
    `from qcdrivers... import X` resolves.

    """
    import importlib

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    missing = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        root = node.module.split(".")[0]
        if root not in CHECKED_PACKAGES:
            continue
        try:
            module = importlib.import_module(node.module)
        except ImportError:
            missing.append(f"{node.module} (module not found)")
            continue
        for alias in node.names:
            if not hasattr(module, alias.name):
                missing.append(f"{node.module}.{alias.name}")

    assert not missing, f"{path.name} imports names that no longer exist: {missing}"
