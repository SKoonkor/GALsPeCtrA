"""No notebook may hard-code a merger tree — must read from galspectra.trees."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from galspectra.trees import LABELS, get_tree

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"

#: Values that identify one specific merger tree. Checked everywhere — code,
#: comments and markdown alike.
DATA_PATTERNS: dict[str, str] = {
    # ── sample files ─────────────────────────────────────────────────────
    "sample: Millennium-I": r"Planck_Mil-I_",
    "sample: Millennium-II": r"Planck_Mil-II_",
    # ── coefficient products ─────────────────────────────────────────────
    "coeffs: Millennium-I": r"lgalaxies_sed_coeffs_bc03\.npz",
    "coeffs: Millennium-II": r"lgalaxies_sed_coeffs_bc03_MRII_test1",
    # ── L-GALAXIES binary output — the class that let cell 7 through ─────
    "output pattern: Millennium-I": r"SA_DM_test3",
    "output pattern: Millennium-II": r"SA_DM_test1",
    "output subdir: Millennium-II": r"['\"]MRII['\"]\s*/|output/MRII",
    # ── volume and resolution constants ──────────────────────────────────
    "box size: Millennium-I": r"480\.279",
    "box size: Millennium-II": r"96\.0558",
    "DM particle mass: Millennium-I": r"0\.0961104",
    "DM particle mass: Millennium-II": r"0\.000768884",
    # ── file ranges ──────────────────────────────────────────────────────
    "file range: Millennium-II": r"range\(\s*40\s*,\s*80\s*\)",
    "file range: Millennium-I": r"range\(\s*5\s*,\s*6\s*\)",
}

#: Human-readable tree names, must reach the reader via TREE.pretty. Checked
#: in executable code only — comments and markdown are exempt.
DISPLAY_PATTERNS: dict[str, str] = {
    "pretty name: Millennium-I": r"Millennium-I(?!I)",
    "pretty name: Millennium-II": r"Millennium-II",
}

#: Everything, for the coverage test.
FORBIDDEN_PATTERNS: dict[str, str] = {**DATA_PATTERNS, **DISPLAY_PATTERNS}

#: The one sanctioned place a tree may be named: the selector call itself.
SELECTOR_RE = re.compile(r'get_tree\(\s*["\'](?:' + "|".join(LABELS) + r')["\']\s*\)')


def _notebooks():
    """Live notebooks only. `archive/` is frozen history and is not held to this."""
    return sorted(NOTEBOOK_DIR.glob("*.ipynb"))


def _cells(path):
    nb = json.loads(path.read_text())
    for i, cell in enumerate(nb["cells"]):
        yield i, cell["cell_type"], "".join(cell["source"])


def _strip_comments(source):
    """Blank out `#` comments, leaving line numbering intact."""
    import io
    import tokenize

    lines = source.splitlines()
    try:
        comments = [
            tok for tok in tokenize.generate_tokens(io.StringIO(source).readline)
            if tok.type == tokenize.COMMENT
        ]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return source

    for tok in comments:
        row = tok.start[0] - 1
        if 0 <= row < len(lines):
            col = tok.start[1]
            lines[row] = lines[row][:col]
    return "\n".join(lines)


def test_notebooks_exist():
    """Guard against the suite silently passing because the glob found nothing."""
    assert _notebooks(), f"no notebooks found in {NOTEBOOK_DIR}"


@pytest.mark.parametrize("path", _notebooks(), ids=lambda p: p.name)
def test_no_hard_coded_tree_data(path):
    """No cell may contain a tree-specific path, constant, or file range."""
    offences = []
    for index, kind, src in _cells(path):
        for line_no, line in enumerate(src.splitlines(), start=1):
            if SELECTOR_RE.search(line):
                continue
            for name, pattern in DATA_PATTERNS.items():
                if re.search(pattern, line):
                    offences.append(
                        f"cell {index} ({kind}) line {line_no} [{name}]: {line.strip()[:100]}")

    assert not offences, (
        f"{path.name} hard-codes tree data in {len(offences)} place(s); read it from "
        f"galspectra.trees instead:\n  " + "\n  ".join(offences)
    )


@pytest.mark.parametrize("path", _notebooks(), ids=lambda p: p.name)
def test_no_hard_coded_tree_name_in_executable_code(path):
    """A tree's display name must reach the reader via TREE.pretty (code only)."""
    offences = []
    for index, kind, src in _cells(path):
        if kind != "code":
            continue
        for line_no, line in enumerate(_strip_comments(src).splitlines(), start=1):
            if SELECTOR_RE.search(line):
                continue
            for name, pattern in DISPLAY_PATTERNS.items():
                if re.search(pattern, line):
                    offences.append(
                        f"cell {index} line {line_no} [{name}]: {line.strip()[:100]}")

    assert not offences, (
        f"{path.name} names a merger tree in executable code in {len(offences)} "
        f"place(s); use TREE.pretty:\n  " + "\n  ".join(offences)
    )


#: use the registry, or use no tree at all
_USES_TREE_RE = re.compile(r"\bTREE\b|galspectra\.trees|from galspectra import trees")


def _reads_simulation_data(path):
    return any(_USES_TREE_RE.search(src)
               for _i, kind, src in _cells(path) if kind == "code")


@pytest.mark.parametrize("path", _notebooks(), ids=lambda p: p.name)
def test_selector_is_present_and_singular(path):
    """A notebook that uses a tree selects it exactly once; one that doesn't, doesn't."""
    calls = [
        (i, line.strip())
        for i, kind, src in _cells(path)
        if kind == "code"
        for line in src.splitlines()
        if SELECTOR_RE.search(line)
    ]
    expected = 1 if _reads_simulation_data(path) else 0
    assert len(calls) == expected, (
        f"{path.name} has {len(calls)} get_tree(...) call(s) in code, expected "
        f"{expected} (it {'does' if expected else 'does not'} reference a tree): {calls}"
    )


def test_every_registry_value_is_covered():
    """Each per-tree string in the registry must be caught by some pattern."""
    uncovered = []
    for label in LABELS:
        tree = get_tree(label)
        for field in ("sample_name", "coeffs_name", "output_pattern", "pretty"):
            value = getattr(tree, field)
            probe = value.replace("{n}", "5")
            if not any(re.search(p, probe) for p in FORBIDDEN_PATTERNS.values()):
                uncovered.append(f"{label}.{field} = {value!r}")

    assert not uncovered, (
        "these registry values would not be caught if pasted into a notebook; "
        "add patterns to FORBIDDEN_PATTERNS:\n  " + "\n  ".join(uncovered)
    )


def test_pattern_list_catches_the_bug_that_motivated_it():
    """A known-bad line must be an offence under FORBIDDEN_PATTERNS."""
    line = "    fpath = LGAL_ROOT / 'output' / 'MRII' / f'SA_DM_test1_z0.00_{fnr}'"
    caught = [n for n, p in FORBIDDEN_PATTERNS.items() if re.search(p, line)]
    assert caught, "the cell-7 line is no longer caught by any pattern"
