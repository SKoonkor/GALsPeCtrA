"""
Per-simulation constants for the Millennium merger trees.

Owns only values that differ between Millennium-I and Millennium-II: sample
file, coefficient product, box size, file range, DM particle mass,
stellar-mass cut, and the L-GALAXIES output path.

Deliberately does **not** own:
* ``hubble_h = 0.673`` — identical for both trees; it's cosmology, not tree
  identity (has six live homes today — see ``cosmology.py`` for the tally).
* ``PART_RES = 20.0`` — a resolution criterion, also the same for both.
* Simulation detection from a filename — that's
  ``scripts/process_lgalaxies.py`` (``detect_simulation``); this module maps
  *label to constants*, that one maps *filename to label*. They compose:
  ``get_tree(detect_simulation(path))``.
* The snapshot ↔ redshift mapping — that's ``lgalaxies.snapshots``. A
  redshift tag is not tree identity (both trees use the same tags); this
  module only knows how to spell one into a filename.

The three name fields describe z = 0, because that's what every caller wants
and a default spelled out at every call site is a default nobody gets right.
Path methods take an optional ``tag`` (the two-decimal token from
``code/save.c:68``'s ``%1.2f``, e.g. ``z0.26``) that rewrites the z = 0 name;
``tag=None`` and ``tag="0.00"`` are the same redshift and normalise to the
same paths (asserted in tests, not assumed).

Labels are ``"MR"``/``"MRII"`` with **no aliases** — one spelling per tree,
anything else raises. (A second accepted spelling is how ``"MR"`` and
``"MRI"`` came to mean the same simulation in different files.)
``Tree.pretty`` supplies the human-readable name for prose and figures.

Why this exists: four notebooks carried a byte-identical copy of this table,
and the sample/coefficient mismatch was exactly the bug
``process_lgalaxies.py`` had in August 2026. But a registry is a machine for
supplying defaults, and that bug's fix was to *remove* defaults — so
``coeffs_name`` is the *expected* name a cross-check tests against, never a
value that fills in ``--output`` (``tests/test_process_lgalaxies_paths.py``
pins the distinction).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Tree", "TREES", "LABELS", "canonical_label", "get_tree"]


@dataclass(frozen=True)
class Tree:
    """Constants for one Millennium merger tree.

    Paths are stored as *names*, not absolute paths, so this module has no
    opinion about where the L-GALAXIES tree or the data directory live. The
    caller supplies those roots.
    """

    label: str                  # "MR" or "MRII"
    pretty: str                 # "Millennium-I" / "Millennium-II"

    # ── data products ────────────────────────────────────────────────────
    sample_name: str            # .npy under <LGAL_ROOT>/output/samples/
    coeffs_name: str            # .npz under <GALSPECTRA_ROOT>/data/

    # ── volume and resolution ────────────────────────────────────────────
    box_mpc_h: float            # comoving box side [Mpc/h]
    n_total_files: int          # tree files the full simulation is split into
    dm_particle_msun_h: float   # DM particle mass [Msun/h]
    sm_resolution_msun: float   # stellar-mass cut used for selection [Msun]

    # ── L-GALAXIES binary output ─────────────────────────────────────────
    output_subdir: str          # relative to <LGAL_ROOT>
    output_pattern: str         # .format(n=<file number>)
    file_range: range           # the tree files actually available

    # ── derived ──────────────────────────────────────────────────────────

    @property
    def n_files_used(self) -> int:
        """How many tree files are in hand.

        Derived from file_range, not stored, so the two can't drift apart —
        the same class of mistake as storing a sample path and output path
        independently used to be.
        """
        return len(self.file_range)

    @property
    def effective_volume_mpc_h3(self) -> float:
        """Comoving volume sampled: box^3 * n_files_used / n_total_files, (Mpc/h)^3.

        **Assumes the tree files partition the box into equal volumes** — the
        standard Millennium split, but unverified; if unequal or spatially
        clustered, this is wrong. Notebook 08's LF amplitude is
        counts/effective_volume, so an error here scales every point by the
        same factor (invisible in shape) — and Paper III's faint-end excess is
        exactly an amplitude claim. Irrelevant when n_files_used == n_total_files.
        """
        return self.box_mpc_h ** 3 * self.n_files_used / self.n_total_files

    # ── paths ────────────────────────────────────────────────────────────

    #: The z = 0 token the three name fields are written around; substituting it
    #: produces a tagged name. The optional second half is main_lgals.py's
    #: ``z<first>-<last>`` range (a single-redshift run repeats the same value) —
    #: both halves must move together, which is why this is a regex: missing the
    #: second half produced ``z0.26-0.00``, a path that cannot exist.
    _Z0_RE = re.compile(r"z0\.00(-0\.00)?")

    @staticmethod
    def _check_tag(tag):
        """Validate a redshift tag, or pass None through as "z = 0".

        Exactly two decimals (code/save.c:68 writes ``%1.2f``) — accepting
        "1.0" or 1.04 would build a path that can't exist, failing several
        frames away as a confusing FileNotFoundError instead of here.
        """
        if tag is None:
            return None
        if not isinstance(tag, str) or not re.fullmatch(r"\d+\.\d\d", tag):
            raise ValueError(
                f"redshift tag must be a string of the form '1.04' (two decimals, "
                f"as code/save.c writes them), got {tag!r}"
            )
        # "0.00" and None must produce the same paths, or every caller has to
        # write `None if tag == "0.00" else tag` to paper over the difference
        return None if tag == "0.00" else tag

    def _retag(self, name, tag) -> str:
        """Rewrite a z = 0 product name for another output redshift."""
        tag = self._check_tag(tag)
        if tag is None:
            return name
        retagged, n = self._Z0_RE.subn(
            lambda m: f"z{tag}-{tag}" if m.group(1) else f"z{tag}", name)
        if n:
            return retagged
        # no z-token: coefficient products are named per simulation, not per
        # redshift, so append the tag before the suffix instead
        stem, dot, suffix = name.rpartition(".")
        return f"{stem}_z{tag}{dot}{suffix}"

    def sample_path(self, lgal_root, tag=None) -> Path:
        """The pre-processed galaxy sample .npy for this tree at one output redshift.

        tag=None is z = 0 (what every pre-multi-redshift caller means).
        """
        return (Path(lgal_root) / "output" / "samples"
                / self._retag(self.sample_name, tag))

    def tag_of_sample(self, path):
        """The redshift tag of a sample filename, or None if it isn't this tree's.

        Inverse of sample_path, deliberately strict: matches the whole
        filename, so a Millennium-II sample in the same directory returns
        None rather than a tag used to build a wrong Millennium-I path. The
        range's two halves are tied by backreference for the same reason —
        z0.51-2.07 (a multi-redshift sample) is never produced or interpreted.
        """
        token = self._Z0_RE.search(self.sample_name)
        if token is None:
            return None
        marker = "\x00"
        pattern = re.escape(self._Z0_RE.sub(marker, self.sample_name)).replace(
            re.escape(marker),
            r"z(?P<tag>\d+\.\d\d)-(?P=tag)" if token.group(1) else r"z(?P<tag>\d+\.\d\d)",
        )
        match = re.fullmatch(pattern, Path(path).name)
        return match.group("tag") if match else None

    def coeffs_path(self, data_dir, tag=None) -> Path:
        """The post-processed PCA coefficient product for this tree.

        Read-only by intention: the *expected* name, to check that a named
        output refers to the right simulation — never a default that fills
        in a write target. See the module docstring.
        """
        return Path(data_dir) / self._retag(self.coeffs_name, tag)

    def file_numbers(self, file_nr=None) -> list[int]:
        """Which tree files a request covers. None means all of file_range.

        Split out of output_paths because callers need numbers *and* paths in
        the same order — deriving one a second time is how they drift apart
        (notebook 09 zipped paths against a file-number list this registry's
        refactor had already deleted, and died on NameError). A number outside
        the range raises rather than silently building a path that can't
        exist — the failure mode that let a hard-coded Millennium-II path
        survive inside a Millennium-I notebook.
        """
        nrs = [file_nr] if file_nr is not None else list(self.file_range)
        outside = [n for n in nrs if n not in self.file_range]
        if outside:
            raise ValueError(
                f"file number(s) {outside} are outside the {self.pretty} range "
                f"{self.file_range.start}-{self.file_range.stop - 1}"
            )
        return nrs

    def output_paths(self, lgal_root, file_nr=None, tag=None) -> list[Path]:
        """L-GALAXIES binary output file(s), in :meth:`file_numbers` order.

        ``file_nr=None`` returns every file in ``file_range``.
        """
        base = Path(lgal_root) / self.output_subdir
        pattern = self._retag(self.output_pattern, tag)
        return [base / pattern.format(n=n) for n in self.file_numbers(file_nr)]

    def require(self, path, what):
        """Return path, or raise naming what is missing and for which tree.

        Never falls back to the other tree — that would mean every number
        and figure downstream quietly described a different simulation.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"tree {self.label!r} ({self.pretty}) needs {what}, which is not "
                f"present:\n    {p}\n"
                f"Millennium-II data lives on an external drive. Mount it, or "
                f"select Millennium-I with get_tree('MR')."
            )
        return p


TREES: dict[str, Tree] = {
    "MR": Tree(
        label="MR",
        pretty="Millennium-I",
        sample_name="Planck_Mil-I_snapshots_default_test3_z0.00-0.00_All.npy",
        coeffs_name="lgalaxies_sed_coeffs_bc03.npz",
        box_mpc_h=480.279,
        n_total_files=512,
        dm_particle_msun_h=0.0961104e10,
        sm_resolution_msun=1e9,
        output_subdir="output",
        output_pattern="SA_DM_test3_z0.00_{n}",
        file_range=range(5, 6),          # FirstFile=5, LastFile=5
    ),
    "MRII": Tree(
        label="MRII",
        pretty="Millennium-II",
        sample_name="Planck_Mil-II_snapshots_default_test1_z0.00-0.00_All.npy",
        coeffs_name="lgalaxies_sed_coeffs_bc03_MRII_test1.npz",
        box_mpc_h=96.0558,
        n_total_files=512,
        dm_particle_msun_h=0.000768884e10,
        sm_resolution_msun=1e8,
        output_subdir="output/MRII",
        output_pattern="SA_DM_test1_z0.00_{n}",
        file_range=range(40, 80),
    ),
}

#: The only accepted labels, in the same order and spelling as
#: ``process_lgalaxies.py``'s ``--simulation`` choices.
LABELS: tuple[str, ...] = tuple(TREES)


def canonical_label(name) -> str:
    """Validate a tree label. Exact match against LABELS — no aliases, no case folding."""
    if name not in LABELS:
        raise ValueError(
            f"unknown merger tree {name!r}. Accepted labels: {', '.join(LABELS)}"
        )
    return name


def get_tree(name) -> Tree:
    """The :class:`Tree` for a label. Raises on anything not in :data:`LABELS`."""
    return TREES[canonical_label(name)]
