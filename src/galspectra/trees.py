"""
Per-simulation constants for the Millennium merger trees.

Scope, stated narrowly on purpose
---------------------------------
This module owns **only values that differ between Millennium-I and
Millennium-II**: which sample file, which coefficient product, the box size, the
file range, the DM particle mass, the stellar-mass cut, and where L-GALAXIES
writes its binary output.

It deliberately does **not** own:

* ``hubble_h = 0.673`` — identical for both trees. It is cosmology, not tree
  identity. It has six live homes today (``lgalaxies/sfh.py``,
  ``photometry/dust.py``, ``process_lgalaxies.py``, ``build_galaxy_table.py``,
  ``verify_pca_onthefly.py``, and the notebooks) and deserves a single home of
  its own — but putting it here would assert a dependence that does not exist.
* ``PART_RES = 20.0`` — a resolution criterion in particles, the same for both.
* **Simulation detection from a filename.** That is
  ``scripts/process_lgalaxies.py`` (``_SIM_PATTERNS`` / ``detect_simulation``)
  and it stays there. This module maps *label to constants*; that one maps
  *filename to label*. They compose: ``get_tree(detect_simulation(path))``.
* **The snapshot ↔ redshift mapping.** That is
  ``galspectra.lgalaxies.snapshots``, which reads it out of the zlist files.
  A redshift tag is not tree identity — both trees use the same tags — so this
  module only knows how to *spell* one into a filename.

Redshift tags
-------------
The three name fields describe the **z = 0** products, because that is what every
existing caller wants and a default that has to be spelled out at each call site
is a default nobody gets right. A run with more than one output redshift produces
one sample and one coefficient file per snapshot, distinguished by the two-decimal
tag in ``code/save.c:68``'s ``%1.2f`` — ``z0.26``, ``z1.04``. The path methods take
an optional ``tag`` that rewrites the z = 0 name into the tagged one, so the naming
convention stays written down once. ``tag=None`` and ``tag="0.00"`` are the same
redshift and are normalised to the same paths, which is asserted in the tests rather
than assumed.

Labels
------
``"MR"`` and ``"MRII"``, matching ``process_lgalaxies.py``'s ``--simulation``
choices and its test suite. There are deliberately **no aliases**: one spelling
per tree, and anything else raises. ``Tree.pretty`` supplies the human-readable
name ("Millennium-I") for prose, headings and figure titles, so display text
never has to hard-code which tree is in play.

Why this exists
---------------
Four notebooks carried a byte-identical copy of this table. A registry makes the
sample file and the coefficient product come from one selection, so they cannot
disagree — which is exactly the inconsistency behind the
``process_lgalaxies.py`` sample-path bug fixed in August 2026.

That cuts both ways, and the boundary matters: a registry is a machine for
supplying defaults, and the fix for that bug was to *remove* defaults.
``coeffs_name`` is therefore the *expected* name a cross-check tests against
(``assert_same_simulation``), never a value that fills in ``--output``.
``tests/test_process_lgalaxies_paths.py`` pins that distinction.
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

        Derived from ``file_range`` rather than stored, so the count and the
        range cannot drift apart. Storing both was the earlier design and it is
        the same class of mistake as storing a sample path and an output path
        independently.
        """
        return len(self.file_range)

    @property
    def effective_volume_mpc_h3(self) -> float:
        """Comoving volume sampled by the files in hand, in (Mpc/h)^3.

        ``box^3 * n_files_used / n_total_files``.

        **This assumes the tree files partition the box into equal volumes.**
        That is the standard Millennium file split, but it is an assumption, not
        a measurement: the files are a spatial decomposition, and if they were
        unequal — or if the available subset were spatially clustered rather
        than a fair sample — this would be wrong.

        It matters. Notebook 08's luminosity-function **amplitude** is
        ``counts / effective_volume``, so an error here scales every LF point by
        the same factor and would be invisible in the shape. The faint-end
        excess Paper III investigates is a statement about amplitude at the faint
        end, so a systematic volume error would land directly on the result.

        With ``n_files_used == n_total_files`` the assumption is irrelevant —
        the whole box is the whole box.
        """
        return self.box_mpc_h ** 3 * self.n_files_used / self.n_total_files

    # ── paths ────────────────────────────────────────────────────────────

    #: The z = 0 token the three name fields are written around. Substituting it is
    #: how a tagged name is produced, so the convention lives in the names above and
    #: not in a second format string here.
    #:
    #: The optional second half is a *range*: ``main_lgals.py`` names each sample
    #: ``z<first>-<last>`` from its ``FullRedshiftList``, and a single-redshift run —
    #: the only kind snapshot mode supports — repeats the same value. Both halves have
    #: to move together, which is why this is a regex and not a plain string. Missing
    #: the second half produced ``z0.26-0.00``, a path that cannot exist.
    _Z0_RE = re.compile(r"z0\.00(-0\.00)?")

    @staticmethod
    def _check_tag(tag):
        """Validate a redshift tag, or pass ``None`` through as "z = 0".

        Tags are exactly two decimals because ``code/save.c:68`` writes them with
        ``%1.2f``. Accepting ``"1.0"`` or ``1.04`` would build a path that cannot
        exist, and the failure would surface as a confusing FileNotFoundError several
        frames away rather than here.
        """
        if tag is None:
            return None
        if not isinstance(tag, str) or not re.fullmatch(r"\d+\.\d\d", tag):
            raise ValueError(
                f"redshift tag must be a string of the form '1.04' (two decimals, "
                f"as code/save.c writes them), got {tag!r}"
            )
        # "0.00" and None are the same redshift, so they must produce the same paths.
        # Without this the sample and binary names agreed by luck — they contain a
        # literal z0.00 — while the coefficient name did not, and every caller had to
        # write `None if tag == "0.00" else tag` to paper over the difference.
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
        # No z-token to replace: the coefficient products are named per simulation,
        # not per redshift, so the tag is appended before the suffix instead.
        stem, dot, suffix = name.rpartition(".")
        return f"{stem}_z{tag}{dot}{suffix}"

    def sample_path(self, lgal_root, tag=None) -> Path:
        """The pre-processed galaxy sample .npy for this tree at one output redshift.

        ``tag=None`` is z = 0, which is what every caller written before the
        multi-redshift run means.
        """
        return (Path(lgal_root) / "output" / "samples"
                / self._retag(self.sample_name, tag))

    def tag_of_sample(self, path):
        """The redshift tag of a sample filename, or ``None`` if it is not this tree's.

        The inverse of :meth:`sample_path`, and deliberately strict: it matches the
        whole filename against this tree's own naming convention, so a Millennium-II
        sample sitting in the same directory returns ``None`` rather than a tag that
        would then be used to build a Millennium-I path. The two halves of the range
        are tied together with a backreference for the same reason — ``z0.51-2.07`` is
        a multi-redshift sample this code has never produced and cannot interpret.
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

        Read-only by intention. This is the *expected* name, used to check that
        an output the user named refers to the right simulation; it is not a
        default that fills in a write target. See the module docstring.
        """
        return Path(data_dir) / self._retag(self.coeffs_name, tag)

    def file_numbers(self, file_nr=None) -> list[int]:
        """Which tree files a request covers. ``None`` means all of ``file_range``.

        Split out of :meth:`output_paths` because callers need the numbers *and* the
        paths, in the same order, and deriving one of them a second time is how they
        come apart. Notebook 09 zipped the paths against a file-number list that the
        refactor to this registry had already deleted, and every cell below it died on
        the resulting ``NameError``.

        A number outside the range raises rather than silently producing a path that
        cannot exist — the failure mode that let a hard-coded Millennium-II path
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
        """Return ``path``, or raise naming what is missing and for which tree.

        Never falls back to the other tree. A silent fallback would mean every
        number and figure downstream quietly described a different simulation
        from the one named in the heading.
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
    """Validate a tree label. Exact match against :data:`LABELS`.

    There is no alias handling and no case folding: one spelling per tree. A
    second accepted spelling is how ``"MR"`` and ``"MRI"`` came to mean the same
    simulation in different files, which is the kind of drift this module exists
    to remove.
    """
    if name not in LABELS:
        raise ValueError(
            f"unknown merger tree {name!r}. Accepted labels: {', '.join(LABELS)}"
        )
    return name


def get_tree(name) -> Tree:
    """The :class:`Tree` for a label. Raises on anything not in :data:`LABELS`."""
    return TREES[canonical_label(name)]
