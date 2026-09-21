"""Snapshot ↔ redshift mapping for the Millennium runs.

Why one snapshot has *two* redshifts
------------------------------------
The merger trees are the original Millennium ones, rescaled to a new cosmology. The
scale factors L-GALAXIES evolves the galaxies on come from ``FileWithZList`` in the
parameter file — ``input/zlists/zlist_planck_MR.txt`` for the current run. That is the
redshift a galaxy in a given snapshot actually sits at, and it is the one
``code/post_process_spec_mags.c:417`` uses for the ``n_H ∝ (1+z)^-1`` correction.
(``model_dust.c`` carries the same line, but it sits inside ``#ifndef POST_PROCESS_MAGS``
and is not compiled in this build.)

The *photometric* tables are a different matter. ``PhotPrefix`` in the parameter file is
``WMAP7``, and ``SpecPhotTables/PhotTables/`` contains only ``WMAP7_*`` files — there are
no Planck-rescaled tables in this repository. Those tables are tabulated per snapshot and
carry the observer-frame k-correction baked in, so ``ObsMag`` in the output is redshifted
on the **WMAP7** snapshot grid, which is lower than the Planck one by 0.12 to 0.26 in z.

That is not a rounding difference. At snapshot 19 the two grids differ by Δz = 0.26; using
the Planck value to redshift a synthetic spectrum and comparing it to ``ObsMag`` produces
colour residuals of a tenth of a magnitude that look like a broken k-correction. It is
also why ``scripts/validate_redshifting.py`` has to *fit* an effective redshift: its five
fitted values reproduce ``zlist_wmap7_MR.txt`` to better than 0.02.

So each :class:`Snapshot` carries both, named for what they are for:

``z_planck``
    where the galaxy is. Use for dust, distances, cosmological age, and any statement
    about the physical epoch.
``z_phot``
    the grid L-GALAXIES' own photometry was computed on. Use *only* to redshift a
    synthetic spectrum that is about to be compared against ``ObsMag``.

Do not "fix" this by setting them equal. The asymmetry is in the C code, and collapsing it
here would hide it rather than remove it. The fix is Planck-rescaled photometric tables,
which would make ``z_phot`` equal to ``z_planck`` on its own.

``z_phot`` is clamped at zero. Snapshot 58 — the z = 0 output — sits at WMAP7 z = −0.165,
because the rescaling maps the same expansion factors onto a different cosmology; the
tables extrapolate into it but a spectrum cannot be blueshifted by
``photometry.redshift.observed_frame``, which rejects z < 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "Snapshot",
    "ZLISTS",
    "redshift_table",
    "format_tag",
    "snapshot",
    "snapshots",
    "discover_tags",
]

#: ``FileWithZList`` values from the parameter files, relative to the L-GALAXIES root.
#: ``"planck"`` is the run's own list; ``"wmap7"`` is the one the ``WMAP7_*`` photometric
#: tables were built against. Both store the scale factor ``a``, not redshift — the
#: conversion at ``code/init.c:284`` is ``z = 1/a - 1``.
ZLISTS: dict[str, str] = {
    "planck": "input/zlists/zlist_planck_MR.txt",
    "wmap7": "input/zlists/zlist_wmap7_MR.txt",
}


@dataclass(frozen=True)
class Snapshot:
    """One L-GALAXIES output snapshot.

    ``tag`` is the two-decimal redshift token that appears in the output filenames.
    ``code/save.c:68`` builds them with ``%1.2f`` of the *snapshot's* redshift, not of the
    redshift that was requested in ``desired_output_redshifts.txt`` — asking for z = 2.5
    yields a file tagged ``z2.44``. The tag is therefore derived from ``z_planck`` here
    rather than stored, so the two cannot disagree.
    """

    snap: int
    z_planck: float
    z_phot: float

    @property
    def tag(self) -> str:
        return format_tag(self.z_planck)

    def __str__(self) -> str:
        return (f"snap {self.snap} (z{self.tag}): z_planck={self.z_planck:.4f}, "
                f"z_phot={self.z_phot:.4f}")


def redshift_table(lgal_root, cosmology="planck") -> list[float]:
    """Redshift per snapshot index, read from the zlist rather than transcribed.

    Returns a list indexed by snapshot number. Entries can be negative: the Planck list
    runs to z = −0.10 at snapshot 63 and the WMAP7 list turns negative at snapshot 56,
    both being the rescaled "future" of the box.
    """
    if cosmology not in ZLISTS:
        raise ValueError(
            f"unknown cosmology {cosmology!r}; expected one of {', '.join(ZLISTS)}"
        )
    path = Path(lgal_root) / ZLISTS[cosmology]
    if not path.exists():
        raise FileNotFoundError(f"zlist for {cosmology!r} not found:\n    {path}")
    a = [float(line) for line in path.read_text().split() if line.strip()]
    return [1.0 / value - 1.0 for value in a]


def format_tag(z) -> str:
    """The filename token for a redshift, matching ``%1.2f`` in ``code/save.c:68``."""
    return f"{z:.2f}"


def snapshot(lgal_root, snap) -> Snapshot:
    """The :class:`Snapshot` record for one snapshot index."""
    planck = redshift_table(lgal_root, "planck")
    wmap7 = redshift_table(lgal_root, "wmap7")
    if not 0 <= snap < len(planck):
        raise ValueError(f"snapshot {snap} is outside 0-{len(planck) - 1}")
    return Snapshot(snap=int(snap),
                    z_planck=planck[snap],
                    z_phot=max(0.0, wmap7[snap]))


def snapshots(lgal_root, tags=None, snaps=None) -> list[Snapshot]:
    """Snapshot records, selected by filename tag or by snapshot index.

    Exactly one of ``tags`` or ``snaps`` must be given. Selecting by tag is the useful
    direction in practice — a tag is what you have when you are holding a sample file —
    and it raises rather than guessing if a tag matches no snapshot, which is what a
    typo'd or stale filename looks like.

    Only non-negative-redshift snapshots are considered when matching a tag, so the
    "future" snapshots cannot capture ``z0.00``.
    """
    if (tags is None) == (snaps is None):
        raise ValueError("pass exactly one of tags= or snaps=")

    if snaps is not None:
        return [snapshot(lgal_root, s) for s in snaps]

    planck = redshift_table(lgal_root, "planck")
    by_tag: dict[str, int] = {}
    for index, z in enumerate(planck):
        if z >= 0.0:
            by_tag.setdefault(format_tag(z), index)

    missing = [t for t in tags if t not in by_tag]
    if missing:
        raise ValueError(
            f"no snapshot has redshift tag(s) {missing}. Available tags: "
            + ", ".join(sorted(by_tag, key=float))
        )
    return [snapshot(lgal_root, by_tag[t]) for t in tags]


def discover_tags(tree, lgal_root) -> list[str]:
    """Redshift tags for which a processed sample exists on disk, low z first.

    Reads the directory rather than a hard-coded list, so it stays true after a run with a
    different ``desired_output_redshifts.txt``. Returns tags, not paths: the caller turns
    one into a path with ``tree.sample_path(lgal_root, tag)``, which keeps the naming
    convention in the registry.
    """
    directory = Path(lgal_root) / "output" / "samples"
    if not directory.is_dir():
        raise FileNotFoundError(f"no sample directory:\n    {directory}")

    found = []
    for path in directory.glob("*.npy"):
        tag = tree.tag_of_sample(path)
        if tag is not None:
            found.append(tag)
    return sorted(set(found), key=float)
