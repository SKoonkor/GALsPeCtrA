"""Snapshot ↔ redshift mapping for the Millennium runs.

Each snapshot has two redshifts. The merger trees are rescaled to a new
cosmology; ``z_planck`` (``input/zlists/zlist_planck_MR.txt``) is where a
galaxy actually sits — use it for dust, distances, cosmological age, and the
``n_H ∝ (1+z)^-1`` correction (``code/post_process_spec_mags.c:417``). But
``SpecPhotTables/PhotTables/`` has only ``WMAP7_*`` files — no Planck-rescaled
tables exist — so ``ObsMag`` in the output is k-corrected on the **WMAP7**
grid (``z_phot``), lower than Planck by 0.12–0.26 in z. Redshifting a
synthetic spectrum with ``z_planck`` and comparing it to ``ObsMag`` produces
tenth-of-a-magnitude colour residuals that look like a broken k-correction
(``scripts/validate_redshifting.py`` fits an effective redshift for exactly
this reason, matching ``zlist_wmap7_MR.txt`` to better than 0.02).

``z_phot``: only to redshift a spectrum being compared against ``ObsMag``.
``z_planck``: everything else.

**Do not "fix" this by setting them equal** — the asymmetry is in the C code;
collapsing it here hides rather than removes it. The real fix is
Planck-rescaled photometric tables. ``z_phot`` is clamped at zero (snapshot
58, z=0, sits at WMAP7 z=−0.165 from the cosmology rescaling; a spectrum
can't be blueshifted by ``photometry.redshift.observed_frame``, which rejects
z < 0).
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

#: FileWithZList values, relative to the L-GALAXIES root. "planck" is the run's
#: own list; "wmap7" is what the WMAP7_* photometric tables were built against.
#: Both store scale factor a, not z (code/init.c:284: z = 1/a - 1).
ZLISTS: dict[str, str] = {
    "planck": "input/zlists/zlist_planck_MR.txt",
    "wmap7": "input/zlists/zlist_wmap7_MR.txt",
}


@dataclass(frozen=True)
class Snapshot:
    """One L-GALAXIES output snapshot.

    ``tag`` is the two-decimal token in output filenames (``code/save.c:68``,
    ``%1.2f`` of the snapshot's actual redshift, not the requested one — z=2.5
    can yield ``z2.44``). Derived from z_planck, not stored, so they can't disagree.
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
    """Redshift per snapshot index, read from the zlist (not transcribed).

    Can be negative: Planck runs to z=−0.10 at snapshot 63, WMAP7 turns
    negative at snapshot 56 — both the rescaled "future" of the box.
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

    Exactly one of tags/snaps required. Selecting by tag raises rather than
    guessing on no match. Only non-negative-redshift snapshots are matched
    (so a "future" snapshot can't capture z0.00).
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
    """Redshift tags with a processed sample on disk, low z first.

    Reads the directory (not a hard-coded list). Returns tags, not paths —
    the caller builds a path with tree.sample_path(lgal_root, tag).
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
