"""
Canonical filter-curve loader and the named filter sets used across the project.

Three header conventions ('#'-commented — PAUS/SVO; bare integer row-count —
L-GALAXIES; none) and two wavelength units (**PAUS is nm, everything else Å**)
— getting either wrong is a wrong-by-tenths-of-a-mag bug that looks fine.
This module is the single parser; `scripts/wavelength_requirements.py` reuses
`load_curve` rather than keeping its own copy.

PAUS narrow-band throughputs are the official *total* throughput (filter ×
atmosphere × optics × CCD QE) — do not apply further correction to them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = [
    "UNIT_TO_ANGSTROM",
    "load_curve",
    "paus_narrow_band_names",
    "load_filter_set",
    "FILTER_SET_LOADERS",
    "describe_set",
]

UNIT_TO_ANGSTROM = {"AA": 1.0, "nm": 10.0, "um": 1.0e4}

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_FILTER_ROOT = _PROJECT_ROOT / "data" / "filters"
_PAUS_DIR = _FILTER_ROOT / "paus" / "OUT_FILTERS"
_SVO_DIR = _FILTER_ROOT / "svo"
_LGAL_FILTER_DIR = (
    _PROJECT_ROOT.parent / "L-GALAXIES" / "LGalaxies2020_PublicRepository-master"
    / "SpecPhotTables" / "Filters"
)


def load_curve(path, unit="AA"):
    """Read a two-column transmission curve; returns (wave_AA, transmission).

    Not renormalised to peak 1 — PAUS curves are absolute throughputs, and AB
    magnitude is invariant to an overall scaling of T anyway.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Filter curve not found: {path}")

    rows = []
    for lineno, raw in enumerate(path.read_text().splitlines()):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 1:
            # A lone integer on the first data line is the L-GALAXIES row count.
            if lineno == 0 or not rows:
                continue
            raise ValueError(f"{path}:{lineno + 1}: expected 2 columns, got 1")
        rows.append((float(parts[0]), float(parts[1])))

    if not rows:
        raise ValueError(f"{path}: no data rows")

    data = np.asarray(rows, dtype=float)
    wave = data[:, 0] * UNIT_TO_ANGSTROM[unit]
    trans = data[:, 1]

    order = np.argsort(wave)
    return wave[order], trans[order]


# ─────────────────────────────────────────────────────────────────────────────
# Named sets
# ─────────────────────────────────────────────────────────────────────────────

def paus_narrow_band_names():
    """The 40 PAUCam narrow bands, centres 455-845 nm in 10 nm steps."""
    return [f"NB{c}" for c in range(455, 855, 10)]


def _load_paus_nb():
    return {
        f"NB{c}": load_curve(_PAUS_DIR / f"AOD_D_{c}.dat", "nm")
        for c in range(455, 855, 10)
    }


def _load_paus_bb():
    return {
        f"PAU_{b}": load_curve(_PAUS_DIR / f"AOD_BBFL_{b}.txt", "nm")
        for b in ("u", "g", "r", "i", "z", "Y")
    }


def _load_megacam():
    return {
        f"MegaCam_{b}": load_curve(_SVO_DIR / f"CFHT_MegaCam.{b}.dat", "AA")
        for b in ("u", "g", "r", "i", "z")
    }


def _load_euclid():
    curves = {"VIS": load_curve(_SVO_DIR / "Euclid_VIS.vis.dat", "AA")}
    for b in ("Y", "J", "H"):
        curves[f"NISP_{b}"] = load_curve(_SVO_DIR / f"Euclid_NISP.{b}.dat", "AA")
    return curves


def _load_2mass_ks():
    return {"2MASS_Ks": load_curve(_SVO_DIR / "2MASS_2MASS.Ks.dat", "AA")}


def _load_lgal_sdss():
    """The SDSS ugriz curves L-GALAXIES itself uses, for the regression pin."""
    return {
        b: load_curve(_LGAL_FILTER_DIR / f"{b}_band_trans.dat", "AA")
        for b in ("u", "g", "r", "i", "z")
    }


FILTER_SET_LOADERS = {
    "paus_nb": _load_paus_nb,
    "paus_bb": _load_paus_bb,
    "megacam": _load_megacam,
    "euclid": _load_euclid,
    "2mass_ks": _load_2mass_ks,
    "lgal_sdss": _load_lgal_sdss,
}


def load_filter_set(*names, prefix=False):
    """Load named sets into one {band: (wave_AA, trans)} dict.

    prefix : prepend '<set>:' to band names (off by default; names are
        already unique across the current sets).
    """
    out = {}
    for name in names:
        if name not in FILTER_SET_LOADERS:
            raise ValueError(
                f"unknown filter set '{name}'. Available: {sorted(FILTER_SET_LOADERS)}"
            )
        for band, curve in FILTER_SET_LOADERS[name]().items():
            key = f"{name}:{band}" if prefix else band
            if key in out:
                raise ValueError(f"duplicate band name '{key}' across sets {names}")
            out[key] = curve
    return out


def describe_set(filters):
    """Pivot wavelength and 1%-of-peak edges for each band, for reporting."""
    rows = []
    for name, (w, t) in filters.items():
        peak = float(t.max())
        above = np.flatnonzero(t >= 0.01 * peak)
        rows.append({
            "band": name,
            "pivot_AA": float(np.sqrt(np.trapezoid(t * w, w) / np.trapezoid(t / w, w))),
            "blue_AA": float(w[above[0]]),
            "red_AA": float(w[above[-1]]),
            "peak_throughput": peak,
        })
    return rows
