#!/usr/bin/env python
"""
Standalone reproduction for the L-GALAXIES photometric-table issue.

Deliberately depends on **nothing but numpy and the L-GALAXIES distribution**, so a
maintainer can run it without installing GALsPeCtrA. It duplicates a little logic that
lives in `galspectra` elsewhere in this repository; that duplication is the point.

What it does
------------
Reads one BC03 `FullSED` file and the SDSS r filter that ship with L-GALAXIES, computes
the r-band AB magnitude of every SSP age two ways, and compares both against the
`PhotTables` entry for the same SSP:

  1. a textbook photon-counting integral,   <f_nu> = int f_nu T dnu/nu / int T dnu/nu
  2. the quadrature in setup_Spec_LumTables_onthefly(), reproduced exactly

Only relative differences matter, so a single global zero point is removed before
comparing; the two products carry different flux normalisations and that is one constant.

Usage
  python upstream_repro_quadrature.py /path/to/LGalaxies2020_PublicRepository-master
"""

import sys
from pathlib import Path

import numpy as np

C_ANG_S = 2.99792458e18          # speed of light, Angstrom/s
METALLICITY = 0.02               # solar; any of the six works
BAND_FILE = "r_band_trans.dat"   # as named in input/Filter_Names.txt
BAND_TOKEN = "rs"


def read_fullsed(path):
    """(ages_yr, wave_AA, flux) from a BC03 FullSED file; flux is f_lambda."""
    raw = np.loadtxt(path)
    wave_all = raw[:, 2]
    n_wave = int(np.argmax(wave_all[1:] < wave_all[:-1]) + 1)
    n_age = raw.shape[0] // n_wave
    ages = raw[:, 0].reshape(n_age, n_wave)[:, 0]
    wave = wave_all[:n_wave]
    flux = raw[:, 3].reshape(n_age, n_wave)
    return ages, wave, flux


def read_phot_table(path):
    """(ages_yr, magnitudes) for the z ~ 0 snapshot of a PhotTable."""
    tok = np.array(path.read_text().split())
    n_snap, n_age = int(tok[2]), int(tok[3])
    ages = tok[4:4 + n_age].astype(float)
    block = tok[4 + n_age:].astype(float).reshape(n_snap, n_age + 1)
    i0 = int(np.argmin(np.abs(block[:, 0])))
    return ages, block[i0, 1:]


# ── the two integrals ────────────────────────────────────────────────────────

def mag_photon_counting(wave, f_lam, filt_wave, filt_trans):
    """<f_nu> weighted by T dnu/nu, on the SSP grid, trapezoidal."""
    f_nu = f_lam * wave ** 2 / C_ANG_S
    nu = C_ANG_S / wave
    T = np.interp(wave, filt_wave, filt_trans, left=0.0, right=0.0)

    order = np.argsort(nu)
    nu_s, fnu_s, T_s = nu[order], f_nu[order], T[order]
    w = T_s / nu_s
    return -2.5 * np.log10(np.trapezoid(fnu_s * w, nu_s) / np.trapezoid(w, nu_s))


def _locate(xx, n, x):
    """Numerical Recipes locate(), as called by interpolate()."""
    jl, ju = 0, n + 1
    ascnd = xx[n] >= xx[1]
    while ju - jl > 1:
        jm = (ju + jl) >> 1
        if (x >= xx[jm]) == ascnd:
            jl = jm
        else:
            ju = jm
    if x == xx[1]:
        return 1
    if x == xx[n]:
        return n - 1
    return jl


def _lgal_integrate(f):
    """integrate() from model_misc.c — Simpson-like weights, no step size."""
    n = f.size
    odd = sum(f[2 * i + 1] for i in range(max(0, n // 2 - 2)))
    even = sum(f[2 * i] for i in range(max(0, n // 2 - 1)))
    return (f[0] + f[n - 1]) / 3.0 + odd * 2.0 / 3.0 + even * 4.0 / 3.0


def mag_lgalaxies(wave, f_lam, filt_wave, filt_trans):
    """setup_Spec_LumTables_onthefly()'s quadrature, reproduced."""
    f_nu = f_lam * wave ** 2 / C_ANG_S

    # create_grid(): the filter's span, the spectrum's binning
    sel = (wave >= filt_wave[0]) & (wave <= filt_wave[-1])
    grid, sed = wave[sel], f_nu[sel]

    # interpolate(): a step sample, not an interpolation
    T = np.zeros(grid.size)
    for i, x in enumerate(grid):
        if filt_wave[0] <= x <= filt_wave[-1]:
            nn = _locate(filt_wave, filt_wave.size - 1, x)
            T[i] = filt_trans[min(max(nn, 1), filt_wave.size - 1)]

    # the average is taken in dlambda, and integrate() supplies no step size
    return -2.5 * np.log10(_lgal_integrate(sed * T) / _lgal_integrate(T))


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    root = Path(sys.argv[1])

    sed_file = root / "SpecPhotTables/FullSEDs" / f"BC03_Chabrier_FullSED_m{METALLICITY:.4f}.dat"
    filt_file = root / "SpecPhotTables/Filters" / BAND_FILE
    tab_file = (root / "SpecPhotTables/PhotTables"
                / f"BC03_Chabrier_Phot_Table_MR_Mag{BAND_TOKEN}_m{METALLICITY:.4f}.dat")
    for p in (sed_file, filt_file, tab_file):
        if not p.exists():
            sys.exit(f"missing: {p}")

    ages, wave, flux = read_fullsed(sed_file)
    # Filter files open with a line giving the number of points, then two columns.
    filt = np.loadtxt(filt_file, skiprows=1)
    fw, ft = filt[:, 0], filt[:, 1]
    tab_ages, tab_mags = read_phot_table(tab_file)

    assert np.allclose(ages, tab_ages, rtol=1e-9), "age grids differ"
    print(f"{len(ages)} ages, {wave.size} wavelengths, Z = {METALLICITY}")
    print(f"age grids agree to {np.max(np.abs(ages - tab_ages) / np.maximum(tab_ages, 1)):.1e}\n")

    ok = np.isfinite(tab_mags) & (ages > 1e6)
    mine = np.array([mag_photon_counting(wave, flux[i], fw, ft) for i in range(len(ages))])
    theirs = np.array([mag_lgalaxies(wave, flux[i], fw, ft) for i in range(len(ages))])

    # remove one global zero point from each
    d_mine = mine - tab_mags
    d_theirs = theirs - tab_mags
    r_mine = np.abs(d_mine - np.median(d_mine[ok]))
    r_theirs = np.abs(d_theirs - np.median(d_theirs[ok]))

    print(f"residual against PhotTables, SDSS {BAND_TOKEN}:")
    print(f"  textbook photon-counting integral : median {np.median(r_mine[ok]):.6f} mag"
          f"   max {r_mine[ok].max():.6f}")
    print(f"  reproducing the C quadrature      : median {np.median(r_theirs[ok]):.6f} mag"
          f"   max {r_theirs[ok].max():.6f}")

    print(f"\n{'age / Gyr':>12}{'table':>12}{'photon':>12}{'C quadrature':>15}")
    for i in np.searchsorted(ages, np.array([1e7, 1e8, 1e9, 5e9, 1.2e10])):
        i = min(i, len(ages) - 1)
        print(f"{ages[i] / 1e9:12.4f}{tab_mags[i]:12.4f}"
              f"{mine[i] - np.median(d_mine[ok]):12.4f}"
              f"{theirs[i] - np.median(d_theirs[ok]):15.4f}")


if __name__ == "__main__":
    main()
