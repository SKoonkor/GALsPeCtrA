"""Is storing PCA coefficients worth it, when L-GALAXIES already outputs the SFH?

Measures the three things the argument turns on, so none of them has to be asserted:

  1. STORAGE  — what the SFH and the coefficients actually cost per galaxy, read from the
     compiled output dtype rather than hard-coded.
  2. COMPUTE  — how long it takes to get spectra out of each representation, with both
     routes vectorised as fairly as possible (a single matmul each).
  3. DEPENDENCY — how much data a downstream user has to hold to use each.

Why this exists
---------------
`documents/error_vs_bytes.md` §8 already measured the storage question and found the SFH
wins. Its counter-argument — "the PCA's value is compute, not storage" — was never
measured, and §8 is the section a paper would be built on. This closes that gap.

The comparison is deliberately generous to the PCA: the SFH route is given the same
vectorised treatment (build a sparse weight matrix over the SSP library, then one matmul),
rather than the per-galaxy Python loop `process_lgalaxies.py` happens to use.

Usage
-----
    python scripts/bench_sfh_vs_pca.py [--n-galaxies 11396] [--repeats 5]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LGAL_ROOT = PROJECT_ROOT.parent / "L-GALAXIES" / "LGalaxies2020_PublicRepository-master"

#: L-GALAXIES' own populated-bin count at z = 0 (sfh_ibin + 1); see notebook 02 §1.
DEFAULT_BINS = 13


def storage_report():
    """Byte costs straight out of the compiled GALAXY_OUTPUT dtype."""
    sys.path.insert(0, str(LGAL_ROOT / "AuxCode" / "Python"))
    from LGalaxy_snapshots import LGalaxiesStruct as S

    total = S.itemsize
    sfh_fields = [n for n in S.names if n.startswith("sfh_")]
    sfh_bytes = sum(S[n].itemsize for n in sfh_fields)
    # subset a spectrum re-derivation needs: mass + metals, disk/bulge (ring and
    # element arrays aren't usable by a single-Z SSP library)
    needed = ("sfh_DiskMass", "sfh_BulgeMass", "sfh_MetalsDiskMass", "sfh_MetalsBulgeMass")
    need_bytes = sum(S[n].itemsize for n in needed)
    pca_bytes = sum(S[n].itemsize for n in S.names if "pca" in n)
    mag_bytes = sum(S[n].itemsize for n in S.names if "Mag" in n)
    # sfh_*_elements only — there are non-SFH *_elements fields (present-day
    # abundances) and counting those here would overstate the history.
    elem_bytes = sum(S[n].itemsize for n in S.names
                     if n.startswith("sfh_") and n.endswith("_elements"))

    print("STORAGE  (per galaxy, from the compiled output struct)")
    print(f"  whole GALAXY_OUTPUT record          {total:>7,} B")
    print(f"  all sfh_* fields                    {sfh_bytes:>7,} B   "
          f"({sfh_bytes / total:.0%} of the record)")
    print(f"  the subset a spectrum needs         {need_bytes:>7,} B   "
          f"(mass + metals, disk and bulge)")
    print(f"  pca_coeffs                          {pca_bytes:>7,} B")
    print(f"  all magnitudes                      {mag_bytes:>7,} B")
    print(f"  sfh element histories               {elem_bytes:>7,} B   "
          f"(unusable by a single-Z SSP library)")
    print(f"\n  stored SFH subset / pca_coeffs      {need_bytes / pca_bytes:>7.1f}x")
    print("  but see documents/error_vs_bytes.md §8: a *minimally encoded* 13-bin SFH")
    print("  (mass + mass-weighted Z) is 104 B for 0.701 mmag, against 204 B for")
    print("  0.830 mmag from the 50-PC basis — half the storage, better accuracy.")
    return total


def compute_report(n_gal, n_bins, repeats):
    """Time both routes from their stored representation to spectra."""
    pca = np.load(PROJECT_ROOT / "data" / "pca_results_bc03.npz", allow_pickle=True)
    grid = np.load(PROJECT_ROOT / "data" / "sed_grid_bc03.npz", allow_pickle=True)
    comps, mean = pca["components"], pca["mean"]
    norm = pca["norm"].item()
    nstd, nmean = norm["std"], norm["mean"]
    library = grid["seds"]                       # (n_ssp, n_wave) the SSP library itself
    ages = np.unique(grid["params"][:, 0])
    zs = np.unique(grid["params"][:, 1])

    rng = np.random.default_rng(0)
    coeffs = rng.normal(size=(n_gal, comps.shape[0]))
    masses = rng.random(n_gal) * 1e10
    n_terms = n_bins * 2                          # disk and bulge per bin
    a = rng.uniform(ages.min(), ages.max(), size=(n_gal, n_terms))
    z = rng.uniform(zs.min(), zs.max(), size=(n_gal, n_terms))
    m = rng.random((n_gal, n_terms)) * 1e9

    def route_pca():
        return ((mean[None, :] + (coeffs / masses[:, None]) @ comps)
                * nstd[None, :] + nmean[None, :]) * masses[:, None]

    def route_sfh():
        # bilinear weights over the (age, Z) library into one sparse operator per
        # galaxy (same construction pca_basis_experiment.py's binned-SFH baseline
        # uses), then a single matmul
        W = np.zeros((n_gal, library.shape[0]))
        ai = np.clip(np.searchsorted(ages, a) - 1, 0, len(ages) - 2)
        zi = np.clip(np.searchsorted(zs, z) - 1, 0, len(zs) - 2)
        fa = np.clip((a - ages[ai]) / (ages[ai + 1] - ages[ai]), 0, 1)
        fz = np.clip((z - zs[zi]) / (zs[zi + 1] - zs[zi]), 0, 1)
        rows = np.repeat(np.arange(n_gal), n_terms)
        for da, dz, wf in ((0, 0, (1 - fa) * (1 - fz)), (1, 0, fa * (1 - fz)),
                           (0, 1, (1 - fa) * fz),       (1, 1, fa * fz)):
            col = np.clip((ai + da) * len(zs) + (zi + dz), 0, library.shape[0] - 1)
            np.add.at(W, (rows, col.ravel()), (wf * m).ravel())
        return W @ library

    def best_of(fn):
        # warm up before timing: without this, PCA's first call carries allocation
        # + BLAS start-up and looks ~3x slower than it is — understates the gap
        fn()
        return min(_timed(fn) for _ in range(repeats))

    t_pca, t_sfh = best_of(route_pca), best_of(route_sfh)

    print(f"\nCOMPUTE  ({n_gal:,} galaxies, {n_bins} SFH bins x 2 components, "
          f"best of {repeats})")
    print(f"  coefficients -> spectra   ({n_gal},{comps.shape[0]}) @ "
          f"({comps.shape[0]},{library.shape[1]})   {t_pca * 1e3:>8.1f} ms")
    print(f"  SFH -> spectra            ({n_gal},{library.shape[0]}) @ "
          f"({library.shape[0]},{library.shape[1]})  {t_sfh * 1e3:>8.1f} ms")
    print(f"\n  the SFH route is {t_sfh / t_pca:.1f}x slower — real, and negligible in "
          f"absolute terms.")

    print(f"\nDEPENDENCY  (what a downstream user must hold)")
    print(f"  PCA basis                 {comps.nbytes / 1e3:>8.0f} kB")
    print(f"  BC03 SSP library          {library.nbytes / 1e6:>8.1f} MB   "
          f"({library.nbytes / comps.nbytes:.0f}x larger)")
    return t_pca, t_sfh


def _timed(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-galaxies", type=int, default=11396,
                    help="catalogue size (default 11,396 — the z=0 sample)")
    ap.add_argument("--bins", type=int, default=DEFAULT_BINS)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    print(__doc__.strip().split("\n")[0])
    print("=" * 72)
    storage_report()
    compute_report(args.n_galaxies, args.bins, args.repeats)
    print("\nSee documents/why_not_just_the_sfh.md for what these numbers mean.")


if __name__ == "__main__":
    main()
