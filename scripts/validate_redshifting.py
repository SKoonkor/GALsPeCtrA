"""
validate_redshifting.py

Checks GALsPeCtrA's observer-frame photometry against L-GALAXIES' own `ObsMag`, at every
output redshift.

Why this is the right reference
-------------------------------
`ObsMag` is an **observer-frame absolute** magnitude (`code/h_galaxy_output.h:234`): the
spectrum is redshifted, but it is still referred to 10 pc. The distance modulus only
enters under L-GALAXIES' `APP` flag, which is off. So comparing against it isolates the
part of the calculation that can plausibly be wrong — the spectral stretch and the
resulting band shift — from the distance term, which is arithmetic.

It is not a perfect reference. L-GALAXIES' magnitudes carry the ~7 mmag quadrature error
characterised in `documents/lgalaxies_quadrature.md`, so agreement should be read against
that floor, not against zero. That is still a sharp instrument for this question: a sign
error or a missing (1+z) shows up at tenths of a magnitude, twenty times larger.

What it reports
---------------
1. Per band and per redshift, the median offset and scatter between the reconstructed
   observer-frame magnitude and `ObsMag`.
2. How often the PCA reconstruction goes **negative** inside each band at each redshift.
   Harmless at z = 0; the far-UV region where it happens enters the observed optical by
   z ~ 1.5, and this is the measurement that says whether it matters yet.

Usage
  python scripts/validate_redshifting.py
  python scripts/validate_redshifting.py --n-gals 2000     # faster
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from galspectra.csp.builder import build_csp_coefficients                # noqa: E402
from galspectra.csp.interpolator import PCACoefficientInterpolator       # noqa: E402
from galspectra.csp.reconstruct import reconstruction_sed_from_pca       # noqa: E402
from galspectra.lgalaxies import load_sample                             # noqa: E402
from galspectra.lgalaxies.sfh import extract_sfh                         # noqa: E402
from galspectra.photometry.filters import load_sdss_filters              # noqa: E402
from galspectra.photometry.redshift import observed_frame                # noqa: E402
from galspectra.pca.metrics import BandOperator                          # noqa: E402
from galspectra.utils.grid import build_param_grid                       # noqa: E402

LGAL_ROOT = PROJECT_ROOT.parent / "L-GALAXIES" / "LGalaxies2020_PublicRepository-master"
SFH_FITS = LGAL_ROOT / "AuxCode/Python/Database_SFH_table.fits"
FILTER_DIR = LGAL_ROOT / "SpecPhotTables/Filters"
SAMPLE_DIR = LGAL_ROOT / "output/samples"
PCA_FILE = PROJECT_ROOT / "data" / "pca_results_bc03.npz"

REDSHIFTS = (0.00, 0.51, 1.04, 2.07, 3.11, 5.03)
BANDS = ("u", "g", "r", "i", "z")
MAG_SENTINEL = 90.0
_D_10PC_CM = 3.0857e19
_FOUR_PI_10PC_SQ = 4.0 * np.pi * _D_10PC_CM ** 2


def sample_path(z):
    return SAMPLE_DIR / (f"Planck_Mil-I_snapshots_default_test3_"
                         f"z{z:0.2f}-{z:0.2f}_All.npy")


def reconstruct(sample, sfh_table, interp, pca, n_gals):
    """Rest-frame SEDs at 10 pc, plus the total formed mass, for the first n galaxies."""
    components = pca["components"]
    mean = pca["mean"]
    norm = pca["norm"].item() if pca["norm"].ndim == 0 else dict(pca["norm"])
    wave = pca["wave"]

    seds, keep, failed = [], [], 0
    for i in range(min(n_gals, len(sample))):
        try:
            sfh = extract_sfh(sample[i], sfh_table)
            ages = sfh["age_Gyr"]
            # Same two-component sum as scripts/process_lgalaxies.py:333-337.
            coeffs = (build_csp_coefficients(interp, ages, sfh["logzsol_disk"],
                                             sfh["disk_mass"])
                      + build_csp_coefficients(interp, ages, sfh["logzsol_bulge"],
                                               sfh["bulge_mass"]))
            mass = float(sfh["disk_mass"].sum() + sfh["bulge_mass"].sum())
        except Exception:
            failed += 1
            continue
        if mass <= 0:
            continue
        sed = reconstruction_sed_from_pca(coeffs / mass, components, mean, norm)
        seds.append(sed * mass / _FOUR_PI_10PC_SQ)
        keep.append(i)
    if failed:
        print(f"    ({failed} galaxies failed SFH/CSP extraction)")
    return wave, np.asarray(seds), np.asarray(keep)


def _fit_redshift(wave, seds, native, filters, z_guess, keep, half_width=0.5, step=0.005):
    """The redshift that best reproduces `ObsMag`, scanned around the requested one.

    Returns the minimiser of the mean |median residual| over the bands that survive the
    coverage test. This is a *measurement of the snapshot's redshift*, not a fitted
    nuisance parameter: the answer lands on tabulated simulation redshifts, which is
    what confirms the interpretation.
    """
    best_z, best_score = z_guess, np.inf
    for zt in np.arange(max(0.0, z_guess - half_width), z_guess + half_width, step):
        wo, so = observed_frame(wave, seds, zt)
        op = BandOperator(wo, filters, min_coverage=0.99)
        mags = op.magnitudes(so)
        res = []
        for b in op.names:
            j = BANDS.index(b)
            k = op.index(b)
            ok = np.isfinite(mags[:, k]) & (np.abs(native[:, j]) < MAG_SENTINEL)
            if ok.sum() > 20:
                res.append(abs(float(np.median(mags[ok, k] - native[ok, j]))))
        if res:
            score = float(np.mean(res))
            if score < best_score:
                best_z, best_score = float(zt), score
    return best_z


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-gals", type=int, default=1500,
                    help="galaxies per redshift (default 1500; the full sample is slow)")
    ap.add_argument("--json", default=str(PROJECT_ROOT / "data" / "redshift_validation.json"))
    args = ap.parse_args()

    from galspectra.lgalaxies import load_sfh_table
    sfh_table = load_sfh_table(SFH_FITS)
    pca = np.load(PCA_FILE, allow_pickle=True)
    # Same construction as scripts/process_lgalaxies.py:253, so the reconstruction here
    # is the production one and any difference is the redshifting.
    ages_grid, z_grid, coeff_grid = build_param_grid(
        pca["params"], pca["coeffs"], list(pca["param_names"]))
    interp = PCACoefficientInterpolator(ages_grid, z_grid, coeff_grid)
    filters = load_sdss_filters(FILTER_DIR)

    print("=" * 78)
    print("OBSERVER-FRAME VALIDATION — reconstruction vs L-GALAXIES ObsMag")
    print("=" * 78)
    print("  reference carries the ~7 mmag quadrature error of "
          "documents/lgalaxies_quadrature.md;\n  read the residuals against that floor.\n")

    results = {}
    for z in REDSHIFTS:
        path = sample_path(z)
        if not path.exists():
            print(f"z = {z:.2f}: sample missing ({path.name}) — skipped")
            continue
        sample = load_sample(path)
        wave, seds, keep = reconstruct(sample, sfh_table, interp, pca, args.n_gals)
        if not len(seds):
            print(f"z = {z:.2f}: no galaxies reconstructed — skipped")
            continue

        native = np.asarray(sample["ObsMag"], dtype=float)[keep]
        snap = int(sample["SnapNum"][0])

        # The output filename carries the requested redshift, not the snapshot's
        # actual one (differs by up to 0.27 here) — using the filename value gives
        # tenths-of-a-mag residuals that look like a broken k-correction. Recover
        # the actual redshift by asking which one reproduces ObsMag.
        z_eff = _fit_redshift(wave, seds, native, filters, z, keep)

        wave_obs, seds_obs = observed_frame(wave, seds, z_eff)
        op = BandOperator(wave_obs, filters, min_coverage=0.99)
        mine = op.magnitudes(seds_obs)

        row = {"z_requested": z, "z_effective": z_eff, "snapnum": snap,
               "n_galaxies": int(len(keep)), "bands": {}}
        print(f"z requested {z:.2f}  ->  snapshot {snap}, effective z = {z_eff:.3f}   "
              f"{len(keep):,} galaxies")
        print(f"    observed grid {wave_obs.min():.0f}-{wave_obs.max():.0f} AA")
        print(f"    {'band':6s}{'median':>10}{'MAD':>10}{'p99|d|':>10}{'N':>8}")
        for b in op.names:
            j = BANDS.index(b)
            k = op.index(b)
            nat = native[:, j]
            ok = np.isfinite(mine[:, k]) & np.isfinite(nat) & (np.abs(nat) < MAG_SENTINEL)
            if ok.sum() < 20:
                continue
            d = mine[ok, k] - nat[ok]
            med = float(np.median(d))
            mad = float(np.median(np.abs(d - med)))
            row["bands"][b] = {"median": med, "mad": mad,
                               "p99_abs": float(np.percentile(np.abs(d), 99)),
                               "n": int(ok.sum())}
            print(f"    {b:6s}{med:+10.4f}{mad:10.4f}"
                  f"{np.percentile(np.abs(d), 99):10.4f}{ok.sum():8d}")

        # Negative reconstructed flux, inside each band's own wavelength span.
        neg = {}
        for b, (fw, ft) in filters.items():
            inband = (wave_obs >= fw[0]) & (wave_obs <= fw[-1])
            if inband.sum() == 0:
                continue
            frac_gal = float(np.mean(np.any(seds_obs[:, inband] < 0, axis=1)))
            frac_bin = float(np.mean(seds_obs[:, inband] < 0))
            neg[b] = {"galaxies_with_any_negative": frac_gal, "negative_bins": frac_bin}
        row["negative_flux"] = neg
        worst = max(neg.items(), key=lambda kv: kv[1]["galaxies_with_any_negative"],
                    default=(None, None))
        if worst[0] is not None:
            print(f"    negative flux in band: worst {worst[0]} — "
                  f"{100*worst[1]['galaxies_with_any_negative']:.2f} % of galaxies, "
                  f"{100*worst[1]['negative_bins']:.3f} % of bins")
        print()
        results[f"{z:.2f}"] = row

    Path(args.json).write_text(json.dumps(results, indent=2) + "\n")
    print(f"  results -> {args.json}")


if __name__ == "__main__":
    main()
