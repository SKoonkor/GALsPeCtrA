"""
wavelength_requirements.py

Compute the rest-frame wavelength coverage a PCA SSP basis must span in order to
synthesise photometry for a given set of filters over a given redshift range.

For a filter with observed-frame response R(lambda) and a source at redshift z, the
rest-frame wavelength sampled by observed lambda_obs is lambda_obs / (1 + z). So:

    rest-frame requirement = [ lambda_blue_edge / (1 + z_max),
                               lambda_red_edge  / (1 + z_min) ]

The blue limit is therefore set by the bluest filter at the HIGHEST redshift, and the
red limit by the reddest filter at the LOWEST redshift.

Filter edges are taken from the real transmission curves at a threshold relative to
each curve's peak, not from nominal specification edges. Two thresholds are reported
(1% and 0.1% by default) so the sensitivity to that choice is visible.

Usage:
  cd /path/to/GALsPeCtrA
  python scripts/wavelength_requirements.py
  python scripts/wavelength_requirements.py --thresholds 0.01 0.001 --csv out.csv

Inputs:
  data/filters/paus/OUT_FILTERS/      40 narrow bands + 6 broad bands (nm, total throughput)
  data/filters/svo/                   Euclid VIS/NISP, CFHT MegaCam ugriz, 2MASS Ks (Angstrom)

Outputs:
  stdout table, and optionally a CSV of per-filter edges.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from galspectra.photometry.filter_sets import load_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILTER_ROOT = PROJECT_ROOT / "data" / "filters"
PAUS_DIR = FILTER_ROOT / "paus" / "OUT_FILTERS"
SVO_DIR = FILTER_ROOT / "svo"

# The committed reference basis, for comparison.
REFERENCE_BASIS = (805.0, 29950.0)
REFERENCE_BASIS_NAME = "committed BC03 basis (data/pca_results_bc03.npz)"


# ─────────────────────────────────────────────────────────────────────────────
# Filter set definitions
#
# Each entry: (label, z_min, z_max, list of (band_name, path, wavelength_unit))
# `role` distinguishes the science drivers from validation-only bands.
# ─────────────────────────────────────────────────────────────────────────────

def _paus_narrow_bands():
    """The 40 PAUCam narrow bands, AOD_D_<centre_nm>.dat, centres 455-845 nm."""
    bands = []
    for centre_nm in range(455, 855, 10):
        path = PAUS_DIR / f"AOD_D_{centre_nm}.dat"
        bands.append((f"NB{centre_nm}", path, "nm"))
    return bands


def _paus_broad_bands():
    return [
        (f"PAU_{b}", PAUS_DIR / f"AOD_BBFL_{b}.txt", "nm")
        for b in ("u", "g", "r", "i", "z", "Y")
    ]


FILTER_SETS = [
    {
        "label": "PAUS 40 narrow bands",
        "z_min": 0.0,
        "z_max": 2.0,
        "role": "science",
        "bands": _paus_narrow_bands(),
    },
    {
        "label": "CFHTLS ugriz (MegaCam)",
        "z_min": 0.0,
        "z_max": 2.0,
        "role": "science",
        "bands": [
            (f"MegaCam_{b}", SVO_DIR / f"CFHT_MegaCam.{b}.dat", "AA")
            for b in ("u", "g", "r", "i", "z")
        ],
    },
    {
        "label": "Euclid VIS",
        "z_min": 0.0,
        "z_max": 2.0,
        "role": "science",
        "bands": [("VIS", SVO_DIR / "Euclid_VIS.vis.dat", "AA")],
    },
    {
        "label": "Euclid NISP Y/J/H",
        "z_min": 0.0,
        "z_max": 2.0,
        "role": "science",
        "bands": [
            (f"NISP_{b}", SVO_DIR / f"Euclid_NISP.{b}.dat", "AA")
            for b in ("Y", "J", "H")
        ],
    },
    {
        "label": "2MASS Ks (validation only)",
        "z_min": 0.0,
        "z_max": 0.3,
        "role": "validation",
        "bands": [("2MASS_Ks", SVO_DIR / "2MASS_2MASS.Ks.dat", "AA")],
    },
    {
        "label": "PAUS 6 broad bands (reference)",
        "z_min": 0.0,
        "z_max": 2.0,
        "role": "reference",
        "bands": _paus_broad_bands(),
    },
]

# `load_curve` now lives in galspectra.photometry.filter_sets — the canonical home,
# shared with the PCA experiment harness so a parser fix reaches both.


# ─────────────────────────────────────────────────────────────────────────────

def curve_edges(wave, trans, threshold):
    """Blue and red edges where transmission first/last exceeds `threshold` x peak.

    Edges are linearly interpolated between the bracketing samples rather than snapped
    to a grid point, so the answer does not depend on the curve's sampling density.
    """
    peak = float(np.max(trans))
    if peak <= 0:
        raise ValueError("curve has non-positive peak transmission")
    level = threshold * peak

    above = np.flatnonzero(trans >= level)
    if above.size == 0:
        raise ValueError(f"no samples above {threshold:g} x peak")

    i_blue, i_red = int(above[0]), int(above[-1])

    def _interp(i_in, i_out):
        """Interpolate the crossing between an inside and an outside sample."""
        if i_out < 0 or i_out >= len(wave):
            return float(wave[i_in])
        t_in, t_out = trans[i_in], trans[i_out]
        if t_in == t_out:
            return float(wave[i_in])
        f = (level - t_out) / (t_in - t_out)
        return float(wave[i_out] + f * (wave[i_in] - wave[i_out]))

    return _interp(i_blue, i_blue - 1), _interp(i_red, i_red + 1)


def pivot_wavelength(wave, trans):
    """Pivot wavelength: sqrt( int(R lam dlam) / int(R dlam / lam) )."""
    num = np.trapezoid(trans * wave, wave)
    den = np.trapezoid(trans / wave, wave)
    return float(np.sqrt(num / den))


# ─────────────────────────────────────────────────────────────────────────────

def analyse(thresholds):
    """Return (per_band_records, per_set_records, union_by_threshold)."""
    per_band, per_set = [], []

    for fs in FILTER_SETS:
        z_min, z_max = fs["z_min"], fs["z_max"]
        set_edges = {t: {"blue": [], "red": []} for t in thresholds}

        for name, path, unit in fs["bands"]:
            wave, trans = load_curve(path, unit)
            rec = {
                "set": fs["label"],
                "role": fs["role"],
                "band": name,
                "file": str(path.relative_to(PROJECT_ROOT)),
                "n_samples": len(wave),
                "peak_throughput": float(np.max(trans)),
                "pivot_AA": pivot_wavelength(wave, trans),
                "sampled_min_AA": float(wave[0]),
                "sampled_max_AA": float(wave[-1]),
            }
            for t in thresholds:
                blue, red = curve_edges(wave, trans, t)
                rec[f"blue_AA@{t:g}"] = blue
                rec[f"red_AA@{t:g}"] = red
                set_edges[t]["blue"].append(blue)
                set_edges[t]["red"].append(red)
            per_band.append(rec)

        for t in thresholds:
            blue_obs = min(set_edges[t]["blue"])
            red_obs = max(set_edges[t]["red"])
            # bluest filter at the highest z sets the blue limit;
            # reddest filter at the lowest z sets the red limit
            per_set.append({
                "set": fs["label"],
                "role": fs["role"],
                "threshold": t,
                "z_min": z_min,
                "z_max": z_max,
                "obs_blue_AA": blue_obs,
                "obs_red_AA": red_obs,
                "rest_blue_AA": blue_obs / (1.0 + z_max),
                "rest_red_AA": red_obs / (1.0 + z_min),
                "blue_driver": set_edges[t]["blue"].index(blue_obs),
                "red_driver": set_edges[t]["red"].index(red_obs),
            })

    union = {}
    for t in thresholds:
        rows = [r for r in per_set if r["threshold"] == t and r["role"] != "reference"]
        blue_row = min(rows, key=lambda r: r["rest_blue_AA"])
        red_row = max(rows, key=lambda r: r["rest_red_AA"])
        union[t] = {
            "rest_blue_AA": blue_row["rest_blue_AA"],
            "rest_red_AA": red_row["rest_red_AA"],
            "blue_set": blue_row["set"],
            "blue_z": blue_row["z_max"],
            "red_set": red_row["set"],
            "red_z": red_row["z_min"],
        }
    return per_band, per_set, union


def _band_name_at(fs_label, idx):
    for fs in FILTER_SETS:
        if fs["label"] == fs_label:
            return fs["bands"][idx][0]
    return "?"


def report(per_band, per_set, union, thresholds):
    print("=" * 78)
    print("REST-FRAME WAVELENGTH REQUIREMENTS")
    print("=" * 78)

    for t in thresholds:
        print(f"\n--- edges at {t:g} x peak throughput " + "-" * 34)
        print(f"{'filter set':<32} {'z range':<11} {'observed (AA)':<19} {'rest-frame (AA)':<19}")
        for r in [x for x in per_set if x["threshold"] == t]:
            zr = f"{r['z_min']:.1f}-{r['z_max']:.1f}"
            obs = f"{r['obs_blue_AA']:.0f}-{r['obs_red_AA']:.0f}"
            rest = f"{r['rest_blue_AA']:.0f}-{r['rest_red_AA']:.0f}"
            tag = "" if r["role"] == "science" else f"  [{r['role']}]"
            print(f"{r['set']:<32} {zr:<11} {obs:<19} {rest:<19}{tag}")

        u = union[t]
        print(f"\n  UNION (science + validation): {u['rest_blue_AA']:.0f} - {u['rest_red_AA']:.0f} AA")
        print(f"    blue limit set by : {u['blue_set']} at z = {u['blue_z']:.1f}")
        print(f"    red  limit set by : {u['red_set']} at z = {u['red_z']:.1f}")

        lo, hi = REFERENCE_BASIS
        covers = lo <= u["rest_blue_AA"] and hi >= u["rest_red_AA"]
        print(f"\n  {REFERENCE_BASIS_NAME}: {lo:.0f} - {hi:.0f} AA")
        print(f"    covers the union? {'YES' if covers else 'NO'}"
              f"   (blue margin {u['rest_blue_AA'] - lo:+.0f} AA,"
              f" red margin {hi - u['rest_red_AA']:+.0f} AA)")

    print("\n" + "=" * 78)
    print("PER-BAND EXTREMES (widest and narrowest, at the tightest threshold)")
    print("=" * 78)
    t = min(thresholds)
    bluest = min(per_band, key=lambda r: r[f"blue_AA@{t:g}"])
    reddest = max(per_band, key=lambda r: r[f"red_AA@{t:g}"])
    print(f"  bluest edge : {bluest['band']:<12} {bluest[f'blue_AA@{t:g}']:.1f} AA   ({bluest['set']})")
    print(f"  reddest edge: {reddest['band']:<12} {reddest[f'red_AA@{t:g}']:.1f} AA   ({reddest['set']})")


def write_csv(per_band, path, thresholds):
    cols = ["set", "role", "band", "file", "n_samples", "peak_throughput",
            "pivot_AA", "sampled_min_AA", "sampled_max_AA"]
    for t in thresholds:
        cols += [f"blue_AA@{t:g}", f"red_AA@{t:g}"]
    lines = [",".join(cols)]
    for r in per_band:
        lines.append(",".join(
            f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols
        ))
    Path(path).write_text("\n".join(lines) + "\n")
    print(f"\nPer-band edges written to {path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.01, 0.001],
                   help="Transmission thresholds as a fraction of peak (default 0.01 0.001)")
    p.add_argument("--csv", default=str(PROJECT_ROOT / "data" / "wavelength_requirements.csv"),
                   help="Where to write the per-band edge table")
    p.add_argument("--json", default=None, help="Optional JSON dump of the union result")
    return p.parse_args()


def main():
    args = parse_args()
    thresholds = sorted(args.thresholds, reverse=True)
    per_band, per_set, union = analyse(thresholds)
    report(per_band, per_set, union, thresholds)
    if args.csv:
        write_csv(per_band, args.csv, thresholds)
    if args.json:
        Path(args.json).write_text(json.dumps(
            {str(k): v for k, v in union.items()}, indent=2) + "\n")
        print(f"Union summary written to {args.json}")


if __name__ == "__main__":
    main()
