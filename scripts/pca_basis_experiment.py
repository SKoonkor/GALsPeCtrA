"""
pca_basis_experiment.py

Weighted-PCA experiment harness and error-vs-bytes budget for the GALsPeCtrA
spectral basis.

The question this answers
------------------------
The committed basis stores 50 PCA coefficients per galaxy and reproduces L-GALAXIES
broadband magnitudes to 0.01-0.03 mag. Two things about that number are unexamined:

1. **Is 50 the right count, and is the current weighting the right weighting?**
   The incumbent basis was fitted on per-wavelength-standardised flux, and the
   component count was never selected against an error budget. Colour-split
   validation subsequently found a systematic, one-directional bias — red galaxies
   are reconstructed ~2x worse than blue, +0.026 mag differential in *g* — which is
   a statement about *where in wavelength* the basis is spending its components.
   Weighting is the lever that moves that.

2. **What is the price of a byte?** 50 floats is a choice, not a constraint. This
   script produces the curve of reconstruction error against storage cost, so the
   choice can be made against a stated requirement instead of a variance threshold.

**Component count is never selected from a cumulative-variance fraction.** Explained
variance measures agreement with the library's own dominant modes; it says nothing
about the magnitude error in any band a telescope actually has. The two disagree
badly here: 50 components explain >99.99 % of the variance while still leaving a
measurable colour bias.

What is measured
----------------
* per-band magnitude error in **mmag**, for PAUS's 40 narrow bands, CFHT MegaCam
  ugriz, Euclid VIS + NISP YJH, 2MASS Ks and the SDSS ugriz L-GALAXIES itself uses
* fractional flux error by rest-frame wavelength region
* error as a function of SSP age and metallicity — which populations the basis fails
* the same, per galaxy, propagated through real L-GALAXIES star-formation histories
* D4000, both directly from the spectrum and via Renard et al. (2022) Eqs. 1-5 from
  the PAUS narrow bands, at the redshifts where the break falls inside the filter set
* bytes per galaxy, counting the total formed stellar mass as the extra number it is
* the red-fraction shift at the calibrated colour cut — the acceptance criterion
  inherited from the colour-split validation

Baselines it is measured against
--------------------------------
  (a) the full spectrum at the evaluation resolution — lossless, expensive
  (b) a Shamshiri-style binned star-formation history at several bin counts
  (c) broadband photometry alone, with a single-SSP fit standing in for the spectrum
  (d) the committed 50-component basis, evaluated through the identical pipeline

Usage
-----
  cd /path/to/GALsPeCtrA
  python scripts/pca_basis_experiment.py --config configs/pca_experiments/default.yaml
  python scripts/pca_basis_experiment.py --config ... --quick        # small scan, no galaxies
  python scripts/pca_basis_experiment.py --config ... --no-figures

Outputs (under the configured output directory)
  <name>_results.json        every number, machine-readable
  <name>_error_vs_bytes.csv  the curve itself
  figures/pca_basis/*.png    the figures

This script is offline analysis. It does not modify the frozen basis, any existing
`data/` product, or the L-GALAXIES C code.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from galspectra.config import load_config                                   # noqa: E402
from galspectra.csp.ssp_weights import SSPGridIndex, build_weight_matrix    # noqa: E402
from galspectra.pca.basis import SpectralBasis                              # noqa: E402
from galspectra.pca.metrics import (                                        # noqa: E402
    BandOperator,
    d4000_from_operator,
    d4000_narrowband_operator,
    d4000_operator,
    mag_difference,
    region_flux_errors,
    summarise_mmag,
)
from galspectra.pca.resample import (                                       # noqa: E402
    log_wavelength_grid,
    native_resolution,
    resampling_matrix,
    resolution_report,
)
from galspectra.photometry.filter_sets import load_filter_set               # noqa: E402

MAG_SENTINEL = 90.0   # L-GALAXIES writes 99.0, not NaN, for "no magnitude"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve(path):
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _jsonable(obj):
    """Recursively convert numpy types so json.dump does not choke."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def _fmt(x, nd=2):
    if x is None:
        return "n/a"
    x = float(x)
    return "n/a" if not np.isfinite(x) else f"{x:.{nd}f}"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — library and evaluation grid
# ─────────────────────────────────────────────────────────────────────────────

def load_library(cfg):
    path = _resolve(cfg["library"]["sed_grid"])
    d = np.load(path, allow_pickle=True)
    wave = np.asarray(d["wave"], dtype=float)
    X = np.asarray(d["seds"], dtype=float)
    print(f"  library      {path.name}: {X.shape[0]} SSPs x {X.shape[1]} bins, "
          f"{wave.min():.0f}-{wave.max():.0f} Å")
    return wave, X, np.asarray(d["params"], dtype=float), list(d["param_names"])


def build_evaluation_grid(cfg, wave_native):
    g = cfg["grid"]
    lo, hi, R = float(g["wave_min"]), float(g["wave_max"]), float(g["resolution"])
    centres, edges = log_wavelength_grid(lo, hi, R)
    M = resampling_matrix(wave_native, edges)
    rep = resolution_report(wave_native, R, lo, hi)

    print(f"  eval grid    R={R:g}, {centres.size} bins over {lo:.0f}-{hi:.0f} Å")
    print(f"               native R here is {rep['native_R_min']:.0f}-{rep['native_R_max']:.0f} "
          f"(median {rep['native_R_median']:.0f})")
    if rep["interpolating"]:
        print(f"               NOTE: {100 * rep['fraction_oversampled']:.0f}% of the range is "
              f"oversampled (up to {rep['max_oversampling_factor']:.1f}x). Resampling there "
              f"interpolates the library; it does not add spectral information.")
    return centres, edges, M, rep


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — operators
# ─────────────────────────────────────────────────────────────────────────────

def build_operators(cfg, wave_eval):
    sets = list(cfg["photometry"]["sets"])
    filters = load_filter_set(*sets)
    band_op = BandOperator(wave_eval, filters,
                           min_coverage=float(cfg["photometry"].get("min_coverage", 0.99)))
    print(f"  photometry   {len(band_op)} of {len(filters)} bands usable"
          + (f"; rejected {band_op.rejected}" if band_op.rejected else ""))

    membership = {}
    for s in sets:
        names = list(load_filter_set(s).keys())
        membership[s] = [n for n in names if n in band_op.names]

    d4000_ops = {defn: d4000_operator(wave_eval, defn)
                 for defn in cfg["d4000"]["definitions"]}

    nb_ops = {}
    if "paus_nb" in sets:
        paus = load_filter_set("paus_nb")
        for z in cfg["d4000"].get("narrowband_redshifts", []):
            for defn in cfg["d4000"]["definitions"]:
                try:
                    op, detail = d4000_narrowband_operator(wave_eval, paus, float(z), defn)
                    nb_ops[f"{defn}/narrowband_z{float(z):g}"] = (op, detail)
                except ValueError as exc:
                    print(f"               D4000 NB {defn} z={z}: skipped — {exc}")
    print(f"  D4000        {len(d4000_ops)} direct, {len(nb_ops)} narrow-band reconstructions")
    return band_op, membership, d4000_ops, nb_ops


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — galaxies
# ─────────────────────────────────────────────────────────────────────────────

def build_galaxy_weights(cfg, params, param_names):
    gcfg = cfg.get("galaxies") or {}
    sample = gcfg.get("sample")
    if not sample:
        return None

    sample = _resolve(sample)
    sfh_table_path = _resolve(gcfg["sfh_table"])
    if not sample.exists() or not sfh_table_path.exists():
        print(f"  galaxies     SKIPPED — {sample} or {sfh_table_path} not found")
        return None

    from galspectra.lgalaxies import extract_sfh, load_sample, load_sfh_table

    G = load_sample(sample)
    table = load_sfh_table(sfh_table_path)
    n_req = gcfg.get("n_galaxies")
    n = len(G) if n_req in (None, "all") else min(int(n_req), len(G))

    records = [extract_sfh(G[i], table) for i in range(n)]
    index = SSPGridIndex(params, param_names)
    W, info = build_weight_matrix(index, records)

    print(f"  galaxies     {n:,} from {sample.name}; "
          f"{info['n_bins_used']:,} SFH bins used, "
          f"{info['n_clamped']['age']} age / {info['n_clamped']['logzsol']} metallicity clamps")

    out = {"W": W, "info": info, "n": n, "sample": str(sample), "records": records}

    bundle_path = _resolve(gcfg.get("bundle", "data/webapp_bundle_MilI.npz"))
    if bundle_path.exists():
        b = np.load(bundle_path, allow_pickle=True)
        if int(b["n_galaxies"]) == len(G):
            out["native_mag"] = {k: np.asarray(b[f"Mag_{k}"], dtype=float)[:n]
                                 for k in ("g", "r")}
            print(f"               native L-GALAXIES Mag_g/Mag_r loaded from {bundle_path.name}")
        else:
            print(f"               {bundle_path.name} has {int(b['n_galaxies'])} galaxies, "
                  f"sample has {len(G)} — native magnitudes not used")
    return out


def load_colour_cut(cfg):
    gcfg = cfg.get("galaxies") or {}
    path = _resolve(gcfg.get("colour_cut", "data/colour_cut_calibration.json"))
    if not path.exists():
        print(f"  colour cut   {path} not found; falling back to 0.4763")
        return 0.4763, "fallback"
    cal = json.loads(path.read_text())["calibrated"]
    # The harness works on dust-free spectra, so the intrinsic calibration is the
    # matching one. It differs from the dust cut by 0.0007 mag.
    cut = float(cal["lgal_gmm_cut_intrinsic"])
    print(f"  colour cut   g-r >= {cut:.4f} (lgal_gmm, intrinsic) from {path.name}")
    return cut, "lgal_gmm_intrinsic"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — evaluation of one reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def _d4000_stats(op, X_true, X_hat):
    t = d4000_from_operator(op, X_true)
    r = d4000_from_operator(op, X_hat)
    good = np.isfinite(t) & np.isfinite(r) & (t > 0)
    if not good.any():
        return None
    return {
        "bias": float(np.median((r - t)[good])),
        "mad": float(np.median(np.abs(r - t)[good])),
        "p99": float(np.percentile(np.abs(r - t)[good], 99.0)),
        "mad_frac": float(np.median(np.abs(r - t)[good] / t[good])),
    }


def evaluate_reconstruction(label, bytes_per_gal, X_true, X_hat, wave_eval, band_op,
                            membership, d4000_ops, nb_ops, params, gal, cut,
                            n_components=None):
    """All metrics for one candidate representation of the SSP library.

    `X_hat` is the library as this representation reproduces it, on the evaluation
    grid. Everything downstream — galaxy spectra, colours, D4000 — follows from it by
    linear operations, so a single (N_ssp, N_wave) array characterises the
    representation completely.
    """
    res = {"label": label, "bytes_per_galaxy": int(bytes_per_gal)}
    if n_components is not None:
        res["n_components"] = int(n_components)

    # ── physicality: a truncated basis can reconstruct negative flux ──────
    # This is not a rounding artefact. A low-N reconstruction genuinely goes
    # negative in the faint UV, which makes the magnitude undefined rather than
    # merely inaccurate, so it is counted separately from the error statistics.
    neg = X_hat < 0
    res["physicality"] = {
        "frac_bins_negative": float(neg.mean()),
        "frac_ssp_with_any_negative": float(neg.any(axis=1).mean()),
        "worst_negative_fraction_of_peak": float(
            (X_hat.min(axis=1) / X_true.max(axis=1)).min()),
    }

    # ── per-SSP photometry ────────────────────────────────────────────────
    d_ssp = mag_difference(X_true, X_hat, band_op)
    s = summarise_mmag(d_ssp, axis=0)
    res["ssp_bands"] = {name: {k: float(v[i]) for k, v in s.items()}
                        for i, name in enumerate(band_op.names)}
    res["ssp_overall"] = {k: (float(np.nanmax(v)) if k == "max" else float(np.nanmedian(v)))
                          for k, v in s.items()}
    res["ssp_by_set"] = {}
    for set_name, bands in membership.items():
        if not bands:
            continue
        idx = [band_op.index(b) for b in bands]
        sub = np.abs(d_ssp[:, idx])
        per_band = np.nanmedian(sub, axis=0)
        res["ssp_by_set"][set_name] = {
            "n_bands": len(bands),
            "mad_mmag": float(np.nanmedian(sub)),
            "p99_mmag": float(np.nanpercentile(sub, 99.0)),
            "max_mmag": float(np.nanmax(sub)),
            "worst_band": bands[int(np.nanargmax(per_band))],
        }

    # ── flux error by wavelength region ───────────────────────────────────
    res["regions"] = region_flux_errors(wave_eval, X_true, X_hat)

    # ── error as a function of SSP age and metallicity ────────────────────
    ages, zsol = params[:, 0], params[:, 1]
    ref_candidates = [b for b in ("g", "MegaCam_g", "NB575") if b in band_op.names]
    ref = band_op.index(ref_candidates[0]) if ref_candidates else 0
    res["reference_band_for_age_Z"] = band_op.names[ref]
    age_edges = np.array([0.0, 0.01, 0.1, 1.0, 5.0, 20.0])
    res["by_age"] = []
    for lo, hi in zip(age_edges[:-1], age_edges[1:]):
        sel = (ages >= lo) & (ages < hi)
        if sel.sum() == 0:
            continue
        a = np.abs(d_ssp[sel, ref])
        res["by_age"].append({
            "age_lo_Gyr": float(lo), "age_hi_Gyr": float(hi), "n_ssp": int(sel.sum()),
            "mad_mmag": float(np.nanmedian(a)), "p99_mmag": float(np.nanpercentile(a, 99.0))})
    res["by_metallicity"] = []
    for z in np.unique(zsol):
        a = np.abs(d_ssp[zsol == z, ref])
        res["by_metallicity"].append({
            "logzsol": float(z), "n_ssp": int((zsol == z).sum()),
            "mad_mmag": float(np.nanmedian(a)), "p99_mmag": float(np.nanpercentile(a, 99.0))})

    # ── D4000 ─────────────────────────────────────────────────────────────
    d4 = {}
    for defn, op in d4000_ops.items():
        st = _d4000_stats(op, X_true, X_hat)
        if st:
            d4[f"{defn}/direct"] = st
    for key, (op, _detail) in nb_ops.items():
        st = _d4000_stats(op, X_true, X_hat)
        if st:
            d4[key] = st
    res["d4000"] = d4

    # ── galaxy level ──────────────────────────────────────────────────────
    if gal is not None:
        res["galaxies"] = galaxy_metrics(X_true, X_hat, band_op, membership, gal, cut,
                                         d4000_ops, nb_ops)
    return res


def galaxy_metrics(X_true, X_hat, band_op, membership, gal, cut, d4000_ops, nb_ops):
    """Propagate the SSP-level reconstruction through real star-formation histories.

    Both sides are `W @ (X @ P.T)`, so the only difference between truth and
    reconstruction is the library itself. Nothing here can drift.
    """
    W = gal["W"]
    F_true = W @ (X_true @ band_op.P.T)
    F_hat = W @ (X_hat @ band_op.P.T)

    with np.errstate(divide="ignore", invalid="ignore"):
        ok = (F_true > 0) & (F_hat > 0)
        dmag = np.where(ok, -2.5 * np.log10(np.where(ok, F_hat / F_true, np.nan)) * 1000.0,
                        np.nan)

    s = summarise_mmag(dmag, axis=0)
    out = {
        "n_galaxies": int(W.shape[0]),
        "bands": {name: {k: float(v[i]) for k, v in s.items()}
                  for i, name in enumerate(band_op.names)},
        "overall": {k: (float(np.nanmax(v)) if k == "max" else float(np.nanmedian(v)))
                    for k, v in s.items()},
        "by_set": {},
    }
    for set_name, bands in membership.items():
        if not bands:
            continue
        idx = [band_op.index(b) for b in bands]
        sub = np.abs(dmag[:, idx])
        out["by_set"][set_name] = {
            "mad_mmag": float(np.nanmedian(sub)),
            "p99_mmag": float(np.nanpercentile(sub, 99.0)),
            "max_mmag": float(np.nanmax(sub)),
        }

    # ── the acceptance criterion ──────────────────────────────────────────
    gr = ("g", "r") if {"g", "r"} <= set(band_op.names) else ("MegaCam_g", "MegaCam_r")
    if set(gr) <= set(band_op.names):
        ig, ir = band_op.index(gr[0]), band_op.index(gr[1])
        with np.errstate(divide="ignore", invalid="ignore"):
            gr_t = -2.5 * np.log10(F_true[:, ig] / F_true[:, ir])
            gr_r = -2.5 * np.log10(F_hat[:, ig] / F_hat[:, ir])
        good = np.isfinite(gr_t) & np.isfinite(gr_r)
        gt, grr = gr_t[good], gr_r[good]
        red_t, red_r = gt >= cut, grr >= cut
        out["colour"] = {
            "bands": list(gr),
            "cut": float(cut),
            "n_valid": int(good.sum()),
            "median_gr_truth": float(np.median(gt)),
            "median_gr_recon": float(np.median(grr)),
            "delta_median_gr": float(np.median(grr - gt)),
            "f_red_truth": float(red_t.mean()),
            "f_red_recon": float(red_r.mean()),
            "red_fraction_shift": float(red_r.mean() - red_t.mean()),
            "blue_to_red": int(np.sum(~red_t & red_r)),
            "red_to_blue": int(np.sum(red_t & ~red_r)),
            "gr_bias_red_pop_mmag": (float(np.median((grr - gt)[red_t]) * 1000.0)
                                     if red_t.any() else None),
            "gr_bias_blue_pop_mmag": (float(np.median((grr - gt)[~red_t]) * 1000.0)
                                      if (~red_t).any() else None),
        }
        rb, bb = out["colour"]["gr_bias_red_pop_mmag"], out["colour"]["gr_bias_blue_pop_mmag"]
        out["colour"]["red_minus_blue_differential_mmag"] = (
            None if rb is None or bb is None else rb - bb)

        # Continuity with the colour-split validation, which classified on
        # L-GALAXIES' own magnitudes rather than on the direct SSP sum.
        nat = gal.get("native_mag")
        if nat is not None:
            gr_n = nat["g"] - nat["r"]
            valid = ((np.abs(nat["g"]) < MAG_SENTINEL) & (np.abs(nat["r"]) < MAG_SENTINEL)
                     & good)
            if valid.any():
                out["colour"]["vs_native_lgalaxies"] = {
                    "n_valid": int(valid.sum()),
                    "f_red_native": float((gr_n[valid] >= cut).mean()),
                    "f_red_recon": float((gr_r[valid] >= cut).mean()),
                    "red_fraction_shift": float((gr_r[valid] >= cut).mean()
                                                - (gr_n[valid] >= cut).mean()),
                }

    # ── galaxy-level D4000 ────────────────────────────────────────────────
    d4 = {}
    all_ops = ([(f"{k}/direct", v) for k, v in d4000_ops.items()]
               + [(k, v[0]) for k, v in nb_ops.items()])
    for name, op in all_ops:
        t = W @ (X_true @ op.T)
        r = W @ (X_hat @ op.T)
        with np.errstate(divide="ignore", invalid="ignore"):
            gt = np.where(t[:, 0] > 0, t[:, 1] / t[:, 0], np.nan)
            gr_ = np.where(r[:, 0] > 0, r[:, 1] / r[:, 0], np.nan)
        m = np.isfinite(gt) & np.isfinite(gr_)
        if m.any():
            d4[name] = {
                "bias": float(np.median((gr_ - gt)[m])),
                "mad": float(np.median(np.abs(gr_ - gt)[m])),
                "p99": float(np.percentile(np.abs(gr_ - gt)[m], 99.0)),
                "mad_frac": float(np.median(np.abs(gr_ - gt)[m] / gt[m])),
            }
    out["d4000"] = d4
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — baselines
# ─────────────────────────────────────────────────────────────────────────────

def _edges_from_centres(centres):
    """Bin edges consistent with the geometric centres of a log-λ grid."""
    ratio = centres[1] / centres[0]
    return np.concatenate([[centres[0] / np.sqrt(ratio)], centres * np.sqrt(ratio)])


def baseline_full_spectrum(X_true, wave_eval, dtype=np.float32):
    return ("baseline: full spectrum",
            wave_eval.size * np.dtype(dtype).itemsize,
            X_true.copy())


def baseline_committed_basis(cfg, wave_eval, resample_M, wave_native):
    """The committed 50-component basis, put through the identical pipeline.

    It lives on the native 831-bin grid over 805-29950 Å, so it is reconstructed
    there and then resampled onto the evaluation grid with the same flux-conserving
    operator that produced the truth. Both sides therefore carry the identical
    resampling, and the difference between them is the basis alone.
    """
    path = _resolve(cfg["baselines"]["committed_basis"])
    if not path.exists():
        print(f"     committed basis not found at {path}; skipped")
        return None
    d = np.load(path, allow_pickle=True)
    comps = np.asarray(d["components"], dtype=float)
    mean = np.asarray(d["mean"], dtype=float)
    wave_c = np.asarray(d["wave"], dtype=float)
    coeffs = np.asarray(d["coeffs"], dtype=float)
    norm = d["norm"].item() if d["norm"].ndim == 0 else dict(d["norm"])

    # Reconstruct on its own grid, exactly as csp/reconstruct.py does.
    recon = (mean + coeffs @ comps) * norm["std"] + norm["mean"]

    if wave_c.size == wave_native.size and np.allclose(wave_c, wave_native):
        X_hat = recon @ resample_M.T
    else:
        M_c = resampling_matrix(wave_c, _edges_from_centres(wave_eval))
        X_hat = recon @ M_c.T

    n_pc = comps.shape[0]
    return (f"baseline: previous {n_pc}-PC basis ({path.stem})",
            (n_pc + 1) * 4,      # +1 for the total formed mass, which it does not store
            X_hat, n_pc)


def baseline_broadband_only(X_true, band_op, bb_bands, dtype=np.float32):
    """Broadband storage, quantified by the best single-SSP fit to those bands.

    Five broadband magnitudes cannot reconstruct a spectrum on their own — the honest
    statement is that this representation *has no spectrum*. To give it a number
    rather than a shrug, each SSP is replaced by the library member whose broadband
    colours match it best after optimal flux scaling, which is exactly the
    single-population SED fit an analyst would do. It is a weak baseline by
    construction; that is the point.
    """
    idx = [band_op.index(b) for b in bb_bands if b in band_op.names]
    if len(idx) < 3:
        return None
    F = X_true @ band_op.P[idx].T                      # (N_ssp, N_bb)
    good = np.all(F > 0, axis=1)
    logF = np.full(F.shape, np.nan)
    logF[good] = np.log10(F[good])
    # A pure flux scaling is an additive constant in log space, so removing each
    # row's mean leaves colour alone — which is all broadband storage constrains.
    colours = logF - np.nanmean(logF, axis=1, keepdims=True)
    d2 = ((colours[:, None, :] - colours[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)                       # never match a spectrum to itself
    d2 = np.where(np.isfinite(d2), d2, np.inf)
    best = np.argmin(d2, axis=1)

    scale = np.nanmean(logF - logF[best], axis=1)
    scale = np.where(np.isfinite(scale), scale, 0.0)
    X_hat = X_true[best] * (10.0 ** scale)[:, None]
    return (f"baseline: {len(idx)} broadbands (single-SSP fit)",
            len(idx) * np.dtype(dtype).itemsize,
            X_hat)


def baseline_binned_sfh(gal, index, n_bins, dtype=np.float32):
    """Shamshiri-style storage: the star-formation history, coarsened to `n_bins`.

    Returns a *galaxy-level* weight matrix rather than a library reconstruction,
    because this representation does not approximate the library at all — it
    approximates the history. Reconstruction is exact for every SSP and lossy only
    where the coarsening merges populations of different age or metallicity.

    Bytes: two numbers per bin (formed mass and mass-weighted metallicity), which is
    half what L-GALAXIES itself stores, since it tracks disk and bulge separately.
    """
    from galspectra.csp.ssp_weights import bilinear_weights

    records = gal["records"]
    W = np.zeros((len(records), index.n_ssp))

    for g, rec in enumerate(records):
        ages, zs, mass = [], [], []
        for comp in ("disk", "bulge"):
            m = np.asarray(rec[f"{comp}_mass"], dtype=float)
            keep = m > 0
            if keep.any():
                ages.append(np.asarray(rec["age_Gyr"], dtype=float)[keep])
                zs.append(np.asarray(rec[f"logzsol_{comp}"], dtype=float)[keep])
                mass.append(m[keep])
        if not mass:
            continue
        a = np.concatenate(ages)
        z = np.concatenate(zs)
        m = np.concatenate(mass)

        # Bins are logarithmic in lookback time, matching how star-formation
        # histories are actually stored: recent history finely, early history coarsely.
        la = np.log10(np.maximum(a, 1e-4))
        edges = np.linspace(la.min(), la.max() * (1 + 1e-9) + 1e-9, n_bins + 1)
        which = np.clip(np.digitize(la, edges) - 1, 0, n_bins - 1)

        ca, cz, cm = [], [], []
        for k in range(n_bins):
            sel = which == k
            if not sel.any():
                continue
            mk = m[sel].sum()
            if mk <= 0:
                continue
            ca.append(np.average(a[sel], weights=m[sel]))
            cz.append(np.average(z[sel], weights=m[sel]))
            cm.append(mk)
        if not cm:
            continue
        rows, wts, _ = bilinear_weights(index, np.asarray(ca), np.asarray(cz))
        np.add.at(W[g], rows.ravel(), (wts * np.asarray(cm)[:, None]).ravel())

    return (f"baseline: binned SFH, {n_bins} bins",
            2 * n_bins * np.dtype(dtype).itemsize,
            W)


def galaxy_metrics_crossW(X_true, band_op, membership, W_true, W_hat, cut):
    """Galaxy metrics where the *history* differs rather than the library."""
    F_true = W_true @ (X_true @ band_op.P.T)
    F_hat = W_hat @ (X_true @ band_op.P.T)
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = (F_true > 0) & (F_hat > 0)
        dmag = np.where(ok, -2.5 * np.log10(np.where(ok, F_hat / F_true, np.nan)) * 1000.0,
                        np.nan)
    s = summarise_mmag(dmag, axis=0)
    out = {
        "n_galaxies": int(W_true.shape[0]),
        "overall": {k: (float(np.nanmax(v)) if k == "max" else float(np.nanmedian(v)))
                    for k, v in s.items()},
        "by_set": {},
    }
    for set_name, bands in membership.items():
        if not bands:
            continue
        idx = [band_op.index(b) for b in bands]
        sub = np.abs(dmag[:, idx])
        out["by_set"][set_name] = {"mad_mmag": float(np.nanmedian(sub)),
                                   "p99_mmag": float(np.nanpercentile(sub, 99.0))}
    gr = ("g", "r") if {"g", "r"} <= set(band_op.names) else ("MegaCam_g", "MegaCam_r")
    if set(gr) <= set(band_op.names):
        ig, ir = band_op.index(gr[0]), band_op.index(gr[1])
        with np.errstate(divide="ignore", invalid="ignore"):
            gt = -2.5 * np.log10(F_true[:, ig] / F_true[:, ir])
            grr = -2.5 * np.log10(F_hat[:, ig] / F_hat[:, ir])
        good = np.isfinite(gt) & np.isfinite(grr)
        gt, grr = gt[good], grr[good]
        red_t, red_r = gt >= cut, grr >= cut
        out["colour"] = {
            "cut": float(cut),
            "f_red_truth": float(red_t.mean()),
            "f_red_recon": float(red_r.mean()),
            "red_fraction_shift": float(red_r.mean() - red_t.mean()),
            "delta_median_gr": float(np.median(grr - gt)),
            "blue_to_red": int(np.sum(~red_t & red_r)),
            "red_to_blue": int(np.sum(red_t & ~red_r)),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Weighted-PCA experiment harness and error-vs-bytes budget")
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs/pca_experiments/default.yaml"),
                   help="Experiment YAML")
    p.add_argument("--output-dir", default=None, help="Override the config's output directory")
    p.add_argument("--n-galaxies", default=None,
                   help="Override the galaxy count ('all', an integer, or 'none' to skip)")
    p.add_argument("--quick", action="store_true",
                   help="Short component scan, one weighting scheme, no galaxies")
    p.add_argument("--no-figures", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    t_start = time.time()
    cfg = load_config(_resolve(args.config))
    name = cfg.get("name", Path(args.config).stem)

    if args.quick:
        cfg["components"]["scan"] = [10, 50]
        cfg["components"]["max"] = 50
        cfg["weighting"] = cfg["weighting"][:1]
        cfg["galaxies"] = {}
        cfg["baselines"]["sfh_bins"] = []
    if args.n_galaxies is not None:
        cfg.setdefault("galaxies", {})
        if args.n_galaxies.lower() == "none":
            cfg["galaxies"] = {}
        else:
            cfg["galaxies"]["n_galaxies"] = (
                None if args.n_galaxies.lower() == "all" else int(args.n_galaxies))

    out_dir = _resolve(args.output_dir or cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"PCA BASIS EXPERIMENT: {name}")
    print("=" * 78)

    # ── setup ─────────────────────────────────────────────────────────────
    wave_native, X_native, params, param_names = load_library(cfg)
    wave_eval, edges, M, res_rep = build_evaluation_grid(cfg, wave_native)
    X_true = X_native @ M.T
    band_op, membership, d4000_ops, nb_ops = build_operators(cfg, wave_eval)
    cut, cut_name = load_colour_cut(cfg)
    gal = build_galaxy_weights(cfg, params, param_names)

    # ── the resampling floor ──────────────────────────────────────────────
    # Magnitudes computed on the native grid versus on the evaluation grid. No basis
    # error below this is measurable, so it is reported before anything else.
    band_op_native = BandOperator(
        wave_native, load_filter_set(*cfg["photometry"]["sets"]),
        min_coverage=float(cfg["photometry"].get("min_coverage", 0.99)))
    shared = [b for b in band_op.names if b in band_op_native.names]
    floor = np.abs(band_op.subset(shared).magnitudes(X_true)
                   - band_op_native.subset(shared).magnitudes(X_native)) * 1000.0
    floor_stats = {"median_mmag": float(np.nanmedian(floor)),
                   "p99_mmag": float(np.nanpercentile(floor, 99.0)),
                   "max_mmag": float(np.nanmax(floor)),
                   "n_bands": len(shared)}
    print(f"  grid floor   resampling alone moves magnitudes by "
          f"{floor_stats['median_mmag']:.3f} mmag (median), "
          f"{floor_stats['p99_mmag']:.3f} (p99), {floor_stats['max_mmag']:.3f} (max)")

    results = {
        "name": name,
        "config": _jsonable(cfg),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "grid": {
            "wave_min": float(wave_eval.min()), "wave_max": float(wave_eval.max()),
            "n_bins": int(wave_eval.size), "resolution_report": _jsonable(res_rep),
            "native_R_at_5500A": float(np.interp(5500.0, wave_native,
                                                 native_resolution(wave_native))),
        },
        "resampling_floor": floor_stats,
        "bands": {"n_usable": len(band_op), "names": band_op.names,
                  "rejected": band_op.rejected, "membership": membership},
        "colour_cut": {"value": float(cut), "definition": cut_name},
        "runs": [],
        "baselines": [],
    }
    if gal is not None:
        results["galaxy_setup"] = _jsonable(gal["info"])

    # ── weighting schemes x component scan ────────────────────────────────
    n_max = int(cfg["components"]["max"])
    scan = sorted({int(n) for n in cfg["components"]["scan"] if int(n) <= n_max})

    for wspec in cfg["weighting"]:
        wspec = dict(wspec)
        scheme = wspec.pop("scheme")
        label_base = wspec.pop("label", scheme)
        print(f"\n  ── weighting '{label_base}' ({scheme}) " + "─" * 34)
        t0 = time.time()
        basis_full = SpectralBasis.fit(wave_eval, X_true, n_max, weight_scheme=scheme,
                                       weight_kwargs=wspec or None)
        print(f"     fitted {n_max} components in {time.time() - t0:.1f} s; "
              f"weight dynamic range {basis_full.meta['weighting']['w_dynamic_range']:.3g}")

        # PCA components are nested, so every point on the curve is a truncation of
        # the same fit rather than a separate decomposition.
        coeffs_full = basis_full.transform(X_true)
        ones = np.ones(X_true.shape[0])

        for n_pc in scan:
            b = basis_full.truncate(n_pc)
            X_hat = b.reconstruct(coeffs_full[:, :n_pc], ones)
            r = evaluate_reconstruction(
                f"{label_base} / {n_pc} PC", b.bytes_per_galaxy(np.float32),
                X_true, X_hat, wave_eval, band_op, membership, d4000_ops, nb_ops,
                params, gal, cut, n_components=n_pc)
            r["weighting"] = {"scheme": scheme, "label": label_base,
                              **_jsonable(basis_full.meta["weighting"])}
            r["explained_variance_cumulative"] = float(
                basis_full.explained_variance_ratio[:n_pc].sum())
            results["runs"].append(r)

            c = r.get("galaxies", {}).get("colour", {})
            gal_mad = r.get("galaxies", {}).get("overall", {}).get("mad")
            print(f"     N={n_pc:4d}  {r['bytes_per_galaxy']:5d} B   "
                  f"SSP MAD {r['ssp_overall']['mad']:9.3f} mmag   "
                  f"gal MAD {_fmt(gal_mad, 3):>9} mmag   "
                  f"Δf_red {_fmt(c.get('red_fraction_shift'), 4):>8}   "
                  f"cumvar {r['explained_variance_cumulative']:.9f}")

        if cfg["output"].get("save_bases"):
            basis_full.save(out_dir / f"{name}_basis_{label_base}.npz")

    # ── baselines ─────────────────────────────────────────────────────────
    print("\n  ── baselines " + "─" * 52)

    lbl, nbytes, X_hat = baseline_full_spectrum(X_true, wave_eval)
    results["baselines"].append(evaluate_reconstruction(
        lbl, nbytes, X_true, X_hat, wave_eval, band_op, membership, d4000_ops,
        nb_ops, params, gal, cut))
    print(f"     {lbl:<44} {nbytes:6d} B   lossless by construction")

    cb = baseline_committed_basis(cfg, wave_eval, M, wave_native)
    if cb is not None:
        lbl, nbytes, X_hat, n_pc = cb
        r = evaluate_reconstruction(lbl, nbytes, X_true, X_hat, wave_eval, band_op,
                                    membership, d4000_ops, nb_ops, params, gal, cut,
                                    n_components=n_pc)
        results["baselines"].append(r)
        c = r.get("galaxies", {}).get("colour", {})
        print(f"     {lbl:<44} {nbytes:6d} B   SSP MAD {r['ssp_overall']['mad']:9.3f} mmag   "
              f"Δf_red {_fmt(c.get('red_fraction_shift'), 4)}")

    bb = baseline_broadband_only(X_true, band_op, cfg["baselines"].get(
        "broadband_bands", ["u", "g", "r", "i", "z"]))
    if bb is not None:
        lbl, nbytes, X_hat = bb
        r = evaluate_reconstruction(lbl, nbytes, X_true, X_hat, wave_eval, band_op,
                                    membership, d4000_ops, nb_ops, params, gal, cut)
        results["baselines"].append(r)
        print(f"     {lbl:<44} {nbytes:6d} B   SSP MAD {r['ssp_overall']['mad']:9.3f} mmag")

    if gal is not None:
        index = SSPGridIndex(params, param_names)
        for nb in cfg["baselines"].get("sfh_bins", []):
            lbl, nbytes, W_coarse = baseline_binned_sfh(gal, index, int(nb))
            r = {"label": lbl, "bytes_per_galaxy": int(nbytes),
                 "note": "reproduces the library exactly; error comes from coarsening the "
                         "star-formation history, not from spectral compression",
                 "galaxies": galaxy_metrics_crossW(X_true, band_op, membership,
                                                   gal["W"], W_coarse, cut)}
            results["baselines"].append(r)
            c = r["galaxies"].get("colour", {})
            print(f"     {lbl:<44} {nbytes:6d} B   gal MAD "
                  f"{_fmt(r['galaxies']['overall']['mad'], 3):>9} mmag   "
                  f"Δf_red {_fmt(c.get('red_fraction_shift'), 4)}")

    # ── write ─────────────────────────────────────────────────────────────
    json_path = out_dir / f"{name}_results.json"
    json_path.write_text(json.dumps(_jsonable(results), indent=2) + "\n")
    print(f"\n  results -> {json_path}")

    csv_path = out_dir / f"{name}_error_vs_bytes.csv"
    write_curve_csv(results, csv_path)
    print(f"  curve   -> {csv_path}")

    if not args.no_figures:
        from galspectra.plotting.basis_experiment import make_all_figures
        fig_dir = _resolve(cfg["output"].get("figure_dir", "figures/pca_basis"))
        paths = make_all_figures(results, fig_dir, name)
        print(f"  figures -> {fig_dir} ({len(paths)} files)")

    print(f"\n  total {time.time() - t_start:.1f} s")
    return results


def write_curve_csv(results, path):
    cols = ["label", "weighting", "n_components", "bytes_per_galaxy",
            "ssp_mad_mmag", "ssp_p99_mmag", "gal_mad_mmag", "gal_p99_mmag",
            "paus_nb_mad_mmag", "red_fraction_shift", "red_minus_blue_mmag",
            "d4000n_mad_frac", "explained_variance_cumulative"]
    lines = [",".join(cols)]
    for r in results["runs"] + results["baselines"]:
        g = r.get("galaxies", {})
        c = g.get("colour", {})
        row = [
            r["label"],
            r.get("weighting", {}).get("label", ""),
            r.get("n_components", ""),
            r["bytes_per_galaxy"],
            r.get("ssp_overall", {}).get("mad", ""),
            r.get("ssp_overall", {}).get("p99", ""),
            g.get("overall", {}).get("mad", ""),
            g.get("overall", {}).get("p99", ""),
            r.get("ssp_by_set", {}).get("paus_nb", {}).get("mad_mmag", ""),
            c.get("red_fraction_shift", ""),
            c.get("red_minus_blue_differential_mmag", ""),
            r.get("d4000", {}).get("D4000_n/direct", {}).get("mad_frac", ""),
            r.get("explained_variance_cumulative", ""),
        ]
        lines.append(",".join(
            f"{v:.6g}" if isinstance(v, float) else ("" if v is None else str(v))
            for v in row))
    Path(path).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
