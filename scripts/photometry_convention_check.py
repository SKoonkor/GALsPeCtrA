"""
photometry_convention_check.py

Measures the filter convolution convention against L-GALAXIES' own photometric
tables, over the full BC03 age x metallicity grid and all 40 bands the model ships.

The question
------------
`documents/error_vs_bytes.md` §5 established that integrating the BC03 `FullSED`
files gives a *g-r* that is redder than the `PhotTables` L-GALAXIES itself reads —
from the same library, the same IMF and the same filter files. Roughly half of that
looked like the filter convolution convention. This measures it properly.

    energy   <f_nu> = int f_nu T dnu / int T dnu        (what GALsPeCtrA does)
    photon   <f_nu> = int f_nu T dnu/nu / int T dnu/nu  (what a CCD does)

Why the comparison is exact
---------------------------
The `FullSED` files and the `PhotTables` are on the **same 221-point age grid**
(verified to 1.2e-16 relative) and the same 6 metallicities, and
`input/Filter_Names.txt` guarantees the same filter curves. So this is a direct,
element-by-element comparison with **no interpolation anywhere**: every difference
reported is a difference in how flux became a magnitude.

Both conventions run through `galspectra.photometry.synthetic`, the production
path, so the measurement tests the code that would actually ship.

What it reports
---------------
1. Which convention reproduces the tables, under the single best zero point.
2. How the residual varies with age and with metallicity.
3. Whether what remains has the structure a BaSeL-vs-STELIB library difference
   would produce — the hypothesis in ResearchPlan_2026 §9.3.
4. **The quadrature ladder**, added 7 Aug 2026. Once the convention was settled, a
   residual of ~6 mmag remained and the STELIB explanation had failed its own
   discriminating test. Reading the code that *writes* the tables —
   `setup_Spec_LumTables_onthefly()` in `code/model_spectro_photometric.c` — showed
   it is not a library difference at all but three defects in that routine's
   quadrature. The ladder switches them on one at a time and reports how much of the
   residual each owns. See `documents/lgalaxies_quadrature.md`.

This script only measures. It changes no product; the ladder's `lgal_native` mode is
a diagnostic and must never be used for science. See `documents/filter_convention.md`.

Usage
  cd /path/to/GALsPeCtrA
  python scripts/photometry_convention_check.py
  python scripts/photometry_convention_check.py --no-figures
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

from galspectra.pca.metrics import BandOperator                        # noqa: E402
from galspectra.photometry.filter_sets import load_curve               # noqa: E402
from galspectra.photometry.lgal_quadrature import (                    # noqa: E402
    lgal_band_matrix, lgal_magnitudes_from_matrix,
)
from galspectra.photometry.synthetic import CONVENTIONS                # noqa: E402
from galspectra.sps.bc03_backend import BC03_METALLICITIES, BC03Library  # noqa: E402

LGAL_ROOT = PROJECT_ROOT.parent / "L-GALAXIES" / "LGalaxies2020_PublicRepository-master"
FULLSED_DIR = LGAL_ROOT / "SpecPhotTables" / "FullSEDs"
PHOT_TABLES = LGAL_ROOT / "SpecPhotTables" / "PhotTables"
FILTER_LIST = LGAL_ROOT / "input" / "Filter_Names.txt"
FILTER_DIR = LGAL_ROOT / "SpecPhotTables" / "Filters"

#: The colour the red/blue split uses. SDSS g and r carry an 's' suffix in
#: L-GALAXIES' naming, separating them from the Johnson set.
COLOUR_PAIR = ("gs", "rs")

#: The five SDSS bands under that same naming — the only ones this programme uses,
#: and therefore the row to read in the quadrature ladder.
SDSS_TOKENS = ("us", "gs", "rs", "is", "zs")

#: BC03's empirical STELIB library covers this range at ~3 Å; outside it the
#: models fall back to theoretical (BaSeL) spectra.
STELIB_RANGE_AA = (3200.0, 9500.0)

AGE_EDGES_GYR = np.array([0.0, 0.01, 0.1, 0.5, 1.0, 3.0, 6.0, 10.0, 20.001])


# ─────────────────────────────────────────────────────────────────────────────

def load_band_registry():
    """Every band with both a filter curve and a Millennium-I photometric table.

    Driven by `input/Filter_Names.txt`, the mapping L-GALAXIES itself reads, so
    the curve used here is the curve the tables were built from.

    Using all 40 bands rather than SDSS ugriz alone is what makes the
    wavelength-structure test possible: every SDSS pivot falls **inside** STELIB's
    3200-9500 Å empirical range, so ugriz on its own cannot distinguish "the
    residual lives where the empirical library lives" from "the residual is
    everywhere". GALEX FUV at ~1500 Å and IRAC at 8 µm can.
    """
    bands = {}
    for raw in FILTER_LIST.read_text().splitlines()[1:]:
        parts = raw.split()
        if len(parts) != 3:
            continue
        fname, nominal_um, token = parts
        curve = FILTER_DIR / fname
        table = PHOT_TABLES / f"BC03_Chabrier_Phot_Table_MR_Mag{token}_m0.0200.dat"
        if not (curve.exists() and table.exists()):
            continue
        wave, trans = load_curve(curve, "AA")
        # Pivot from the curve itself, not the nominal value in the list: the
        # wavelength test should use where the band actually sits.
        pivot = float(np.sqrt(np.trapezoid(trans * wave, wave)
                              / np.trapezoid(trans / wave, wave)))
        bands[token] = {"file": fname, "nominal_um": float(nominal_um),
                        "pivot_AA": pivot, "curve": (wave, trans)}
    if not bands:
        raise SystemExit(f"no usable bands found via {FILTER_LIST}")
    return dict(sorted(bands.items(), key=lambda kv: kv[1]["pivot_AA"]))


def read_phot_table(band_token, metallicity, sim="MR", prefix="BC03_Chabrier"):
    """Return (ages_yr, magnitudes, redshift) for the z~0 snapshot.

    Layout, from `setup_LumTables_precomputed()` in
    `code/model_spectro_photometric.c`:

        <token> <pivot_micron> <n_snapshots> <n_ages>
        <n_ages ages in yr>
        for each snapshot:  <redshift>  <n_ages magnitudes>
    """
    path = PHOT_TABLES / f"{prefix}_Phot_Table_{sim}_Mag{band_token}_m{metallicity:.4f}.dat"
    tok = np.array(path.read_text().split())
    n_snap, n_age = int(tok[2]), int(tok[3])
    ages_yr = tok[4:4 + n_age].astype(float)
    block = tok[4 + n_age:].astype(float).reshape(n_snap, n_age + 1)
    i0 = int(np.argmin(np.abs(block[:, 0])))
    return ages_yr, block[i0, 1:], float(block[i0, 0])


def measure(registry):
    """Magnitudes for every (convention, metallicity, age, band), plus the tables."""
    filters = {tok: info["curve"] for tok, info in registry.items()}
    lib = BC03Library(FULLSED_DIR)

    out = {"ages_gyr": None, "metallicities": [float(z) for z in BC03_METALLICITIES],
           "bands": None, "mine": {c: {} for c in CONVENTIONS}, "table": {},
           "rejected": []}

    ops = None
    for Z in BC03_METALLICITIES:
        ages_gyr, wave, flux = lib.get(Z)      # (n_age,), (n_wave,), (n_age, n_wave)
        if ops is None:
            ops = {c: BandOperator(wave, filters, min_coverage=0.99, convention=c)
                   for c in CONVENTIONS}
            assert ops["energy"].names == ops["photon"].names
            out["bands"] = list(ops["energy"].names)
            out["rejected"] = list(ops["energy"].rejected)
            out["ages_gyr"] = ages_gyr.tolist()
            out["n_wave"] = int(wave.size)
            out["wave_range_AA"] = [float(wave.min()), float(wave.max())]
        else:
            assert np.allclose(ages_gyr, out["ages_gyr"]), "age grid varies with Z"

        for c in CONVENTIONS:
            out["mine"][c][f"{Z:.4f}"] = ops[c].magnitudes(flux)

        tab = np.full((len(ages_gyr), len(out["bands"])), np.nan)
        for j, tok in enumerate(out["bands"]):
            ages_yr, mags, z0 = read_phot_table(tok, Z)
            assert np.allclose(ages_yr, np.asarray(out["ages_gyr"]) * 1e9,
                               rtol=1e-9, atol=0), f"{tok}: table age grid differs"
            tab[:, j] = mags
            out["table_redshift"] = z0
        out["table"][f"{Z:.4f}"] = tab

    return out


#: The quadrature ladder: each step adds one defect of L-GALAXIES' own integration.
#: `None` means the production path (the adopted photon convention) — the baseline the
#: residual is measured down from.
#:
#: The order matters. The measure comes first because it is the largest single term;
#: the step size second because it only bites once the measure is wrong; the filter
#: sampling last. They are **not independent** — see `photon + no step size`, a control
#: that shows the missing step size on its own makes the agreement slightly *worse*.
QUADRATURE_LADDER = (
    ("photon (production)", None),
    ("+ dlambda measure",
     dict(measure="dlambda", use_step_size=True, step_filter=False)),
    ("+ no step size",
     dict(measure="dlambda", use_step_size=False, step_filter=False)),
    ("+ nearest-index filter (= lgal_native)",
     dict(measure="dlambda", use_step_size=False, step_filter=True)),
    ("control: photon + no step size",
     dict(measure="dnu_over_nu", use_step_size=False, step_filter=False)),
)


def quadrature_ladder(data, registry):
    """How much of the post-convention residual each C defect owns.

    Evaluated through `lgal_band_matrix`, which is the same quadrature expressed as a
    matrix — L-GALAXIES' integral is still linear in f_lambda, so 221 x 6 x 40
    magnitudes are one matrix product per step rather than 53,000 Python loops.
    """
    filters = {tok: info["curve"] for tok, info in registry.items()}
    lib = BC03Library(FULLSED_DIR)
    bands = data["bands"]
    sdss = [j for j, b in enumerate(bands) if b in SDSS_TOKENS]

    per_step = {name: [] for name, _ in QUADRATURE_LADDER}
    mats = None
    for Z in BC03_METALLICITIES:
        _ages, wave, flux = lib.get(Z)
        if mats is None:
            mats = {}
            for name, kw in QUADRATURE_LADDER:
                if kw is None:
                    continue
                nm, P, den = lgal_band_matrix(wave, filters, **kw)
                idx = [nm.index(b) for b in bands]
                mats[name] = (P[idx], den[idx])
        for name, kw in QUADRATURE_LADDER:
            if kw is None:
                per_step[name].append(data["mine"]["photon"][f"{Z:.4f}"])
            else:
                P, den = mats[name]
                per_step[name].append(lgal_magnitudes_from_matrix(flux, P, den))

    tab = np.concatenate([data["table"][f"{Z:.4f}"] for Z in BC03_METALLICITIES], axis=0)

    out = {"steps": [], "bands": bands}
    per_band_first = per_band_last = None
    for name, _kw in QUADRATURE_LADDER:
        mag = np.concatenate(per_step[name], axis=0)
        ok = np.isfinite(mag) & np.isfinite(tab)
        delta = mag - tab
        # One global zero point, exactly as the convention comparison does: the two
        # products carry different flux units and that is one constant.
        resid = np.abs(delta - np.nanmedian(delta[ok]))
        s = resid[:, sdss][ok[:, sdss]]
        a = resid[ok]
        out["steps"].append({
            "step": name,
            "sdss_median": float(np.median(s)), "sdss_rms": float(np.sqrt(np.mean(s ** 2))),
            "sdss_max": float(s.max()),
            "all_median": float(np.median(a)), "all_rms": float(np.sqrt(np.mean(a ** 2))),
        })
        if per_band_first is None:
            per_band_first = (resid, ok)
        if name.startswith("+ nearest"):
            per_band_last = (resid, ok)

    if per_band_last is not None:
        r0, ok0 = per_band_first
        r1, ok1 = per_band_last
        out["per_band"] = {
            b: {"photon": float(np.median(r0[ok0[:, j], j])),
                "lgal_native": float(np.median(r1[ok1[:, j], j]))}
            for j, b in enumerate(bands) if ok0[:, j].any()
        }
        out["n_bands_under_1mmag"] = sum(
            1 for v in out["per_band"].values() if v["lgal_native"] < 1e-3)
    return out


def duplicate_wavelength_audit(registry):
    """Filter curves with a repeated wavelength, which breaks the C's bisection.

    `locate()` (`model_misc.c:3923`) is a bisection and assumes a strictly monotonic
    array. Half of the shipped curves are not, and that is what separates the bands
    whose residual the ladder closes from those it does not.
    """
    out = {}
    for tok, info in registry.items():
        wave, _t = info["curve"]
        d = np.diff(wave)
        out[tok] = {"n_points": int(wave.size),
                    "n_duplicate": int((d == 0).sum()),
                    "strictly_increasing": bool(np.all(d > 0))}
    return out


def _stack(data, conv=None):
    """(n_Z, n_age, n_band) over metallicities, in registry order."""
    src = data["mine"][conv] if conv else data["table"]
    return np.stack([src[f"{Z:.4f}"] for Z in BC03_METALLICITIES])


def analyse(data, registry, min_age_gyr=0.001):
    """Residuals under the single best zero point, sliced by age, band and Z."""
    ages = np.asarray(data["ages_gyr"])
    bands = data["bands"]
    tab = _stack(data)
    usable = ages >= min_age_gyr
    ig, ir = bands.index(COLOUR_PAIR[0]), bands.index(COLOUR_PAIR[1])

    res = {"min_age_gyr": min_age_gyr, "n_ages_used": int(usable.sum()),
           "n_bands": len(bands), "bands": bands, "conventions": {}}

    for c in CONVENTIONS:
        mine = _stack(data, conv=c)
        delta = mine - tab
        delta[:, ~usable, :] = np.nan

        # One global zero point: the two products carry different flux units, and
        # that is a single constant. Whatever survives removing it is a genuine
        # disagreement about the shape of the spectrum.
        zp = float(np.nanmedian(delta))
        resid = delta - zp

        entry = {
            "zero_point": zp,
            "n_finite": int(np.isfinite(delta).sum()),
            "overall": {
                "median_abs": float(np.nanmedian(np.abs(resid))),
                "rms": float(np.sqrt(np.nanmean(resid ** 2))),
                "p99_abs": float(np.nanpercentile(np.abs(resid), 99.0)),
                "max_abs": float(np.nanmax(np.abs(resid))),
            },
            "by_band": {}, "by_metallicity": {}, "by_age": [],
        }

        for j, tok in enumerate(bands):
            pivot = registry[tok]["pivot_AA"]
            entry["by_band"][tok] = {
                "pivot_AA": pivot,
                "inside_stelib": bool(STELIB_RANGE_AA[0] <= pivot <= STELIB_RANGE_AA[1]),
                "median": float(np.nanmedian(resid[:, :, j])),
                "median_abs": float(np.nanmedian(np.abs(resid[:, :, j]))),
                "rms": float(np.sqrt(np.nanmean(resid[:, :, j] ** 2))),
            }

        for k, Z in enumerate(BC03_METALLICITIES):
            entry["by_metallicity"][f"{Z:.4f}"] = {
                "logzsol": float(np.log10(Z / 0.02)),
                "median_abs": float(np.nanmedian(np.abs(resid[k]))),
                "rms": float(np.sqrt(np.nanmean(resid[k] ** 2))),
            }

        for lo, hi in zip(AGE_EDGES_GYR[:-1], AGE_EDGES_GYR[1:]):
            sel = (ages >= lo) & (ages < hi) & usable
            if not sel.any():
                continue
            entry["by_age"].append({
                "age_lo_Gyr": float(lo), "age_hi_Gyr": float(hi), "n_ages": int(sel.sum()),
                "median_abs": float(np.nanmedian(np.abs(resid[:, sel, :]))),
                "rms": float(np.sqrt(np.nanmean(resid[:, sel, :] ** 2))),
            })

        # g-r is what the red/blue split uses, and it is zero-point free.
        gr = (mine[:, :, ig] - mine[:, :, ir]) - (tab[:, :, ig] - tab[:, :, ir])
        gr[:, ~usable] = np.nan
        old, young = ages > 5.0, (ages > 0.01) & (ages < 0.3)
        entry["colour_g_minus_r"] = {
            "median_all": float(np.nanmedian(gr)),
            "median_old": float(np.nanmedian(gr[:, old & usable])),
            "median_young": float(np.nanmedian(gr[:, young & usable])),
            "by_metallicity_old": {
                f"{Z:.4f}": float(np.nanmedian(gr[k, old & usable]))
                for k, Z in enumerate(BC03_METALLICITIES)},
            "by_age": [
                {"age_lo_Gyr": float(lo), "age_hi_Gyr": float(hi),
                 "median": float(np.nanmedian(gr[:, (ages >= lo) & (ages < hi) & usable]))}
                for lo, hi in zip(AGE_EDGES_GYR[:-1], AGE_EDGES_GYR[1:])
                if ((ages >= lo) & (ages < hi) & usable).any()],
        }
        res["conventions"][c] = entry

    e, p = res["conventions"]["energy"], res["conventions"]["photon"]
    res["verdict"] = {
        "better": "photon" if p["overall"]["rms"] < e["overall"]["rms"] else "energy",
        "rms_energy": e["overall"]["rms"],
        "rms_photon": p["overall"]["rms"],
        "rms_improvement_factor": e["overall"]["rms"] / p["overall"]["rms"],
        "g_minus_r_old_energy": e["colour_g_minus_r"]["median_old"],
        "g_minus_r_old_photon": p["colour_g_minus_r"]["median_old"],
        "fraction_of_old_gr_offset_removed": (
            1.0 - abs(p["colour_g_minus_r"]["median_old"])
            / abs(e["colour_g_minus_r"]["median_old"])),
        "bands_better_under_photon": int(sum(
            1 for t in res["bands"]
            if p["by_band"][t]["median_abs"] < e["by_band"][t]["median_abs"])),
    }
    return res


def stelib_test(data, registry, analysis, convention, min_age_gyr=0.001):
    """Does the residual look like a BaSeL-vs-STELIB library difference?

    Three predictions, if `FullSED` is the low-resolution product and the tables
    are not:

      1. the residual concentrates in bands inside STELIB's 3200-9500 Å range,
      2. it strengthens with metallicity — line blanketing scales with metal
         content, and
      3. it strengthens with age — cool giants dominate old populations and carry
         the deepest molecular and metal features.

    Reported as measurements. Two of three is not a confirmation.
    """
    ages = np.asarray(data["ages_gyr"])
    bands = data["bands"]
    usable = ages >= min_age_gyr
    resid = (_stack(data, conv=convention) - _stack(data)
             - analysis["conventions"][convention]["zero_point"])
    resid[:, ~usable, :] = np.nan

    inside = [j for j, t in enumerate(bands)
              if STELIB_RANGE_AA[0] <= registry[t]["pivot_AA"] <= STELIB_RANGE_AA[1]]
    outside = [j for j in range(len(bands)) if j not in inside]

    zs = np.array([np.log10(Z / 0.02) for Z in BC03_METALLICITIES])
    per_Z = np.array([np.nanmedian(np.abs(resid[k])) for k in range(len(zs))])
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            # Ages below `min_age_gyr` are all-NaN by construction; that is the
            # point of the mask, not a problem to warn about.
            warnings.simplefilter("ignore", RuntimeWarning)
            per_age = np.array([np.nanmedian(np.abs(resid[:, i, :]))
                                for i in range(ages.size)])
    old, young = ages > 5.0, (ages > 0.01) & (ages < 0.3)

    def corr(x, y):
        ok = np.isfinite(x) & np.isfinite(y)
        return float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 2 else float("nan")

    m_in = float(np.nanmedian(np.abs(resid[:, :, inside]))) if inside else float("nan")
    m_out = float(np.nanmedian(np.abs(resid[:, :, outside]))) if outside else float("nan")

    return {
        "convention": convention,
        "prediction_1_wavelength": {
            "n_bands_inside": len(inside), "n_bands_outside": len(outside),
            "bands_inside": [bands[j] for j in inside],
            "bands_outside": [bands[j] for j in outside],
            "median_abs_inside": m_in, "median_abs_outside": m_out,
            "ratio_inside_over_outside": float(m_in / m_out) if m_out else None,
            "supports_hypothesis": bool(m_in > m_out),
        },
        "prediction_2_metallicity": {
            "median_abs_by_logzsol": {f"{z:+.3f}": float(v) for z, v in zip(zs, per_Z)},
            "correlation_with_logzsol": corr(zs, per_Z),
            "ratio_highest_over_lowest_Z": float(per_Z[-1] / per_Z[0]),
            "supports_hypothesis": bool(per_Z[-1] > per_Z[0]),
        },
        "prediction_3_age": {
            "median_abs_young_0p01_0p3_Gyr": float(np.nanmedian(np.abs(resid[:, young, :]))),
            "median_abs_old_gt5_Gyr": float(np.nanmedian(np.abs(resid[:, old, :]))),
            "ratio_old_over_young": float(np.nanmedian(np.abs(resid[:, old, :]))
                                          / np.nanmedian(np.abs(resid[:, young, :]))),
            "correlation_with_log_age": corr(np.log10(np.maximum(ages, 1e-6)), per_age),
            "supports_hypothesis": bool(np.nanmedian(np.abs(resid[:, old, :]))
                                        > np.nanmedian(np.abs(resid[:, young, :]))),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────

def make_figures(data, registry, analysis, fig_dir, winner):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    ages = np.asarray(data["ages_gyr"])
    bands = data["bands"]
    usable = ages >= analysis["min_age_gyr"]
    tab = _stack(data)
    ig, ir = bands.index(COLOUR_PAIR[0]), bands.index(COLOUR_PAIR[1])

    ZCOL = ["#0072B2", "#D55E00", "#009E73", "#56B4E9", "#CC79A7", "#4d4d4d"]
    INK, MUTED, GRID = "#1a1a1a", "#666666", "#d9d9d9"

    def style(ax, xl=None, yl=None, ti=None):
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        if xl: ax.set_xlabel(xl, fontsize=9.5, color=INK)
        if yl: ax.set_ylabel(yl, fontsize=9.5, color=INK)
        if ti: ax.set_title(ti, fontsize=10.5, color=INK, loc="left", pad=8)

    # ── fig 1: g-r residual vs age, per convention ───────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7), sharey=True)
    for ax, conv in zip(axes, CONVENTIONS):
        mine = _stack(data, conv=conv)
        gr = (mine[:, :, ig] - mine[:, :, ir]) - (tab[:, :, ig] - tab[:, :, ir])
        for k, Z in enumerate(BC03_METALLICITIES):
            ax.plot(ages[usable], gr[k, usable], "-", color=ZCOL[k], linewidth=1.7,
                    label=f"Z = {Z:g}")
        ax.axhline(0.0, color=INK, linewidth=1.0, linestyle="--")
        ax.set_xscale("log")
        med = analysis["conventions"][conv]["colour_g_minus_r"]["median_old"]
        style(ax, "SSP age  (Gyr)",
              "Δ(g−r)   FullSED − PhotTables   (mag)" if conv == "energy" else None,
              f"{conv}      median at age > 5 Gyr: {med:+.4f}")
    axes[0].legend(fontsize=7.5, frameon=False, ncol=2, loc="upper left")
    fig.suptitle("Filter convolution convention against L-GALAXIES' own tables",
                 fontsize=12, color=INK, x=0.06, ha="left", y=0.99)
    fig.text(0.06, 0.925, "Same library, same IMF, same filter files, same 221 ages "
             "— no interpolation anywhere.", fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(fig_dir / "figA1_gr_residual_vs_age.png", dpi=180, facecolor="white")
    plt.close(fig)

    # ── fig 2: residual structure — the STELIB test ──────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.3))
    for conv, mk, col in (("energy", "o", "#0072B2"), ("photon", "s", "#D55E00")):
        e = analysis["conventions"][conv]
        piv = [e["by_band"][t]["pivot_AA"] for t in bands]
        axes[0].plot(piv, [e["by_band"][t]["median_abs"] for t in bands],
                     marker=mk, color=col, markersize=5, markeredgecolor="white",
                     markeredgewidth=0.6, linestyle="none", label=conv)
        axes[1].plot([e["by_metallicity"][f"{Z:.4f}"]["logzsol"] for Z in BC03_METALLICITIES],
                     [e["by_metallicity"][f"{Z:.4f}"]["median_abs"] for Z in BC03_METALLICITIES],
                     "-", marker=mk, color=col, linewidth=1.9, markersize=6,
                     markeredgecolor="white", label=conv)
        axes[2].plot([0.5 * (a["age_lo_Gyr"] + min(a["age_hi_Gyr"], 20.0)) for a in e["by_age"]],
                     [a["median_abs"] for a in e["by_age"]],
                     "-", marker=mk, color=col, linewidth=1.9, markersize=6,
                     markeredgecolor="white", label=conv)
    axes[0].axvspan(*STELIB_RANGE_AA, color="#cccccc", alpha=0.4, zorder=0)
    axes[0].annotate("STELIB empirical range", xy=(np.sqrt(np.prod(STELIB_RANGE_AA)),
                     axes[0].get_ylim()[1]), xytext=(0, -10), textcoords="offset points",
                     fontsize=7.5, color=MUTED, ha="center", va="top")
    axes[0].set_xscale("log")
    for ax in axes:
        ax.set_yscale("log")
        ax.legend(fontsize=8, frameon=False)
    style(axes[0], "band pivot wavelength  (Å)", "median |residual|  (mag)",
          f"By band  ({len(bands)} bands)")
    style(axes[1], "log(Z/Z☉)", None, "By metallicity")
    style(axes[2], "SSP age  (Gyr)", None, "By age")
    axes[2].set_xscale("log")
    fig.suptitle(f"What remains after adopting '{winner}' — the BaSeL-vs-STELIB test",
                 fontsize=11.5, color=INK, x=0.04, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(fig_dir / "figA2_residual_structure.png", dpi=180, facecolor="white")
    plt.close(fig)

    return [fig_dir / "figA1_gr_residual_vs_age.png",
            fig_dir / "figA2_residual_structure.png"]


def report(analysis, stelib):
    v = analysis["verdict"]
    print("\n" + "=" * 78)
    print("1. WHICH CONVENTION REPRODUCES THE TABLES?")
    print("=" * 78)
    print(f"  {'':24s} {'energy':>11} {'photon':>11}")
    for key, lbl in (("median_abs", "median |residual|"), ("rms", "RMS residual"),
                     ("p99_abs", "p99 |residual|"), ("max_abs", "max |residual|")):
        print(f"  {lbl:24s} "
              f"{analysis['conventions']['energy']['overall'][key]:11.5f} "
              f"{analysis['conventions']['photon']['overall'][key]:11.5f}")
    print(f"\n  WINNER: {v['better'].upper()}   RMS better by "
          f"{v['rms_improvement_factor']:.2f}x")
    print(f"  better in {v['bands_better_under_photon']} of {analysis['n_bands']} bands")
    print(f"  g-r at age > 5 Gyr:  energy {v['g_minus_r_old_energy']:+.4f}   "
          f"photon {v['g_minus_r_old_photon']:+.4f}   "
          f"({100 * v['fraction_of_old_gr_offset_removed']:.0f}% removed)")

    print("\n" + "=" * 78)
    print("2. RESIDUAL vs AGE AND METALLICITY, under the better convention")
    print("=" * 78)
    b = analysis["conventions"][v["better"]]
    print(f"  {'age bin (Gyr)':<20} {'median |resid|':>15}")
    for a in b["by_age"]:
        print(f"  {a['age_lo_Gyr']:>6.2f} - {a['age_hi_Gyr']:<10.2f} {a['median_abs']:15.5f}")
    print(f"\n  {'log(Z/Zsun)':<20} {'median |resid|':>15}")
    for Z in BC03_METALLICITIES:
        m = b["by_metallicity"][f"{Z:.4f}"]
        print(f"  {m['logzsol']:>+6.3f}{'':<14} {m['median_abs']:15.5f}")

    print("\n" + "=" * 78)
    print("3. IS THE REMAINDER BaSeL-vs-STELIB?")
    print("=" * 78)
    p1, p2, p3 = (stelib["prediction_1_wavelength"], stelib["prediction_2_metallicity"],
                  stelib["prediction_3_age"])
    print(f"  1 wavelength : inside {p1['median_abs_inside']:.5f} ({p1['n_bands_inside']} bands), "
          f"outside {p1['median_abs_outside']:.5f} ({p1['n_bands_outside']} bands), "
          f"ratio {p1['ratio_inside_over_outside']:.2f}   "
          f"-> {'SUPPORTS' if p1['supports_hypothesis'] else 'CONTRADICTS'}")
    print(f"  2 metallicity: corr = {p2['correlation_with_logzsol']:+.3f}, "
          f"high/low Z = {p2['ratio_highest_over_lowest_Z']:.2f}x   "
          f"-> {'SUPPORTS' if p2['supports_hypothesis'] else 'CONTRADICTS'}")
    print(f"  3 age        : old/young = {p3['ratio_old_over_young']:.2f}x, "
          f"corr = {p3['correlation_with_log_age']:+.3f}   "
          f"-> {'SUPPORTS' if p3['supports_hypothesis'] else 'CONTRADICTS'}")
    n = sum(p["supports_hypothesis"] for p in (p1, p2, p3))
    print(f"\n  {n} of 3 predictions supported")


def report_ladder(ladder, curves):
    """Print the quadrature decomposition."""
    print()
    print("=" * 78)
    print("QUADRATURE — how much of the residual is L-GALAXIES' own integration")
    print("=" * 78)
    print("  Each step adds one defect of setup_Spec_LumTables_onthefly(), the routine")
    print("  that writes the tables. Residual after removing one global zero point.")
    print()
    print(f"  {'step':40s}{'ugriz med':>11}{'ugriz max':>11}{'all med':>11}")
    for s in ladder["steps"]:
        mark = "  <-- full port" if s["step"].startswith("+ nearest") else ""
        print(f"  {s['step']:40s}{s['sdss_median']:11.5f}{s['sdss_max']:11.5f}"
              f"{s['all_median']:11.5f}{mark}")

    if "per_band" in ladder:
        print(f"\n  bands reaching < 1 mmag under the full port: "
              f"{ladder['n_bands_under_1mmag']} of {len(ladder['per_band'])}")
        left = {b: v for b, v in ladder["per_band"].items() if v["lgal_native"] >= 1e-3}
        if left:
            worst = sorted(left.items(), key=lambda kv: -kv[1]["lgal_native"])
            print("  not closed:")
            for b, v in worst[:6]:
                dup = curves.get(b, {}).get("n_duplicate", 0)
                note = "duplicate wavelength in curve" if dup else "curve is clean"
                print(f"    {b:10s} {v['photon']:8.5f} -> {v['lgal_native']:8.5f}   {note}")

    n_dup = sum(1 for v in curves.values() if v["n_duplicate"])
    print(f"\n  filter curves with a repeated wavelength: {n_dup} of {len(curves)}")
    print("  locate() (model_misc.c:3923) is a bisection and assumes strict monotonicity,")
    print("  which is why those bands are the ones the port does not fully reproduce.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Measure the filter convolution convention against L-GALAXIES' tables")
    p.add_argument("--json", default=str(PROJECT_ROOT / "data" / "filter_convention.json"))
    p.add_argument("--figure-dir", default=str(PROJECT_ROOT / "figures" / "filter_convention"))
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--min-age-gyr", type=float, default=0.001,
                   help="Ignore ages below this; the youngest table entries are zero flux")
    return p.parse_args()


def main():
    args = parse_args()
    for d in (FULLSED_DIR, PHOT_TABLES, FILTER_LIST):
        if not d.exists():
            raise SystemExit(
                f"needs the full L-GALAXIES distribution on disk; missing {d}\n"
                "FullSEDs and PhotTables are excluded from the git repository for size.")

    print("=" * 78)
    print("FILTER CONVOLUTION CONVENTION — MEASUREMENT")
    print("=" * 78)
    registry = load_band_registry()
    print(f"  bands      {len(registry)} from {FILTER_LIST.name}, "
          f"pivots {min(v['pivot_AA'] for v in registry.values()):.0f}"
          f"-{max(v['pivot_AA'] for v in registry.values()):.0f} Å")

    data = measure(registry)
    if data["rejected"]:
        print(f"  rejected   {data['rejected']} (coverage < 99%)")
    print(f"  grid       {len(data['ages_gyr'])} ages x {len(data['metallicities'])} "
          f"metallicities x {len(data['bands'])} bands, {data['n_wave']} wavelengths")
    print(f"  tables at  z = {data['table_redshift']:.4f}")
    print("  no interpolation: FullSED and PhotTables share the age grid exactly")

    analysis = analyse(data, registry, min_age_gyr=args.min_age_gyr)
    winner = analysis["verdict"]["better"]
    stelib = stelib_test(data, registry, analysis, winner, min_age_gyr=args.min_age_gyr)
    report(analysis, stelib)

    ladder = quadrature_ladder(data, registry)
    curves = duplicate_wavelength_audit(registry)
    report_ladder(ladder, curves)

    payload = {
        "generated_by": "scripts/photometry_convention_check.py",
        "grid": {"n_ages": len(data["ages_gyr"]),
                 "metallicities": data["metallicities"],
                 "n_bands": len(data["bands"]), "bands": data["bands"],
                 "rejected_bands": data["rejected"],
                 "n_wave": data["n_wave"], "wave_range_AA": data["wave_range_AA"],
                 "table_redshift": data["table_redshift"],
                 "interpolation": "none — FullSED and PhotTables share the age grid"},
        "band_pivots_AA": {t: v["pivot_AA"] for t, v in registry.items()},
        "analysis": analysis,
        "stelib_hypothesis": stelib,
        "quadrature_ladder": ladder,
        "filter_curve_audit": curves,
    }
    Path(args.json).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n  results -> {args.json}")

    if not args.no_figures:
        for p in make_figures(data, registry, analysis, args.figure_dir, winner):
            print(f"  figure  -> {p}")

    print("\n  Nothing was changed. The adopted default is unaffected; 'lgal_native'")
    print("  is a diagnostic and is never used for science.")
    return payload


if __name__ == "__main__":
    main()
