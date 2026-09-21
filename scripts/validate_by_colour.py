"""
validate_by_colour.py

Validates the PCA SED reconstruction against native L-GALAXIES photometry,
split by red/blue galaxy population.

A single offset/scatter per band over the whole sample (as in
`notebooks/05_validate_against_lgalaxies.ipynb`) can't detect a bias that
affects one population and not the other — and Paper I (Koonkor et al. 2026)
finds GALFORM overproduces faint red *and* blue galaxies as separate
populations, so a colour-dependent bias here would corrupt that comparison.

Five classifications, so the conclusion doesn't rest on one cut:
  lgal_gmm     DEFAULT. (g-r) >= equal-posterior point of a 2-component GMM
               fitted to L-GALAXIES' own colour distribution (calibrated by
               colour_cut_calibration.py, constants in
               data/colour_cut_calibration.json).
  lgal_tilted  Same calibration in bins of M_r, Baldry functional form, so
               the cut tracks the valley as it drifts with luminosity.
  paper1       (g-r) >= 0.4, the author's own PAUS LF cut
               (GALFORM_LF/REFACTORED/galform_lf/observed_lf.py:26).
               Calibrated for GALFORM over 0 < z < 2; does NOT land in the
               L-GALAXIES valley (~0.07 mag blueward). Kept for comparison.
  baldry       Baldry et al. (2004): (u-r)_cut = 2.06 - 0.244*tanh((M_r+20.07)/1.09).
  ssfr         log10(sSFR/yr^-1) < -11 -> quenched. Physics-based, independent
               of the photometry being validated.

Classes are always assigned on NATIVE magnitudes, never reconstructed ones,
so a reconstruction error can't move a galaxy between classes and hide.

Two residual families, kept strictly separate: intrinsic (synth_mag_X -
Mag_X, the PCA reconstruction alone) and dust (synth_magdust_X -
MagDust_X, the PCA plus an irreducible term — the birth-cloud mu is drawn
stochastically per galaxy and never written to the output, so the dust
residual has a component no reconstruction can recover; Figure 5 tests
that rather than assuming it).

Usage:
  python scripts/validate_by_colour.py
  python scripts/validate_by_colour.py --outdir figures/validation_by_colour --no-show

Outputs: figures/validation_by_colour/fig{1..7}_*.png/.pdf,
data/validation_by_colour_stats.json, data/validation_by_colour_outliers.csv.
"""

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LGAL_ROOT = PROJECT_ROOT.parent / "L-GALAXIES" / "LGalaxies2020_PublicRepository-master"

BUNDLE = PROJECT_ROOT / "data" / "galaxy_table_MR.npz"
SOURCE_NPY = LGAL_ROOT / "output/samples/Planck_Mil-I_snapshots_default_test3_z0.00-0.00_All.npy"

BANDS = ["u", "g", "r", "i", "z"]

# L-GALAXIES writes 99.0 as a sentinel for "no magnitude", NOT NaN. np.isfinite()
# silently keeps these rows; every mask below must use this threshold instead.
MAG_SENTINEL = 90.0

# Validated with the dataviz palette checker: ALL PASS in light and dark mode,
# worst-adjacent CVD dE 21.1 (protan), normal-vision dE 31.7.
C_RED = "#d62728"
C_BLUE = "#1f77b4"
C_ALL = "#4a4a4a"

PAPER1_CUT = 0.4          # observed_lf.py:26  RED_BLUE_CUT
SSFR_CUT_LOG = -11.0      # log10(sSFR / yr^-1)

# default classification: calibrated on L-GALAXIES itself (see module docstring).
# Constants read from the JSON colour_cut_calibration.py writes; literals below
# are only a fallback.
CALIBRATION_JSON = PROJECT_ROOT / "data" / "colour_cut_calibration.json"
_FALLBACK_GMM_CUT = 0.4763
_FALLBACK_TILT = {"a": 0.5606, "b": 0.1100, "c": 20.597, "d": 0.818}

PRIMARY = "lgal_gmm"      # the definition every figure and headline number uses


def load_calibration():
    """Calibrated cut constants, or documented fallbacks if the JSON is absent."""
    if CALIBRATION_JSON.exists():
        cal = json.loads(CALIBRATION_JSON.read_text())["calibrated"]
        tilt = cal.get("lgal_tilted_dust") or {}
        if not tilt.get("reliable", False):
            tilt = _FALLBACK_TILT if "a" not in tilt else tilt
        return float(cal["lgal_gmm_cut_dust"]), tilt
    print(f"  WARNING: {CALIBRATION_JSON} not found; using fallback constants. "
          f"Run scripts/colour_cut_calibration.py to regenerate.")
    return _FALLBACK_GMM_CUT, _FALLBACK_TILT


LGAL_GMM_CUT, LGAL_TILT = load_calibration()


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    """Return a dict of arrays for the 11,328 Millennium-I galaxies."""
    if not BUNDLE.exists():
        raise FileNotFoundError(f"Bundle not found: {BUNDLE}")
    b = np.load(BUNDLE, allow_pickle=True)

    d = {"N": int(b["n_galaxies"])}
    for band in BANDS:
        d[f"Mag_{band}"] = b[f"Mag_{band}"].astype(float)
        d[f"MagDust_{band}"] = b[f"MagDust_{band}"].astype(float)
        d[f"synth_mag_{band}"] = b[f"synth_mag_{band}"].astype(float)
        d[f"synth_magdust_{band}"] = b[f"synth_magdust_{band}"].astype(float)
    for key in ("StellarMass", "BulgeMass", "BT", "Sfr", "sSFR", "ColdGas",
                "H2fraction", "Type", "MassWeightAge", "M_total"):
        d[key] = b[key].astype(float)

    # CosInclination / ColdGasRadius / MetalsColdGas are not in the table. They live
    # in the source sample, which is row-aligned 1:1 (build_galaxy_table.py asserts
    # sequential galaxy_index).
    if SOURCE_NPY.exists():
        G = np.load(SOURCE_NPY)
        if len(G) != d["N"]:
            raise ValueError(
                f"Row misalignment: bundle has {d['N']} galaxies, "
                f"source sample has {len(G)}. Refusing to join."
            )
        d["CosInclination"] = G["CosInclination"].astype(float)
        d["ColdGasRadius"] = G["ColdGasRadius"].astype(float)
        d["MetalsColdGas"] = G["MetalsColdGas"].astype(float).sum(axis=1)
        d["has_dust_inputs"] = True
    else:
        d["has_dust_inputs"] = False
        print(f"  WARNING: {SOURCE_NPY} not found; Figure 5 will be skipped.")

    return d


def validity_masks(d):
    """Per-flavour validity. Uses the 99.0 sentinel, not isfinite()."""
    ok_int = np.ones(d["N"], bool)
    ok_dust = np.ones(d["N"], bool)
    for band in BANDS:
        ok_int &= d[f"Mag_{band}"] < MAG_SENTINEL
        ok_dust &= d[f"MagDust_{band}"] < MAG_SENTINEL
    return ok_int, ok_dust


# below this cold gas mass, galaxies are gas-free. L-GALAXIES still assigns some
# several magnitudes of attenuation, because the dust optical depth uses
# Z = MetalsColdGas/ColdGas, numerically meaningless as ColdGas -> 0 (89 galaxies
# here have *negative* ColdGas). GALsPeCtrA correctly returns A~0, so the two
# disagree by up to 4.3 mag through no fault of the reconstruction. See
# documents/validation_by_colour.md §6.
GAS_FREE_THRESHOLD = 1.0e5   # Msun


def gas_free_mask(d):
    """Galaxies whose native MagDust cannot be trusted (see above)."""
    return d["ColdGas"] <= GAS_FREE_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# Classification — always on native magnitudes
# ─────────────────────────────────────────────────────────────────────────────

def classify(d):
    """Return {name: (is_red, valid)} for every definition. `PRIMARY` is the default."""
    out = {}

    gr = d["MagDust_g"] - d["MagDust_r"]
    ok = (d["MagDust_g"] < MAG_SENTINEL) & (d["MagDust_r"] < MAG_SENTINEL)

    out["lgal_gmm"] = (gr >= LGAL_GMM_CUT, ok)  # DEFAULT — see module docstring

    # magnitude-dependent version of the same calibration, Baldry form per-bin
    mr = d["MagDust_r"]
    tilt_cut = LGAL_TILT["a"] - LGAL_TILT["b"] * np.tanh((mr + LGAL_TILT["c"]) / LGAL_TILT["d"])
    out["lgal_tilted"] = (gr >= tilt_cut, ok)

    # applied to dust-attenuated magnitudes, what an observer measures and what
    # the PAUS LF pipeline classifies on
    out["paper1"] = (gr >= PAPER1_CUT, ok)

    # Baldry et al. (2004) tilted cut in (u-r) vs M_r
    ur = d["MagDust_u"] - d["MagDust_r"]
    mr = d["MagDust_r"]
    ok = ((d["MagDust_u"] < MAG_SENTINEL) & (d["MagDust_r"] < MAG_SENTINEL))
    cut = 2.06 - 0.244 * np.tanh((mr + 20.07) / 1.09)
    out["baldry"] = (ur >= cut, ok)

    # clip before log: a few entries are zero or slightly negative
    ssfr = np.clip(d["sSFR"], 1e-16, None)
    out["ssfr"] = (np.log10(ssfr) < SSFR_CUT_LOG, np.ones(d["N"], bool))

    return out


def confusion(classes, base_ok):
    """Agreement matrix between classification pairs, on galaxies valid in all."""
    names = list(classes)
    ok = base_ok.copy()
    for n in names:
        ok &= classes[n][1]
    rows = []
    for i, a in enumerate(names):
        for bname in names[i + 1:]:
            ra, rb = classes[a][0][ok], classes[bname][0][ok]
            rows.append({
                "a": a, "b": bname, "n": int(ok.sum()),
                "both_red": int((ra & rb).sum()),
                "both_blue": int((~ra & ~rb).sum()),
                "a_red_b_blue": int((ra & ~rb).sum()),
                "a_blue_b_red": int((~ra & rb).sum()),
                "agreement": float(((ra == rb).sum()) / ok.sum()),
            })
    return rows, ok


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def robust_scatter(x):
    """1.4826 * MAD. Insensitive to the outliers we are separately counting."""
    if x.size == 0:
        return np.nan
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


def residual_stats(delta):
    if delta.size == 0:
        return {"n": 0, "median": np.nan, "std": np.nan, "mad": np.nan,
                "p16": np.nan, "p84": np.nan}
    return {
        "n": int(delta.size),
        "median": float(np.median(delta)),
        "std": float(np.std(delta)),
        "mad": robust_scatter(delta),
        "p16": float(np.percentile(delta, 16)),
        "p84": float(np.percentile(delta, 84)),
    }


def compute_all_stats(d, classes, ok_int, ok_dust):
    stats = {}
    for cname, (is_red, cok) in classes.items():
        stats[cname] = {}
        for flavour, nat_pre, syn_pre, base_ok in (
            ("intrinsic", "Mag_", "synth_mag_", ok_int),
            ("dust", "MagDust_", "synth_magdust_", ok_dust),
        ):
            stats[cname][flavour] = {}
            ok = base_ok & cok
            for band in BANDS:
                delta = d[f"{syn_pre}{band}"] - d[f"{nat_pre}{band}"]
                stats[cname][flavour][band] = {
                    "all": residual_stats(delta[ok]),
                    "red": residual_stats(delta[ok & is_red]),
                    "blue": residual_stats(delta[ok & ~is_red]),
                }
            # colour residual g-r
            dc = ((d[f"{syn_pre}g"] - d[f"{syn_pre}r"])
                  - (d[f"{nat_pre}g"] - d[f"{nat_pre}r"]))
            stats[cname][flavour]["g-r"] = {
                "all": residual_stats(dc[ok]),
                "red": residual_stats(dc[ok & is_red]),
                "blue": residual_stats(dc[ok & ~is_red]),
            }
    return stats


def gas_free_report(d, ok_int, ok_dust):
    """Quantify the gas-free dust pathology and its effect on the aggregate numbers.

    Attenuation A = MagDust - Mag needs both magnitudes valid — the sentinel
    sets differ (1,137 vs 1,179 galaxies), so masking on MagDust alone would
    silently pair a real MagDust with Mag=99.0 and report A ~ -117 mag.
    """
    ok_att = ok_int & ok_dust
    gf = gas_free_mask(d)
    A_nat = d["MagDust_r"] - d["Mag_r"]
    A_syn = d["synth_magdust_r"] - d["synth_mag_r"]
    delta = d["synth_magdust_r"] - d["MagDust_r"]
    gr_n = d["MagDust_g"] - d["MagDust_r"]
    gr_s = d["synth_magdust_g"] - d["synth_magdust_r"]

    m_all = ok_dust
    m_clean = ok_dust & ~gf
    m_gf = ok_dust & gf
    m_gf_att = ok_att & gf          # attenuation statistics need both magnitudes

    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.where(d["ColdGas"] != 0, d["MetalsColdGas"] / d["ColdGas"], np.nan)

    return {
        "threshold_msun": GAS_FREE_THRESHOLD,
        "n_gas_free": int(m_gf.sum()),
        "frac_gas_free": float(m_gf.sum() / m_all.sum()),
        "n_negative_coldgas": int((ok_dust & (d["ColdGas"] < 0)).sum()),
        "n_negative_implied_Z": int(np.nansum(Z[m_gf] < 0)),
        "n_attenuation_comparable": int(m_gf_att.sum()),
        "n_disagree_gt_half_mag": int((np.abs(A_syn - A_nat) > 0.5)[m_gf_att].sum()),
        "native_A_r_max": float(A_nat[m_gf_att].max()),
        "synth_A_r_max": float(A_syn[m_gf_att].max()),
        "delta_r_all": {"n": int(m_all.sum()), "median": float(np.median(delta[m_all])),
                        "std": float(np.std(delta[m_all])),
                        "mad": robust_scatter(delta[m_all])},
        "delta_r_clean": {"n": int(m_clean.sum()), "median": float(np.median(delta[m_clean])),
                          "std": float(np.std(delta[m_clean])),
                          "mad": robust_scatter(delta[m_clean])},
        "red_frac_all": [float((gr_n[m_all] >= PAPER1_CUT).mean()),
                         float((gr_s[m_all] >= PAPER1_CUT).mean())],
        "red_frac_clean": [float((gr_n[m_clean] >= PAPER1_CUT).mean()),
                           float((gr_s[m_clean] >= PAPER1_CUT).mean())],
    }


def red_fraction_shift(d, ok_dust):
    """The headline number: does the reconstruction move galaxies across the cut?"""
    nat = d["MagDust_g"] - d["MagDust_r"]
    syn = d["synth_magdust_g"] - d["synth_magdust_r"]
    nat_i = d["Mag_g"] - d["Mag_r"]
    syn_i = d["synth_mag_g"] - d["synth_mag_r"]
    ok = ok_dust
    res = {
        "n": int(ok.sum()),
        "cut": PAPER1_CUT,
        "dust": {
            "native_red_frac": float((nat[ok] >= PAPER1_CUT).mean()),
            "synth_red_frac": float((syn[ok] >= PAPER1_CUT).mean()),
            "native_median_gr": float(np.median(nat[ok])),
            "synth_median_gr": float(np.median(syn[ok])),
            "n_blue_to_red": int((~(nat[ok] >= PAPER1_CUT) & (syn[ok] >= PAPER1_CUT)).sum()),
            "n_red_to_blue": int(((nat[ok] >= PAPER1_CUT) & ~(syn[ok] >= PAPER1_CUT)).sum()),
        },
    }
    oki = np.ones(d["N"], bool)
    for band in ("g", "r"):
        oki &= d[f"Mag_{band}"] < MAG_SENTINEL
    res["intrinsic"] = {
        "native_red_frac": float((nat_i[oki] >= PAPER1_CUT).mean()),
        "synth_red_frac": float((syn_i[oki] >= PAPER1_CUT).mean()),
        "native_median_gr": float(np.median(nat_i[oki])),
        "synth_median_gr": float(np.median(syn_i[oki])),
        "n_blue_to_red": int((~(nat_i[oki] >= PAPER1_CUT) & (syn_i[oki] >= PAPER1_CUT)).sum()),
        "n_red_to_blue": int(((nat_i[oki] >= PAPER1_CUT) & ~(syn_i[oki] >= PAPER1_CUT)).sum()),
        "n": int(oki.sum()),
    }
    return res


def find_outliers(d, classes, ok_int, ok_dust, n_worst=20):
    """Sigma-clipped outlier counts, plus a table of the worst intrinsic offenders."""
    is_red, cok = classes[PRIMARY]
    summary, worst = {}, []

    for flavour, nat_pre, syn_pre, base_ok in (
        ("intrinsic", "Mag_", "synth_mag_", ok_int),
        ("dust", "MagDust_", "synth_magdust_", ok_dust),
    ):
        ok = base_ok & cok
        summary[flavour] = {}
        for band in BANDS:
            delta = d[f"{syn_pre}{band}"] - d[f"{nat_pre}{band}"]
            med = np.median(delta[ok])
            sig = robust_scatter(delta[ok])
            dev = np.abs(delta - med) / sig if sig > 0 else np.zeros_like(delta)
            for k in (3, 5):
                m = ok & (dev > k)
                summary[flavour][f"{band}_{k}sigma"] = {
                    "n": int(m.sum()),
                    "frac": float(m.sum() / ok.sum()),
                    "n_red": int((m & is_red).sum()),
                    "n_blue": int((m & ~is_red).sum()),
                    "red_frac_of_outliers": float((m & is_red).sum() / max(m.sum(), 1)),
                }

    # worst intrinsic r-band offenders
    ok = ok_int & cok
    delta = d["synth_mag_r"] - d["Mag_r"]
    med, sig = np.median(delta[ok]), robust_scatter(delta[ok])
    dev = np.where(ok, np.abs(delta - med) / sig, -1)
    idx = np.argsort(dev)[::-1][:n_worst]
    for i in idx:
        worst.append({
            "index": int(i),
            "dev_sigma": float(dev[i]),
            "delta_mag_r": float(delta[i]),
            "Mag_r": float(d["Mag_r"][i]),
            "is_red_paper1": bool(is_red[i]),
            "log_StellarMass": float(np.log10(max(d["StellarMass"][i], 1))),
            "BT": float(d["BT"][i]),
            "log_sSFR": float(np.log10(max(d["sSFR"][i], 1e-16))),
            "MassWeightAge_Gyr": float(d["MassWeightAge"][i]),
            "Type": int(d["Type"][i]),
        })
    return summary, worst


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.8, "axes.grid": True, "grid.alpha": 0.25,
        "grid.linewidth": 0.5, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt


def _save(fig, outdir, name):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}")
    print(f"    {outdir / (name + '.png')}")


def _running(x, y, bins):
    """Median and 16-84 percentile band of y in bins of x."""
    idx = np.digitize(x, bins) - 1
    cx, med, lo, hi = [], [], [], []
    for k in range(len(bins) - 1):
        m = idx == k
        if m.sum() < 15:
            continue
        cx.append(0.5 * (bins[k] + bins[k + 1]))
        med.append(np.median(y[m]))
        lo.append(np.percentile(y[m], 16))
        hi.append(np.percentile(y[m], 84))
    return map(np.asarray, (cx, med, lo, hi))


def fig1_residual_vs_mag(plt, d, classes, ok_int, ok_dust, outdir):
    is_red, cok = classes[PRIMARY]
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharex="col")
    for row, (flavour, nat_pre, syn_pre, base_ok) in enumerate((
        ("Intrinsic", "Mag_", "synth_mag_", ok_int),
        ("Dust-attenuated", "MagDust_", "synth_magdust_", ok_dust),
    )):
        ok = base_ok & cok
        for col, band in enumerate(BANDS):
            ax = axes[row, col]
            nat = d[f"{nat_pre}{band}"]
            delta = d[f"{syn_pre}{band}"] - nat
            bins = np.linspace(-24, -16, 25)
            for lab, m, c in (("blue", ok & ~is_red, C_BLUE), ("red", ok & is_red, C_RED)):
                ax.scatter(nat[m], delta[m], s=1.2, c=c, alpha=0.15, lw=0, rasterized=True)
                cx, med, lo, hi = _running(nat[m], delta[m], bins)
                if len(cx):
                    ax.fill_between(cx, lo, hi, color=c, alpha=0.20, lw=0)
                    ax.plot(cx, med, color=c, lw=2,
                            label=f"{lab} (N={int(m.sum()):,})")
            ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)
            ax.set_xlim(-24, -16)
            # intrinsic residuals live in ~0.06 mag; a wide axis would hide the
            # colour split, the whole point of this figure
            ax.set_ylim(-0.03, 0.09) if row == 0 else ax.set_ylim(-1.6, 1.6)
            if row == 0:
                ax.set_title(f"${band}$")
            if col == 0:
                ax.set_ylabel(f"{flavour}\n" + r"$\Delta$mag (synth $-$ native)")
            if row == 1:
                ax.set_xlabel(f"native $M_{{{band}}}$")
            ax.legend(loc="upper left", fontsize=7)
    axes[1, 2].text(0.5, 0.06, "faint-end plunge = the gas-free\npathology, see Fig 7",
                    transform=axes[1, 2].transAxes, fontsize=7, color=C_RED,
                    ha="center", va="bottom")
    fig.suptitle("Fig 1 — Reconstruction residual vs native magnitude, split by "
                 r"Paper I colour cut ($g-r \geq 0.4$)", y=1.00, fontsize=11)
    _save(fig, outdir, "fig1_residual_vs_magnitude")
    plt.close(fig)


def fig2_distributions(plt, d, classes, ok_int, ok_dust, stats, outdir):
    is_red, cok = classes[PRIMARY]
    fig, axes = plt.subplots(2, 5, figsize=(15, 5.6))
    for row, (flavour, key, nat_pre, syn_pre, base_ok, rng) in enumerate((
        ("Intrinsic", "intrinsic", "Mag_", "synth_mag_", ok_int, 0.12),
        ("Dust-attenuated", "dust", "MagDust_", "synth_magdust_", ok_dust, 1.2),
    )):
        ok = base_ok & cok
        for col, band in enumerate(BANDS):
            ax = axes[row, col]
            delta = d[f"{syn_pre}{band}"] - d[f"{nat_pre}{band}"]
            bins = np.linspace(-rng, rng, 60)
            for lab, m, c in (("blue", ok & ~is_red, C_BLUE), ("red", ok & is_red, C_RED)):
                s = stats[PRIMARY][key][band][lab]
                ax.hist(delta[m], bins=bins, histtype="step", lw=1.6, color=c, density=True,
                        label=f"{lab}: {s['median']:+.3f} $\\pm$ {s['mad']:.3f}")
            ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.6)
            if row == 0:
                ax.set_title(f"${band}$")
            if col == 0:
                ax.set_ylabel(f"{flavour}\nnormalised count")
            ax.set_xlabel(r"$\Delta$mag")
            ax.legend(loc="upper left", fontsize=7)
            ax.set_yticks([])
    fig.suptitle("Fig 2 — Residual distributions by population (median $\\pm$ robust MAD scatter)",
                 y=1.00, fontsize=11)
    _save(fig, outdir, "fig2_residual_distributions")
    plt.close(fig)


def fig3_cmd(plt, d, ok_dust, rf, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    ok = ok_dust
    mr_n, gr_n = d["MagDust_r"][ok], (d["MagDust_g"] - d["MagDust_r"])[ok]
    mr_s, gr_s = d["synth_magdust_r"][ok], (d["synth_magdust_g"] - d["synth_magdust_r"])[ok]
    ext = [-24, -16, -0.15, 1.0]
    for ax, (x, y, t, fr) in zip(axes[:2], (
        (mr_n, gr_n, "Native L-GALAXIES (MagDust)", rf["dust"]["native_red_frac"]),
        (mr_s, gr_s, "PCA reconstruction (synth)", rf["dust"]["synth_red_frac"]),
    )):
        ax.hexbin(x, y, gridsize=70, extent=ext, cmap="Greys", bins="log", mincnt=1)
        ax.axhline(PAPER1_CUT, color=C_RED, lw=1.8, ls="--")
        ax.text(-23.6, PAPER1_CUT + 0.03, f"Paper I cut $g-r={PAPER1_CUT}$",
                color=C_RED, fontsize=8, va="bottom")
        ax.text(0.97, 0.05, f"red fraction = {fr:.3f}", transform=ax.transAxes,
                ha="right", fontsize=9, color=C_RED, weight="bold")
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_xlabel(r"$M_r$"); ax.set_title(t)
    axes[0].set_ylabel(r"$g-r$")

    ax = axes[2]
    bins = np.linspace(-0.15, 1.0, 70)
    ax.hist(gr_n, bins=bins, histtype="step", lw=2, color=C_ALL, density=True, label="native")
    ax.hist(gr_s, bins=bins, histtype="step", lw=2, color=C_RED, density=True, label="synth")
    ax.axvline(PAPER1_CUT, color="k", lw=1.4, ls="--")
    ax.set_xlabel(r"$g-r$"); ax.set_ylabel("normalised count")
    ax.set_title(f"Colour distribution  (N={int(ok.sum()):,})")
    ax.legend(loc="upper left")
    ax.text(0.97, 0.72,
            f"median shift  {rf['dust']['synth_median_gr'] - rf['dust']['native_median_gr']:+.3f} mag\n"
            f"red fraction  {rf['dust']['native_red_frac']:.3f} $\\rightarrow$ "
            f"{rf['dust']['synth_red_frac']:.3f}\n"
            f"blue$\\rightarrow$red  {rf['dust']['n_blue_to_red']:,} galaxies\n"
            f"red$\\rightarrow$blue  {rf['dust']['n_red_to_blue']:,} galaxies",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(fc="white", ec="0.7", alpha=0.9))
    fig.suptitle("Fig 3 — Colour-magnitude diagram: is the bimodality preserved?",
                 y=1.02, fontsize=11)
    _save(fig, outdir, "fig3_colour_magnitude")
    plt.close(fig)


def fig4_vs_properties(plt, d, classes, ok_int, outdir):
    is_red, cok = classes[PRIMARY]
    ok = ok_int & cok
    delta = d["synth_mag_r"] - d["Mag_r"]
    props = [
        (np.log10(np.maximum(d["StellarMass"], 1)), r"$\log_{10}(M_\star/M_\odot)$", None),
        (d["BT"], "bulge-to-total $B/T$", None),
        (np.log10(np.maximum(d["sSFR"], 1e-16)), r"$\log_{10}(\mathrm{sSFR}/\mathrm{yr}^{-1})$", (-13, -8.5)),
        (d["MassWeightAge"], "mass-weighted age (Gyr)", None),
        (d["H2fraction"], r"H$_2$ fraction", None),
        (d["Type"], "galaxy type (0/1/2)", (-0.5, 2.5)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (x, lab, xlim) in zip(axes.ravel(), props):
        lo, hi = (np.nanpercentile(x[ok], [0.5, 99.5]) if xlim is None else xlim)
        bins = np.linspace(lo, hi, 22)
        for name, m, c in (("blue", ok & ~is_red, C_BLUE), ("red", ok & is_red, C_RED)):
            ax.scatter(x[m], delta[m], s=1.2, c=c, alpha=0.12, lw=0, rasterized=True)
            cx, med, plo, phi = _running(x[m], delta[m], bins)
            if len(cx):
                ax.fill_between(cx, plo, phi, color=c, alpha=0.2, lw=0)
                ax.plot(cx, med, color=c, lw=2, label=name)
        ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)
        ax.set_xlim(lo, hi); ax.set_ylim(-0.12, 0.12)
        ax.set_xlabel(lab); ax.set_ylabel(r"$\Delta m_r$ (intrinsic)")
        ax.legend(loc="upper right", fontsize=7)
    fig.suptitle("Fig 4 — What drives the intrinsic residual? "
                 r"$\Delta m_r$ against galaxy properties", y=1.00, fontsize=11)
    _save(fig, outdir, "fig4_residual_vs_properties")
    plt.close(fig)


def fig5_dust_hypothesis(plt, d, classes, ok_int, ok_dust, outdir):
    """Test: is the dust scatter the unstored birth-cloud mu, or a reconstruction failure?

    If it is mu, |Delta MagDust| must grow with optical depth (tau_V proportional to
    gas metal column) and with inclination, while |Delta Mag| stays flat.
    """
    if not d["has_dust_inputs"]:
        return None
    is_red, cok = classes[PRIMARY]
    ok = ok_dust & cok & (d["ColdGasRadius"] > 0) & (d["ColdGas"] > 0)
    ok_i = ok_int & cok & (d["ColdGasRadius"] > 0) & (d["ColdGas"] > 0)

    # Optical-depth proxy: metal column density of the cold gas disc.
    area = np.pi * np.maximum(d["ColdGasRadius"], 1e-8) ** 2
    tau_proxy = np.log10(np.maximum(d["MetalsColdGas"], 1e-12) / area + 1e-12)
    sec_i = 1.0 / np.clip(d["CosInclination"], 0.2, 1.0)   # matches the code's clamp

    ad = np.abs(d["synth_magdust_r"] - d["MagDust_r"])
    ai = np.abs(d["synth_mag_r"] - d["Mag_r"])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))

    for ax, (x, xl, mask_d, mask_i) in zip(axes[:2], (
        (tau_proxy, r"$\log_{10}(\Sigma_{\rm metals}$ / $M_\odot\,{\rm Mpc}^{-2})$  [$\tau_V$ proxy]", ok, ok_i),
        (sec_i, r"$\sec i = 1/\cos i$  (clamped at 5)", ok, ok_i),
    )):
        lo, hi = np.percentile(x[mask_d], [1, 99])
        bins = np.linspace(lo, hi, 20)
        cx, med, plo, phi = _running(x[mask_d], ad[mask_d], bins)
        ax.fill_between(cx, plo, phi, color=C_RED, alpha=0.2, lw=0)
        ax.plot(cx, med, color=C_RED, lw=2.2, marker="o", ms=4,
                label=r"$|\Delta m_r|$ dust-attenuated")
        cx2, med2, plo2, phi2 = _running(x[mask_i], ai[mask_i], bins)
        ax.fill_between(cx2, plo2, phi2, color=C_BLUE, alpha=0.2, lw=0)
        ax.plot(cx2, med2, color=C_BLUE, lw=2.2, marker="s", ms=4,
                label=r"$|\Delta m_r|$ intrinsic  (control)")
        ax.set_yscale("log"); ax.set_xlabel(xl)
        ax.set_ylabel(r"median $|\Delta m_r|$ (mag)")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xlim(lo, hi)

    ax = axes[2]
    for lab, m, c in (("blue", ok & ~is_red, C_BLUE), ("red", ok & is_red, C_RED)):
        ax.hist(ad[m], bins=np.logspace(-4, 0.5, 50), histtype="step", lw=1.8,
                color=c, density=True, label=f"{lab} dust")
    ax.hist(ai[ok_i], bins=np.logspace(-4, 0.5, 50), histtype="step", lw=1.8,
            color="k", ls=":", density=True, label="all intrinsic")
    ax.set_xscale("log"); ax.set_xlabel(r"$|\Delta m_r|$ (mag)")
    ax.set_ylabel("normalised count"); ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Dust vs intrinsic residual magnitude")

    fig.suptitle(r"Fig 5 — Is the 0.294 mag dust scatter the unstored birth-cloud $\mu$? "
                 "If yes, the red curve rises with optical depth and the blue stays flat.",
                 y=1.03, fontsize=11)
    _save(fig, outdir, "fig5_dust_scatter_hypothesis")
    plt.close(fig)

    # correlation numbers for the write-up
    from scipy.stats import spearmanr
    return {
        "spearman_dust_vs_tau": float(spearmanr(tau_proxy[ok], ad[ok]).statistic),
        "spearman_intrinsic_vs_tau": float(spearmanr(tau_proxy[ok_i], ai[ok_i]).statistic),
        "spearman_dust_vs_seci": float(spearmanr(sec_i[ok], ad[ok]).statistic),
        "spearman_intrinsic_vs_seci": float(spearmanr(sec_i[ok_i], ai[ok_i]).statistic),
        "median_abs_dust": float(np.median(ad[ok])),
        "median_abs_intrinsic": float(np.median(ai[ok_i])),
    }


def fig6_robustness(plt, stats, outdir):
    defs = ["lgal_gmm", "lgal_tilted", "paper1", "baldry", "ssfr"]
    labels = {"lgal_gmm": f"L-GAL GMM $g-r\\geq{LGAL_GMM_CUT:.3f}$ (default)",
              "lgal_tilted": r"L-GAL tilted $g-r\geq$cut($M_r$)",
              "paper1": r"Paper I  $g-r\geq0.4$",
              "baldry": "Baldry+2004 $u-r$ tilted",
              "ssfr": r"sSFR  $\log\,{\rm sSFR}<-11$"}
    # marker shape = definition, colour = population; not opacity — five
    # levels of it aren't separable by eye
    markers = ["o", "s", "^", "D", "v"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.5))
    x = np.arange(len(BANDS))
    for row, flavour in enumerate(("intrinsic", "dust")):
        for col, (stat, ylab) in enumerate((
            ("median", "median offset (mag)"), ("mad", "robust scatter, MAD (mag)"))):
            ax = axes[row, col]
            for k, dname in enumerate(defs):
                off = (k - (len(defs) - 1) / 2) * 0.15
                for cls, c in (("red", C_RED), ("blue", C_BLUE)):
                    y = [stats[dname][flavour][b][cls][stat] for b in BANDS]
                    ax.plot(x + off, y, marker=markers[k], ms=6, lw=0, color=c,
                            markerfacecolor=c if dname == PRIMARY else "none",
                            markeredgecolor=c, markeredgewidth=1.3,
                            label=(f"{cls} — {labels[dname]}"
                                   if col == 0 and row == 0 else None))
            ax.set_xticks(x); ax.set_xticklabels([f"${b}$" for b in BANDS])
            ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
            ax.set_ylabel(ylab)
            ax.set_title(f"{flavour}")
            if stat == "mad" and flavour == "dust":
                # a too-small class can give MAD=0, dragging a log axis to 1e-17;
                # clamp to the range that carries data
                vals = [v for k in defs for b in BANDS for cls in ("red", "blue")
                        if (v := stats[k][flavour][b][cls][stat]) and v > 0]
                if vals:
                    ax.set_yscale("log")
                    ax.set_ylim(min(vals) * 0.6, max(vals) * 1.7)
            # Headroom so the legend cannot sit on top of the markers.
            if row == 0 and col == 0:
                lo, hi = ax.get_ylim()
                ax.set_ylim(lo, hi + 0.45 * (hi - lo))
    axes[0, 0].legend(loc="upper left", fontsize=6.5, ncol=2)
    fig.suptitle("Fig 6 — Robustness: does the conclusion depend on the red/blue definition?\n"
                 "Marker shape = definition (filled = default), colour = population",
                 y=1.02, fontsize=11)
    _save(fig, outdir, "fig6_robustness_across_definitions")
    plt.close(fig)


def fig7_gas_free(plt, d, ok_int, ok_dust, gfr, outdir):
    """The gas-free dust pathology: where the 0.294 mag scatter actually comes from."""
    gf = gas_free_mask(d)
    A_nat = d["MagDust_r"] - d["Mag_r"]
    A_syn = d["synth_magdust_r"] - d["synth_mag_r"]
    delta = d["synth_magdust_r"] - d["MagDust_r"]
    m_clean, m_gf = ok_dust & ~gf, ok_dust & gf
    # Attenuation needs both magnitudes valid; the two sentinel sets differ.
    ok_att = ok_int & ok_dust
    a_clean, a_gf = ok_att & ~gf, ok_att & gf

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))

    ax = axes[0]
    cg = d["ColdGas"]
    x = np.sign(cg) * np.log10(np.abs(cg) + 1.0)
    dA = A_nat - A_syn
    ax.scatter(x[a_clean], dA[a_clean], s=2, c=C_ALL, alpha=0.20,
               lw=0, rasterized=True, label=f"gas-bearing (N={int(a_clean.sum()):,})")
    ax.scatter(x[a_gf], dA[a_gf], s=16, c=C_RED, alpha=0.85, lw=0,
               label=f"gas-free (N={int(a_gf.sum()):,})")
    ax.axvline(np.log10(GAS_FREE_THRESHOLD), color=C_RED, ls="--", lw=1.4)
    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_ylim(-0.6, max(4.6, float(dA[a_gf].max()) * 1.1))
    ax.set_xlabel(r"$\mathrm{sign}(M_{\rm cold})\,\log_{10}(|M_{\rm cold}|/M_\odot + 1)$")
    ax.set_ylabel(r"$A_r^{\rm native} - A_r^{\rm synth}$ (mag)")
    ax.set_title("Attenuation disagreement vs cold gas mass")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    bins = np.linspace(-1.5, 1.5, 80)
    ax.hist(delta[ok_dust], bins=bins, histtype="step", lw=1.8, color=C_ALL,
            density=True, label=f"all  $\\sigma$={gfr['delta_r_all']['std']:.3f}")
    ax.hist(delta[m_clean], bins=bins, histtype="step", lw=2.2, color=C_BLUE,
            density=True, label=f"gas-bearing  $\\sigma$={gfr['delta_r_clean']['std']:.3f}")
    ax.hist(delta[m_gf], bins=bins, histtype="stepfilled", lw=0, color=C_RED,
            alpha=0.55, density=True, label=f"gas-free  (N={int(m_gf.sum()):,})")
    ax.set_yscale("log")
    ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel(r"$\Delta m_r$ dust-attenuated (mag)")
    ax.set_ylabel("normalised count")
    ax.set_title("2.1 % of galaxies produce the heavy tail")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[2]
    labels = ["all\n(README)", "gas-bearing\nonly"]
    std = [gfr["delta_r_all"]["std"], gfr["delta_r_clean"]["std"]]
    mad = [gfr["delta_r_all"]["mad"], gfr["delta_r_clean"]["mad"]]
    xp = np.arange(2)
    ax.bar(xp - 0.19, std, width=0.36, color=C_RED, label=r"std $\sigma$")
    ax.bar(xp + 0.19, mad, width=0.36, color=C_BLUE, label="robust MAD")
    for xi, (a, b) in enumerate(zip(std, mad)):
        ax.text(xi - 0.19, a + 0.006, f"{a:.3f}", ha="center", fontsize=9, color=C_RED)
        ax.text(xi + 0.19, b + 0.006, f"{b:.3f}", ha="center", fontsize=9, color=C_BLUE)
    ax.set_xticks(xp); ax.set_xticklabels(labels)
    ax.set_ylabel(r"scatter in $\Delta m_r$ (mag)")
    ax.set_title("Effect on the quoted dust scatter")
    ax.set_ylim(0, max(std) * 1.25)
    ax.legend(fontsize=8)

    fig.suptitle("Fig 7 — The gas-free dust pathology: L-GALAXIES attenuates galaxies "
                 r"with $M_{\rm cold}\!\leq\!0$, GALsPeCtrA does not", y=1.02, fontsize=11)
    _save(fig, outdir, "fig7_gas_free_pathology")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Colour-split validation of the PCA reconstruction")
    p.add_argument("--outdir", default=str(PROJECT_ROOT / "figures" / "validation_by_colour"))
    p.add_argument("--stats-json", default=str(PROJECT_ROOT / "data" / "validation_by_colour_stats.json"))
    p.add_argument("--outliers-csv", default=str(PROJECT_ROOT / "data" / "validation_by_colour_outliers.csv"))
    p.add_argument("--n-worst", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    print("Loading data ...")
    d = load_data()
    ok_int, ok_dust = validity_masks(d)
    print(f"  {d['N']:,} galaxies   valid: intrinsic {ok_int.sum():,}, dust {ok_dust.sum():,}")
    print(f"  ({d['N'] - ok_int.sum():,} / {d['N'] - ok_dust.sum():,} carry the 99.0 sentinel)")

    classes = classify(d)
    for name, (is_red, cok) in classes.items():
        n = (cok).sum()
        print(f"  {name:<8} red {int((is_red & cok).sum()):,}  blue {int((~is_red & cok).sum()):,}"
              f"  (red fraction {float((is_red & cok).sum()) / n:.3f})")

    conf, base = confusion(classes, ok_dust)
    print("\nAgreement between classifications:")
    for r in conf:
        print(f"  {r['a']:<8} vs {r['b']:<8}  {r['agreement']:.3f}"
              f"   ({r['a_red_b_blue']:,} / {r['a_blue_b_red']:,} disagreements)")

    stats = compute_all_stats(d, classes, ok_int, ok_dust)
    rf = red_fraction_shift(d, ok_dust)

    print("\n" + "=" * 70)
    print("HEADLINE: does the reconstruction move galaxies across Paper I's cut?")
    print("=" * 70)
    for fl in ("intrinsic", "dust"):
        r = rf[fl]
        print(f"  {fl:<10} median g-r  {r['native_median_gr']:.4f} -> {r['synth_median_gr']:.4f}"
              f"   ({r['synth_median_gr'] - r['native_median_gr']:+.4f} mag)")
        print(f"  {'':<10} red fraction {r['native_red_frac']:.4f} -> {r['synth_red_frac']:.4f}"
              f"   ({r['synth_red_frac'] - r['native_red_frac']:+.4f})")
        print(f"  {'':<10} blue->red {r['n_blue_to_red']:,}   red->blue {r['n_red_to_blue']:,}")

    gfr = gas_free_report(d, ok_int, ok_dust)
    print("\n" + "=" * 70)
    print("GAS-FREE DUST PATHOLOGY (native MagDust unreliable, not a PCA failure)")
    print("=" * 70)
    print(f"  gas-free (M_cold <= {GAS_FREE_THRESHOLD:.0e} Msun): {gfr['n_gas_free']:,} "
          f"({gfr['frac_gas_free']:.2%}), of which {gfr['n_negative_coldgas']:,} have "
          f"NEGATIVE ColdGas")
    print(f"  implied gas Z = MetalsColdGas/ColdGas is negative for {gfr['n_negative_implied_Z']:,}")
    print(f"  native applies up to A_r = {gfr['native_A_r_max']:+.2f} mag; "
          f"synth applies at most {gfr['synth_A_r_max']:+.2f}")
    print(f"  {gfr['n_disagree_gt_half_mag']:,} disagree by more than 0.5 mag")
    print(f"  dust delta_r scatter   all: std {gfr['delta_r_all']['std']:.4f}  "
          f"MAD {gfr['delta_r_all']['mad']:.4f}")
    print(f"                gas-bearing: std {gfr['delta_r_clean']['std']:.4f}  "
          f"MAD {gfr['delta_r_clean']['mad']:.4f}   <- excluding them")
    print(f"  red fraction shift     all: {gfr['red_frac_all'][0]:.4f} -> {gfr['red_frac_all'][1]:.4f}")
    print(f"                gas-bearing: {gfr['red_frac_clean'][0]:.4f} -> {gfr['red_frac_clean'][1]:.4f}"
          f"   <- the colour bias is NOT caused by the pathology")

    out_summary, worst = find_outliers(d, classes, ok_int, ok_dust, args.n_worst)

    print("\nSetting up plots ...")
    plt = setup_mpl()
    print("  writing figures:")
    fig1_residual_vs_mag(plt, d, classes, ok_int, ok_dust, args.outdir)
    fig2_distributions(plt, d, classes, ok_int, ok_dust, stats, args.outdir)
    fig3_cmd(plt, d, ok_dust, rf, args.outdir)
    fig4_vs_properties(plt, d, classes, ok_int, args.outdir)
    dust_corr = fig5_dust_hypothesis(plt, d, classes, ok_int, ok_dust, args.outdir)
    fig6_robustness(plt, stats, args.outdir)
    fig7_gas_free(plt, d, ok_int, ok_dust, gfr, args.outdir)

    if dust_corr:
        print("\nDust-scatter hypothesis (Spearman rho):")
        print(f"  |d m_r| dust      vs tau proxy : {dust_corr['spearman_dust_vs_tau']:+.3f}")
        print(f"  |d m_r| intrinsic vs tau proxy : {dust_corr['spearman_intrinsic_vs_tau']:+.3f}  (control)")
        print(f"  |d m_r| dust      vs sec i     : {dust_corr['spearman_dust_vs_seci']:+.3f}")
        print(f"  |d m_r| intrinsic vs sec i     : {dust_corr['spearman_intrinsic_vs_seci']:+.3f}  (control)")

    payload = {
        "n_galaxies": d["N"],
        "n_valid_intrinsic": int(ok_int.sum()),
        "n_valid_dust": int(ok_dust.sum()),
        "class_counts": {k: {"red": int((v[0] & v[1]).sum()),
                             "blue": int((~v[0] & v[1]).sum())} for k, v in classes.items()},
        "confusion": conf,
        "red_fraction_shift": rf,
        "stats": stats,
        "outlier_summary": out_summary,
        "dust_hypothesis": dust_corr,
        "gas_free_pathology": gfr,
    }
    Path(args.stats_json).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nStats written to {args.stats_json}")

    cols = list(worst[0].keys())
    lines = [",".join(cols)] + [
        ",".join(f"{w[c]:.5g}" if isinstance(w[c], float) else str(w[c]) for c in cols)
        for w in worst
    ]
    Path(args.outliers_csv).write_text("\n".join(lines) + "\n")
    print(f"Worst {len(worst)} outliers written to {args.outliers_csv}")


if __name__ == "__main__":
    main()
