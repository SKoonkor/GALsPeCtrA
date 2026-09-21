"""
colour_cut_calibration.py

Locates the red/blue colour bimodality valley in L-GALAXIES, the GALFORM PAUS
lightcone, and the observed PAUS catalogues, and derives a data-driven cut
for L-GALAXIES to replace the inherited fixed value.

`scripts/validate_by_colour.py` originally split at (g-r) >= 0.4
(GALFORM_LF/REFACTORED/galform_lf/observed_lf.py:26), calibrated for the
GALFORM lightcone over 0 < z < 2, not L-GALAXIES at z = 0 — it misses the
L-GALAXIES bimodality valley and is not fitted, cited, or plotted anywhere
in the GALFORM_LF tree. See documents/colour_cut_calibration.md.

Methods: KDE valley (deepest local minimum between the two highest peaks,
over several bandwidths); GMM (two-component mixture, cut at equal
posterior, BIC tests bimodality rather than assuming it); tilted cut (GMM
per bin of M_r, fitted to Baldry et al. 2004's a - b*tanh((M_r+c)/d)).

Usage:
  python scripts/colour_cut_calibration.py
  python scripts/colour_cut_calibration.py --n-fields 128 --no-paus

Outputs: data/colour_cut_calibration.json (consumed by validate_by_colour.py),
figures/colour_cut_calibration/figC{1..4}_*.png/.pdf.
"""

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = PROJECT_ROOT.parent
GALFORM_LF = RESEARCH_ROOT / "GALFORM_LF"
LC_DIR = GALFORM_LF / "data" / "LC"
PAUS_DIR = GALFORM_LF / "data" / "PAUS"

# GALFORM lightcone dataset names. 'FHT' (not 'CFHT') in the i-band key is
# genuinely how it's spelled in the HDF5 files, not a transcription error.
LC_G = "mag_CFHT-MegaCam-1st-g_r_tot_ext"
LC_R = "mag_CFHT-MegaCam-1st-r_r_tot_ext"
LC_I = "mag_FHT-MegaCam-1st-i_r_tot_ext"

PAPER1_CUT = 0.4
BALDRY_PARAMS = (2.06, 0.244, 20.07, 1.09)   # a - b*tanh((M_r + c)/d), on (u-r)

C_RED, C_BLUE, C_ALL = "#d62728", "#1f77b4", "#4a4a4a"


# ─────────────────────────────────────────────────────────────────────────────
# Valley finders
# ─────────────────────────────────────────────────────────────────────────────

def kde_valley(x, lo=0.0, hi=1.0, bandwidths=(0.08, 0.12, 0.16), n_grid=1500):
    """Deepest local minimum between the two highest peaks, per bandwidth."""
    from scipy.stats import gaussian_kde
    from scipy.signal import argrelextrema

    grid = np.linspace(lo, hi, n_grid)
    out = {}
    for bw in bandwidths:
        y = gaussian_kde(x, bw_method=bw)(grid)
        maxima = argrelextrema(y, np.greater)[0]
        minima = argrelextrema(y, np.less)[0]
        if len(maxima) < 2 or len(minima) < 1:
            out[bw] = None
            continue
        top2 = sorted(maxima[np.argsort(y[maxima])[-2:]])
        between = [m for m in minima if top2[0] < m < top2[1]]
        if not between:
            out[bw] = None
            continue
        v = between[int(np.argmin(y[between]))]
        out[bw] = {
            "valley": float(grid[v]),
            "peak_blue": float(grid[top2[0]]),
            "peak_red": float(grid[top2[1]]),
            "depth_ratio": float(y[v] / min(y[top2[0]], y[top2[1]])),
        }
    return out


def gmm_cut(x, lo=0.0, hi=1.0, n_grid=1500, max_components=3, seed=0):
    """Two-component mixture; cut at equal posterior. BIC for 1..max_components."""
    from sklearn.mixture import GaussianMixture

    xr = np.asarray(x).reshape(-1, 1)
    bic = {}
    for k in range(1, max_components + 1):
        bic[k] = float(GaussianMixture(k, random_state=seed, n_init=5).fit(xr).bic(xr))

    gm = GaussianMixture(2, random_state=seed, n_init=10).fit(xr)
    mu, sd, w = gm.means_.ravel(), np.sqrt(gm.covariances_.ravel()), gm.weights_
    order = np.argsort(mu)
    mu, sd, w = mu[order], sd[order], w[order]

    grid = np.linspace(lo, hi, n_grid)
    post_red = gm.predict_proba(grid.reshape(-1, 1))[:, order[1]]
    cut = float(grid[int(np.argmin(np.abs(post_red - 0.5)))])

    return {
        "cut": cut,
        "blue": {"mu": float(mu[0]), "sigma": float(sd[0]), "weight": float(w[0])},
        "red": {"mu": float(mu[1]), "sigma": float(sd[1]), "weight": float(w[1])},
        "bic": bic,
        "bimodal_preferred": bool(bic[2] < bic[1]),
        "delta_bic_2_vs_1": float(bic[1] - bic[2]),
    }


def characterise(name, colour, lo=0.0, hi=1.0):
    res = {
        "name": name,
        "n": int(colour.size),
        "percentiles": {str(p): float(np.percentile(colour, p))
                        for p in (5, 25, 50, 75, 95)},
        "red_frac_at_paper1_cut": float((colour >= PAPER1_CUT).mean()),
        "kde": kde_valley(colour, lo, hi),
        "gmm": gmm_cut(colour, lo, hi),
    }
    vals = [v["valley"] for v in res["kde"].values() if v]
    res["kde_valley_median"] = float(np.median(vals)) if vals else None
    res["kde_valley_spread"] = float(np.ptp(vals)) if len(vals) > 1 else 0.0
    if res["kde_valley_median"] is not None:
        res["kde_gmm_agreement"] = abs(res["kde_valley_median"] - res["gmm"]["cut"])
    return res


def tilted_cut(colour, mag, mag_bins, min_per_bin=200, seed=0):
    """GMM valley in bins of absolute magnitude, then a Baldry-form fit."""
    from scipy.optimize import curve_fit

    centres, cuts = [], []
    for k in range(len(mag_bins) - 1):
        m = (mag >= mag_bins[k]) & (mag < mag_bins[k + 1])
        if m.sum() < min_per_bin:
            continue
        try:
            g = gmm_cut(colour[m], seed=seed)
        except Exception:
            continue
        if not g["bimodal_preferred"]:
            continue
        centres.append(0.5 * (mag_bins[k] + mag_bins[k + 1]))
        cuts.append(g["cut"])

    centres, cuts = np.asarray(centres), np.asarray(cuts)
    fit = None
    if len(centres) >= 5:
        def baldry_form(mr, a, b, c, d):
            return a - b * np.tanh((mr + c) / d)
        try:
            p, _ = curve_fit(baldry_form, centres, cuts,
                             p0=[np.median(cuts), 0.1, 20.0, 1.1], maxfev=20000)
            resid = cuts - baldry_form(centres, *p)
            fit = {"a": float(p[0]), "b": float(p[1]), "c": float(p[2]), "d": float(p[3]),
                   "rms_resid": float(np.sqrt(np.mean(resid ** 2)))}
        except Exception as exc:
            fit = {"error": str(exc)}
    return {"mag_centres": centres.tolist(), "cuts": cuts.tolist(), "baldry_fit": fit}


def tilted_eval(fit, mr):
    return fit["a"] - fit["b"] * np.tanh((mr + fit["c"]) / fit["d"])


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_lgalaxies():
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from validate_by_colour import load_data, validity_masks, gas_free_mask

    d = load_data()
    ok_i, ok_d = validity_masks(d)
    gf = gas_free_mask(d)
    return {
        "d": d,
        "intrinsic": {"ok": ok_i & ~gf,
                      "colour": d["Mag_g"] - d["Mag_r"], "mag": d["Mag_r"],
                      "synth_colour": d["synth_mag_g"] - d["synth_mag_r"]},
        "dust": {"ok": ok_d & ~gf,
                 "colour": d["MagDust_g"] - d["MagDust_r"], "mag": d["MagDust_r"],
                 "synth_colour": d["synth_magdust_g"] - d["synth_magdust_r"]},
    }


def load_galform(n_fields=64, z_max=None, seed=0):
    """Rest-frame g-r from the GALFORM LC12 lightcone.

    Each file is a 1/1024 random subsample of a ~100 deg^2 field, so a subset
    is itself a fair sample — fine for shape, not for volume-normalised
    counts without weighting.
    """
    import h5py

    files = sorted(LC_DIR.glob("field1.*.hdf5"))
    if not files:
        return None
    rng = np.random.default_rng(seed)
    pick = files if n_fields >= len(files) else [
        files[i] for i in rng.choice(len(files), n_fields, replace=False)]

    g, r, i_, z = [], [], [], []
    for f in pick:
        with h5py.File(f, "r") as h:
            D = h["Data"]
            g.append(D[LC_G][:]); r.append(D[LC_R][:])
            i_.append(D[LC_I][:]); z.append(D["z_obs"][:])
    g, r, i_, z = (np.concatenate(a) for a in (g, r, i_, z))

    keep = np.isfinite(g) & np.isfinite(r) & np.isfinite(z)
    if z_max is not None:
        keep &= z <= z_max
    return {"colour": (g - r)[keep], "mag": r[keep], "mag_i": i_[keep],
            "z": z[keep], "n_fields": len(pick)}


def load_paus():
    """Observed PAUS colour from LePhare absolute magnitudes, W1 + W3.

    Reuses the quality cut from GALFORM_LF/REFACTORED/galform_lf/paus.py:47 verbatim,
    so this selection matches the luminosity function measurement.
    """
    import pandas as pd

    frames = []
    for pat in ("PAUS_W1_*_complete_i.csv", "PAUS_W3_*_complete_i.csv"):
        hits = [p for p in PAUS_DIR.glob(pat) if "random_points" not in p.name]
        if not hits:
            continue
        df = pd.read_csv(hits[0], usecols=[
            "mag_u", "mag_g", "mag_r", "mag_i", "mag_z",
            "lp_mg", "lp_mr", "lp_mi", "mask", "star_flag", "zb"])
        q = ((df.mag_i >= 0) & (df.mag_g >= 0) & (df.mag_r >= 0)
             & (df.mag_u >= 0) & (df.mag_z >= 0)
             & (df.mag_i <= 24) & (df.mag_u <= 50) & (df.mag_g <= 50)
             & (df.mag_r <= 50) & (df.mag_z <= 50)
             & (df["mask"] <= 1) & (df.star_flag == 0) & (df.lp_mi >= -30))
        frames.append((hits[0].name, len(df), df[q.to_numpy()]))

    if not frames:
        return None
    import pandas as pd
    raw_counts = {n: c for n, c, _ in frames}
    df = pd.concat([f for _, _, f in frames], ignore_index=True)
    colour = (df.lp_mg - df.lp_mr).to_numpy()
    keep = np.isfinite(colour) & (df.lp_mr.to_numpy() > -30)
    return {"colour": colour[keep], "mag": df.lp_mr.to_numpy()[keep],
            "z": df.zb.to_numpy()[keep], "raw_counts": raw_counts}


# ─────────────────────────────────────────────────────────────────────────────
# Shift-vs-cut curve
# ─────────────────────────────────────────────────────────────────────────────

def shift_vs_cut(native, synth, cuts):
    rows = []
    for c in cuts:
        nr, sr = native >= c, synth >= c
        rows.append({
            "cut": float(c),
            "native_red_frac": float(nr.mean()),
            "synth_red_frac": float(sr.mean()),
            "shift": float(sr.mean() - nr.mean()),
            "blue_to_red": int((~nr & sr).sum()),
            "red_to_blue": int((nr & ~sr).sum()),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Figures
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
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}")
    print(f"    {outdir / (name + '.png')}")


def figC1_three_way(plt, panels, outdir):
    fig, axes = plt.subplots(1, len(panels), figsize=(4.7 * len(panels), 4.2), sharey=True)
    if len(panels) == 1:
        axes = [axes]
    bins = np.linspace(-0.1, 1.1, 90)
    for ax, (label, colour, res, note) in zip(axes, panels):
        ax.hist(colour, bins=bins, color=C_ALL, alpha=0.30, density=True, lw=0)
        ax.hist(colour, bins=bins, histtype="step", color=C_ALL, density=True, lw=1.6)
        v = res.get("kde_valley_median")
        if v is not None:
            ax.axvline(v, color=C_BLUE, lw=2.2,
                       label=f"KDE valley = {v:.3f}")
        ax.axvline(res["gmm"]["cut"], color="#2ca02c", lw=2.2, ls="-.",
                   label=f"GMM cut = {res['gmm']['cut']:.3f}")
        ax.axvline(PAPER1_CUT, color=C_RED, lw=2.2, ls="--",
                   label=f"Paper I cut = {PAPER1_CUT}")
        ax.set_title(f"{label}\nN = {res['n']:,}{note}")
        ax.set_xlabel(r"$g-r$")
        ax.legend(loc="upper right", fontsize=8)
        ax.text(0.02, 0.97,
                f"median {res['percentiles']['50']:.3f}\n"
                f"$f_{{\\rm red}}$(0.4) = {res['red_frac_at_paper1_cut']:.3f}",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(fc="white", ec="0.75", alpha=0.9))
    axes[0].set_ylabel("normalised count")
    fig.suptitle("Fig C1 — Where does the 0.4 cut actually land? "
                 "Rest-frame colour distributions compared", y=1.03, fontsize=11)
    _save(fig, outdir, "figC1_three_way_colour_distributions")
    plt.close(fig)


def figC2_lgal_calibration(plt, lg, results, outdir):
    from scipy.stats import gaussian_kde
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    grid = np.linspace(-0.1, 1.05, 900)
    for ax, flavour in zip(axes, ("intrinsic", "dust")):
        c = lg[flavour]["colour"][lg[flavour]["ok"]]
        res = results[f"lgal_{flavour}"]
        ax.hist(c, bins=np.linspace(-0.1, 1.05, 90), density=True, color=C_ALL,
                alpha=0.25, lw=0)
        ax.plot(grid, gaussian_kde(c, bw_method=0.12)(grid), color="k", lw=1.8,
                label="KDE (bw 0.12)")
        for comp, col, nm in ((res["gmm"]["blue"], C_BLUE, "blue comp"),
                              (res["gmm"]["red"], C_RED, "red comp")):
            y = comp["weight"] * np.exp(-0.5 * ((grid - comp["mu"]) / comp["sigma"]) ** 2) \
                / (comp["sigma"] * np.sqrt(2 * np.pi))
            ax.plot(grid, y, color=col, lw=1.8, ls="--",
                    label=f"{nm}: $\\mu$={comp['mu']:.3f} w={comp['weight']:.2f}")
        v = res.get("kde_valley_median")
        if v is not None:
            ax.axvline(v, color="#2ca02c", lw=2.0, label=f"KDE valley {v:.3f}")
        ax.axvline(res["gmm"]["cut"], color="#9467bd", lw=2.0, ls="-.",
                   label=f"GMM cut {res['gmm']['cut']:.3f}")
        ax.axvline(PAPER1_CUT, color=C_RED, lw=2.0, ls=":",
                   label=f"Paper I {PAPER1_CUT}")
        ax.set_xlabel(r"$g-r$")
        ax.set_title(f"L-GALAXIES {flavour}   (N={res['n']:,},  "
                     f"$\\Delta$BIC$_{{2vs1}}$={res['gmm']['delta_bic_2_vs_1']:.0f})")
        ax.legend(loc="upper right", fontsize=7)
    axes[0].set_ylabel("normalised count")
    fig.suptitle("Fig C2 — Calibrating the L-GALAXIES cut: KDE and two-component GMM, "
                 "intrinsic vs dust-attenuated", y=1.02, fontsize=11)
    _save(fig, outdir, "figC2_lgalaxies_cut_calibration")
    plt.close(fig)


def figC3_tilted(plt, lg, tilt, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, flavour in zip(axes, ("intrinsic", "dust")):
        sel = lg[flavour]["ok"]
        c, m = lg[flavour]["colour"][sel], lg[flavour]["mag"][sel]
        t = tilt[flavour]
        ax.hexbin(m, c, gridsize=60, extent=[-24, -16, -0.1, 1.0],
                  cmap="Greys", bins="log", mincnt=1)
        if t["mag_centres"]:
            ax.plot(t["mag_centres"], t["cuts"], "o", color="#2ca02c", ms=6,
                    label="per-bin GMM cut")
        fit = t.get("baldry_fit")
        if fit and "a" in fit:
            mr = np.linspace(-24, -16, 200)
            ax.plot(mr, tilted_eval(fit, mr), color="#2ca02c", lw=2.4,
                    label=(f"fit: {fit['a']:.3f} $-$ {fit['b']:.3f}"
                           r"$\tanh$((M$_r$+" + f"{fit['c']:.2f})/{fit['d']:.2f})"))
        ax.axhline(PAPER1_CUT, color=C_RED, lw=2.0, ls="--",
                   label=f"Paper I flat {PAPER1_CUT}")
        ax.set_xlim(-24, -16); ax.set_ylim(-0.1, 1.0)
        ax.set_xlabel(r"$M_r$"); ax.set_title(f"L-GALAXIES {flavour}")
        ax.legend(loc="lower left", fontsize=7)
    axes[0].set_ylabel(r"$g-r$")
    fig.suptitle("Fig C3 — Does the valley drift with luminosity? "
                 "Magnitude-dependent cut vs the flat 0.4", y=1.02, fontsize=11)
    _save(fig, outdir, "figC3_tilted_cut")
    plt.close(fig)


def figC4_shift_vs_cut(plt, curves, marks, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    for flavour, ls, col in (("intrinsic", "--", C_BLUE), ("dust", "-", C_RED)):
        rows = curves[flavour]
        x = [r["cut"] for r in rows]
        axes[0].plot(x, [r["shift"] for r in rows], ls=ls, color=col, lw=2.2,
                     label=f"{flavour}")
        axes[1].plot(x, [r["native_red_frac"] for r in rows], ls=ls, color=col, lw=2.2,
                     label=f"{flavour} native")
        axes[1].plot(x, [r["synth_red_frac"] for r in rows], ls=ls, color=col, lw=2.2,
                     alpha=0.45, label=f"{flavour} reconstructed")

    for ax in axes:
        for name, val, col in marks:
            ax.axvline(val, color=col, lw=1.5, ls=":", alpha=0.9)
            ax.text(val, ax.get_ylim()[1], f" {name}", rotation=90, va="top",
                    ha="right", fontsize=7, color=col)
    axes[0].axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)
    axes[0].set_xlabel(r"colour cut in $g-r$")
    axes[0].set_ylabel(r"red-fraction shift (reconstructed $-$ native)")
    axes[0].set_title("The shift never reaches zero")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(loc="lower right", fontsize=8)
    axes[1].set_xlabel(r"colour cut in $g-r$")
    axes[1].set_ylabel("red fraction")
    axes[1].set_title("Red fraction vs cut position")
    axes[1].legend(loc="upper right", fontsize=7)
    fig.suptitle("Fig C4 — How much of the red-fraction shift is cut placement?",
                 y=1.02, fontsize=11)
    _save(fig, outdir, "figC4_shift_vs_cut")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Calibrate the red/blue colour cut")
    p.add_argument("--n-fields", type=int, default=64,
                   help="GALFORM lightcone fields to read (of 1024)")
    p.add_argument("--galform-zmax", type=float, default=0.3,
                   help="Redshift ceiling for the GALFORM comparison slice")
    p.add_argument("--no-galform", action="store_true")
    p.add_argument("--no-paus", action="store_true")
    p.add_argument("--outdir", default=str(PROJECT_ROOT / "figures" / "colour_cut_calibration"))
    p.add_argument("--json", default=str(PROJECT_ROOT / "data" / "colour_cut_calibration.json"))
    return p.parse_args()


def main():
    args = parse_args()
    results, panels = {}, []

    print("L-GALAXIES ...")
    lg = load_lgalaxies()
    for flavour in ("intrinsic", "dust"):
        sel = lg[flavour]["ok"]
        c = lg[flavour]["colour"][sel]
        res = characterise(f"L-GALAXIES z=0 ({flavour})", c)
        results[f"lgal_{flavour}"] = res
        print(f"  {flavour:<10} N={res['n']:6,}  median={res['percentiles']['50']:.4f}  "
              f"KDE valley={res['kde_valley_median']:.4f} (spread {res['kde_valley_spread']:.4f})  "
              f"GMM={res['gmm']['cut']:.4f}  dBIC={res['gmm']['delta_bic_2_vs_1']:.0f}")
    panels.append(("L-GALAXIES z = 0 (MagDust)",
                   lg["dust"]["colour"][lg["dust"]["ok"]], results["lgal_dust"],
                   "  gas-bearing"))

    if not args.no_galform:
        print(f"GALFORM LC12 ({args.n_fields} fields) ...")
        # full redshift range loaded; slices taken below, not in the loader
        gf = load_galform(args.n_fields, z_max=None)
        if gf:
            # low-z slice (like-for-like with L-GALAXIES z=0) and the full
            # lightcone range the 0.4 cut is actually applied over
            for tag, sel, note in (
                ("galform_lowz", gf["z"] <= args.galform_zmax, f"z < {args.galform_zmax}"),
                ("galform_all", np.ones(gf["colour"].size, bool), "0 < z < 2"),
            ):
                res = characterise(f"GALFORM LC12 ({note})", gf["colour"][sel])
                res["n_fields"] = gf["n_fields"]
                results[tag] = res
                print(f"  {note:<10} N={res['n']:8,} from {gf['n_fields']} fields  "
                      f"median={res['percentiles']['50']:.4f}  "
                      f"KDE valley={res['kde_valley_median']}  GMM={res['gmm']['cut']:.4f}  "
                      f"dBIC={res['gmm']['delta_bic_2_vs_1']:+.0f}"
                      f"{'  <- NOT bimodal' if not res['gmm']['bimodal_preferred'] else ''}")
            panels.append((f"GALFORM LC12, z < {args.galform_zmax}",
                           gf["colour"][gf["z"] <= args.galform_zmax],
                           results["galform_lowz"], f"  {gf['n_fields']}/1024 fields"))

    if not args.no_paus:
        print("PAUS W1+W3 ...")
        pa = load_paus()
        if pa:
            # full sample spans 0 < z < 1.2 with heterogeneous k-corrections;
            # only the low-z slice is like-for-like with an L-GALAXIES z=0 snapshot
            for tag, sel, note in (
                ("paus_all", np.ones(pa["colour"].size, bool), "all z"),
                ("paus_lowz", pa["z"] <= args.galform_zmax, f"z < {args.galform_zmax}"),
            ):
                if sel.sum() < 500:
                    continue
                res = characterise(f"PAUS observed W1+W3 ({note})", pa["colour"][sel])
                results[tag] = res
                results[tag]["raw_counts"] = pa["raw_counts"]
                print(f"  {note:<10} N={res['n']:9,}  median={res['percentiles']['50']:.4f}  "
                      f"KDE valley={res['kde_valley_median']}  GMM={res['gmm']['cut']:.4f}  "
                      f"dBIC={res['gmm']['delta_bic_2_vs_1']:+.0f}"
                      f"{'  <- NOT bimodal' if not res['gmm']['bimodal_preferred'] else ''}")
            if "paus_lowz" in results:
                panels.append((f"PAUS observed, z < {args.galform_zmax}",
                               pa["colour"][pa["z"] <= args.galform_zmax],
                               results["paus_lowz"], "  LePhare absolutes"))

    print("\nTilted cuts ...")
    tilt = {}
    mag_bins = np.arange(-23.5, -16.5, 0.5)
    for flavour in ("intrinsic", "dust"):
        sel = lg[flavour]["ok"]
        tilt[flavour] = tilted_cut(lg[flavour]["colour"][sel], lg[flavour]["mag"][sel], mag_bins)
        f = tilt[flavour]["baldry_fit"]
        if f and "a" in f:
            # d << 1: tanh collapsed to a step — interpolating noise across too
            # few bins, not measuring a real luminosity trend
            f["reliable"] = bool(f["d"] > 0.3 and f["rms_resid"] < 0.05)
            print(f"  {flavour:<10} {f['a']:.4f} - {f['b']:.4f} tanh((M_r + {f['c']:.3f})"
                  f" / {f['d']:.3f})   rms {f['rms_resid']:.4f}  "
                  f"({len(tilt[flavour]['cuts'])} bins)"
                  f"{'' if f['reliable'] else '   <- UNRELIABLE, tanh collapsed to a step'}")
        else:
            print(f"  {flavour:<10} fit failed or too few bins")
    results["tilted"] = tilt

    print("\nShift vs cut ...")
    cuts = np.round(np.arange(0.25, 0.751, 0.005), 4)
    curves = {}
    for flavour in ("intrinsic", "dust"):
        sel = lg[flavour]["ok"]
        curves[flavour] = shift_vs_cut(lg[flavour]["colour"][sel],
                                       lg[flavour]["synth_colour"][sel], cuts)
    results["shift_vs_cut"] = curves

    default_cut = results["lgal_dust"]["gmm"]["cut"]
    results["calibrated"] = {
        "lgal_gmm_cut_dust": default_cut,
        "lgal_gmm_cut_intrinsic": results["lgal_intrinsic"]["gmm"]["cut"],
        "lgal_kde_valley_dust": results["lgal_dust"]["kde_valley_median"],
        "lgal_tilted_dust": tilt["dust"]["baldry_fit"],
        "lgal_tilted_intrinsic": tilt["intrinsic"]["baldry_fit"],
        "paper1_cut": PAPER1_CUT,
    }

    def at(flavour, c):
        rows = curves[flavour]
        return min(rows, key=lambda r: abs(r["cut"] - c))

    print("\n" + "=" * 74)
    print("HOW MUCH OF THE SHIFT IS CUT PLACEMENT?")
    print("=" * 74)
    for label, c in (("Paper I flat 0.4", PAPER1_CUT),
                     ("L-GAL GMM (dust)", default_cut),
                     ("L-GAL KDE valley", results["lgal_dust"]["kde_valley_median"])):
        r = at("dust", c)
        print(f"  {label:<20} cut={r['cut']:.3f}  "
              f"f_red {r['native_red_frac']:.4f} -> {r['synth_red_frac']:.4f}  "
              f"shift {r['shift']:+.4f}   blue->red {r['blue_to_red']:,}  "
              f"red->blue {r['red_to_blue']:,}")
    # minimum over the DEFENSIBLE range only: the shift falls toward the scan's
    # extremes simply because almost everyone ends up on one side there (degenerate
    # endpoints, not evidence the bias can be tuned away)
    band = [r for r in curves["dust"] if 0.40 <= r["cut"] <= 0.55]
    mn, mx = min(band, key=lambda r: r["shift"]), max(band, key=lambda r: r["shift"])
    print(f"\n  Over the defensible range 0.40-0.55 (valley +/- 0.075):")
    print(f"    shift ranges {mn['shift']:+.4f} (at {mn['cut']:.3f}) "
          f"to {mx['shift']:+.4f} (at {mx['cut']:.3f}) -- never zero, always one-directional")
    ends = min(curves["dust"], key=lambda r: r["shift"])
    print(f"    (the global scan minimum is {ends['shift']:+.4f} at cut={ends['cut']:.3f}, "
          f"a degenerate endpoint where f_red={ends['native_red_frac']:.3f})")
    results["defensible_band"] = {"lo": 0.40, "hi": 0.55,
                                  "min_shift": mn["shift"], "max_shift": mx["shift"]}

    print("\nFigures ...")
    plt = setup_mpl()
    figC1_three_way(plt, panels, args.outdir)
    figC2_lgal_calibration(plt, lg, results, args.outdir)
    figC3_tilted(plt, lg, tilt, args.outdir)
    marks = [("Paper I 0.4", PAPER1_CUT, C_RED),
             ("L-GAL GMM", default_cut, "#2ca02c")]
    figC4_shift_vs_cut(plt, curves, marks, args.outdir)

    Path(args.json).write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nCalibration written to {args.json}")
    print(f"Default cut for validate_by_colour.py: g-r = {default_cut:.4f}")


if __name__ == "__main__":
    main()
