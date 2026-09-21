"""
Figures for the weighted-PCA experiment.

Five categorical hues (Okabe-Ito), validated with the dataviz palette
checker: CVD separation PASS (worst pair ΔE 9.6 deutan), contrast WARN on
two of five — so every series is direct-labelled at its right end with its
own marker shape; identity is never carried by colour alone. Baselines are
neutral grey with distinct dash patterns, not a sixth categorical hue.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["SCHEME_COLOURS", "SCHEME_MARKERS", "make_all_figures"]

# Validated categorical palette, assigned in fixed order and never cycled.
_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#56B4E9", "#CC79A7"]
_MARKERS = ["o", "s", "^", "D", "v"]

SCHEME_COLOURS: dict[str, str] = {}
SCHEME_MARKERS: dict[str, str] = {}

_GREY = "#4d4d4d"
_GRID = "#d9d9d9"
_INK = "#1a1a1a"
_MUTED = "#666666"


def _assign_styles(labels):
    """Fixed-order hue assignment. A ninth series would fold into 'other', not cycle."""
    SCHEME_COLOURS.clear()
    SCHEME_MARKERS.clear()
    for i, lab in enumerate(labels):
        if i >= len(_PALETTE):
            SCHEME_COLOURS[lab] = _GREY
            SCHEME_MARKERS[lab] = "x"
        else:
            SCHEME_COLOURS[lab] = _PALETTE[i]
            SCHEME_MARKERS[lab] = _MARKERS[i]


def _style_axes(ax, xlabel=None, ylabel=None, title=None):
    ax.grid(True, color=_GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_MUTED, labelsize=8.5, length=3)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9.5, color=_INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5, color=_INK)
    if title:
        ax.set_title(title, fontsize=10.5, color=_INK, loc="left", pad=8)


def _by_scheme(results):
    """Group runs by weighting label, sorted by component count."""
    out = {}
    for r in results["runs"]:
        lab = r.get("weighting", {}).get("label", "?")
        out.setdefault(lab, []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r["n_components"])
    return out


def _metric(run, path, default=np.nan):
    node = run
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return default if node is None else node


def _direct_label(ax, x, y, text, colour, dx=1.03, fontsize=8.0):
    """Label at the right end of a curve, so identity never depends on colour."""
    if not (np.isfinite(x) and np.isfinite(y)):
        return
    ax.annotate(text, xy=(x, y), xytext=(x * dx, y), color=colour,
                fontsize=fontsize, va="center", ha="left",
                annotation_clip=False, fontweight="medium")


class _LabelStack:
    """Collects labels and places them with a guaranteed vertical gap.

    Several curves lie almost on top of one another (itself a finding), so
    labels are pushed apart in display space and joined to their anchor by a
    hairline leader, keeping the association unambiguous without moving data.
    """

    def __init__(self, ax, min_gap_px=11.0, fontsize=8.0):
        self.ax = ax
        self.min_gap = min_gap_px
        self.fontsize = fontsize
        self.items = []

    def add(self, x, y, text, colour, side="right"):
        if np.isfinite(x) and np.isfinite(y):
            self.items.append([float(x), float(y), text, colour, side])

    def draw(self):
        if not self.items:
            return
        ax = self.ax
        ax.figure.canvas.draw()
        trans = ax.transData
        inv = trans.inverted()

        for side in ("right", "below"):
            group = [it for it in self.items if it[4] == side]
            if not group:
                continue
            pts = trans.transform([(it[0], it[1]) for it in group])
            order = np.argsort(pts[:, 1])
            ys = pts[order, 1].astype(float)
            for i in range(1, len(ys)):                      # push upward
                ys[i] = max(ys[i], ys[i - 1] + self.min_gap)
            # Recentre so the stack does not drift off the top of the axes.
            drift = ys.mean() - pts[order, 1].mean()
            ys -= drift

            for k, idx in enumerate(order):
                x_px = pts[idx, 0]
                dx_px = 9.0 if side == "right" else 0.0
                x_lab, y_lab = inv.transform((x_px + dx_px, ys[k]))
                it = group[idx]
                ax.annotate(
                    it[2], xy=(it[0], it[1]), xytext=(x_lab, y_lab),
                    color=it[3], fontsize=self.fontsize, fontweight="medium",
                    va="center", ha="left" if side == "right" else "center",
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=it[3], linewidth=0.6,
                                    alpha=0.55, shrinkA=0, shrinkB=2),
                )


# ─────────────────────────────────────────────────────────────────────────────

def fig_error_vs_bytes(results, path):
    """The headline: what a byte buys, in millimagnitudes."""
    import matplotlib.pyplot as plt

    groups = _by_scheme(results)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))

    panels = [
        (axes[0], ("galaxies", "by_set", "paus_nb", "mad_mmag"),
         "PAUS 40 narrow bands", "median |Δm| per galaxy  (mmag)"),
        (axes[1], ("galaxies", "by_set", "lgal_sdss", "mad_mmag"),
         "SDSS ugriz (the bands L-GALAXIES itself writes)", None),
    ]
    for ax, key, title, ylabel in panels:
        stack = _LabelStack(ax)
        for lab, runs in groups.items():
            x = np.array([r["bytes_per_galaxy"] for r in runs], dtype=float)
            y = np.array([_metric(r, key) for r in runs], dtype=float)
            ok = np.isfinite(y) & (y > 0)
            if ok.sum() == 0:
                continue
            ax.plot(x[ok], y[ok], "-", color=SCHEME_COLOURS[lab], linewidth=2.0,
                    marker=SCHEME_MARKERS[lab], markersize=4.5,
                    markeredgecolor="white", markeredgewidth=0.7, zorder=3)
            stack.add(x[ok][-1], y[ok][-1], lab, SCHEME_COLOURS[lab])

        # only baselines that carry an argument get named on the plot (rest in CSV) —
        # a chart labelled with everything is labelled with nothing
        keep = ("previous", "13 bins", "broadband", "2 bins")
        for b in results["baselines"]:
            y = _metric(b, key)
            if not np.isfinite(y) or y <= 0:
                continue
            ax.scatter([b["bytes_per_galaxy"]], [y], s=52, marker="*",
                       color=_GREY, zorder=4, edgecolor="white", linewidth=0.6)
            short = b["label"].replace("baseline: ", "")
            if any(k in short for k in keep):
                stack.add(b["bytes_per_galaxy"], y, short, _GREY, side="below")

        ax.set_xscale("log")
        ax.set_yscale("log")
        _style_axes(ax, "bytes per galaxy  (float32, mass term included)", ylabel, title)
        ax.set_xlim(ax.get_xlim()[0] * 0.75, ax.get_xlim()[1] * 3.0)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo * 0.5, hi * 2.0)
        stack.draw()

    fig.suptitle("Reconstruction error against storage cost",
                 fontsize=12, color=_INK, x=0.06, ha="left", y=0.985)
    fig.text(0.06, 0.925,
             "Stars are baselines. Component count is read off this curve, never from a "
             "cumulative-variance threshold.",
             fontsize=8.5, color=_MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def fig_error_by_wavelength(results, path, pivots, n_target=50):
    """Where in wavelength each weighting scheme spends its accuracy."""
    import matplotlib.pyplot as plt

    groups = _by_scheme(results)
    fig, axes = plt.subplots(2, 1, figsize=(10.6, 7.4), sharex=True)

    for ax, level, ylab in (
        (axes[0], "ssp_bands", "per SSP"),
        (axes[1], "galaxies", "per galaxy"),
    ):
        for lab, runs in groups.items():
            run = min(runs, key=lambda r: abs(r["n_components"] - n_target))
            bands = run["ssp_bands"] if level == "ssp_bands" else \
                run.get("galaxies", {}).get("bands", {})
            if not bands:
                continue
            xs, ys = [], []
            for band, stats in bands.items():
                if band not in pivots:
                    continue
                xs.append(pivots[band])
                ys.append(abs(stats["mad"]))
            if not xs:
                continue
            o = np.argsort(xs)
            xs = np.asarray(xs)[o]
            ys = np.asarray(ys)[o]
            ax.plot(xs, ys, "-", color=SCHEME_COLOURS[lab], linewidth=1.6, alpha=0.95,
                    marker=SCHEME_MARKERS[lab], markersize=3.2, markeredgewidth=0)
            _direct_label(ax, xs[-1], ys[-1], lab, SCHEME_COLOURS[lab])

        floor = results["resampling_floor"]["median_mmag"]
        if floor > 0:
            ax.axhline(floor, color=_GREY, linestyle=":", linewidth=1.2)
            ax.annotate("resampling floor", xy=(ax.get_xlim()[0], floor),
                        xytext=(4, 3), textcoords="offset points",
                        fontsize=7.5, color=_GREY)
        ax.set_yscale("log")
        ax.set_xscale("log")
        _style_axes(ax, None, f"median |Δm|, {ylab}  (mmag)", None)
        ax.set_xlim(right=ax.get_xlim()[1] * 1.9)

    axes[1].set_xlabel("band pivot wavelength  (Å, rest frame)", fontsize=9.5, color=_INK)
    axes[0].set_title(f"Magnitude error by band, at N ≈ {n_target} components",
                      fontsize=11, color=_INK, loc="left", pad=8)
    fig.text(0.065, 0.945,
             "The blue end is where the schemes differ, and where the red-galaxy colour "
             "bias is generated.", fontsize=8.5, color=_MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def fig_red_fraction(results, path, threshold=0.01):
    """The acceptance criterion, as a function of storage."""
    import matplotlib.pyplot as plt

    groups = _by_scheme(results)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))

    for ax, key, title in (
        (axes[0], ("galaxies", "colour", "red_fraction_shift"),
         "Red-fraction shift at the calibrated cut"),
        (axes[1], ("galaxies", "colour", "red_minus_blue_differential_mmag"),
         "g−r bias: red population minus blue"),
    ):
        stack = _LabelStack(ax)
        for lab, runs in groups.items():
            x = np.array([r["bytes_per_galaxy"] for r in runs], dtype=float)
            y = np.array([_metric(r, key) for r in runs], dtype=float)
            ok = np.isfinite(y) & (np.abs(y) > 0)
            if ok.sum() == 0:
                continue
            ax.plot(x[ok], np.abs(y[ok]), "-", color=SCHEME_COLOURS[lab], linewidth=2.0,
                    marker=SCHEME_MARKERS[lab], markersize=4.5,
                    markeredgecolor="white", markeredgewidth=0.7, zorder=3)
            stack.add(x[ok][-1], abs(y[ok][-1]), lab, SCHEME_COLOURS[lab])

        keep = ("previous", "13 bins", "2 bins")
        for b in results["baselines"]:
            y = _metric(b, key)
            if not np.isfinite(y) or abs(y) == 0:
                continue
            ax.scatter([b["bytes_per_galaxy"]], [abs(y)], s=52, marker="*", color=_GREY,
                       zorder=4, edgecolor="white", linewidth=0.6)
            short = b["label"].replace("baseline: ", "")
            if any(k in short for k in keep):
                stack.add(b["bytes_per_galaxy"], abs(y), short, _GREY, side="below")

        ax.set_xscale("log")
        ax.set_yscale("log")
        _style_axes(ax, "bytes per galaxy", None, title)
        ax.set_xlim(ax.get_xlim()[0] * 0.75, ax.get_xlim()[1] * 3.0)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo * 0.5, hi * 2.0)
        stack.draw()

    axes[0].axhline(threshold, color="#B22222", linestyle="--", linewidth=1.4)
    axes[0].annotate(f"acceptance criterion  |Δf_red| < {threshold:g}",
                     xy=(axes[0].get_xlim()[0], threshold), xytext=(6, 4),
                     textcoords="offset points", fontsize=8.0, color="#B22222")
    axes[0].set_ylabel("|Δf_red|", fontsize=9.5, color=_INK)
    axes[1].set_ylabel("|red − blue| differential  (mmag)", fontsize=9.5, color=_INK)

    fig.suptitle("Does the basis move galaxies across the red/blue boundary?",
                 fontsize=12, color=_INK, x=0.06, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def fig_variance_trap(results, path):
    """Cumulative explained variance against the error a telescope would measure."""
    import matplotlib.pyplot as plt

    groups = _by_scheme(results)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))

    stack = _LabelStack(ax)
    for lab, runs in groups.items():
        v = np.array([1.0 - r.get("explained_variance_cumulative", np.nan) for r in runs])
        y = np.array([_metric(r, ("ssp_by_set", "paus_nb", "mad_mmag")) for r in runs])
        ok = np.isfinite(v) & np.isfinite(y) & (v > 0) & (y > 0)
        if ok.sum() == 0:
            continue
        ax.plot(v[ok], y[ok], "-", color=SCHEME_COLOURS[lab], linewidth=2.0,
                marker=SCHEME_MARKERS[lab], markersize=4.5,
                markeredgecolor="white", markeredgewidth=0.7)
        # component counts marked on one series only (same on every series; repeating obscures)
        if lab == "uniform":
            for r, xx, yy in zip(np.array(runs)[ok], v[ok], y[ok]):
                if r["n_components"] in (10, 25, 50, 100, 200):
                    ax.annotate(f"N={r['n_components']}", xy=(xx, yy), xytext=(5, 6),
                                textcoords="offset points", fontsize=7.5,
                                color=SCHEME_COLOURS[lab])
        stack.add(v[ok][-1], y[ok][-1], lab, SCHEME_COLOURS[lab])

    ax.set_xscale("log")
    ax.set_yscale("log")
    for (frac, lbl), dy in (((1e-4, "99.99 % of variance"), 8), ((1e-5, "99.999 %"), 22)):
        ax.axvline(frac, color=_GREY, linestyle=":", linewidth=1.0)
        ax.annotate(lbl, xy=(frac, ax.get_ylim()[0]), xytext=(4, dy),
                    textcoords="offset points", fontsize=7.5, color=_GREY, va="bottom")
    ax.invert_xaxis()
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi / 6.0)          # room at the low-variance end for the labels
    stack.draw()
    _style_axes(ax, "unexplained variance fraction  (1 − cumulative)",
                "PAUS narrow-band median |Δm|  (mmag)",
                "Explained variance is not an error budget")
    fig.text(0.075, 0.90,
             "Variance saturates long before the photometric error does. A '99.99 % of "
             "variance' rule would stop far too early.",
             fontsize=8.5, color=_MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def fig_age_metallicity(results, path, n_target=50):
    """Which stellar populations the basis reproduces badly."""
    import matplotlib.pyplot as plt

    groups = _by_scheme(results)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))

    for lab, runs in groups.items():
        run = min(runs, key=lambda r: abs(r["n_components"] - n_target))
        by_age = run.get("by_age", [])
        if by_age:
            x = [0.5 * (b["age_lo_Gyr"] + min(b["age_hi_Gyr"], 14.0)) for b in by_age]
            y = [b["mad_mmag"] for b in by_age]
            axes[0].plot(x, y, "-", color=SCHEME_COLOURS[lab], linewidth=2.0,
                         marker=SCHEME_MARKERS[lab], markersize=5,
                         markeredgecolor="white", markeredgewidth=0.7)
            _direct_label(axes[0], x[-1], y[-1], lab, SCHEME_COLOURS[lab])
        by_z = run.get("by_metallicity", [])
        if by_z:
            x = [b["logzsol"] for b in by_z]
            y = [b["mad_mmag"] for b in by_z]
            axes[1].plot(x, y, "-", color=SCHEME_COLOURS[lab], linewidth=2.0,
                         marker=SCHEME_MARKERS[lab], markersize=5,
                         markeredgecolor="white", markeredgewidth=0.7)
            _direct_label(axes[1], x[-1], y[-1], lab, SCHEME_COLOURS[lab], dx=1.0)

    ref = results["runs"][0].get("reference_band_for_age_Z", "g") if results["runs"] else "g"
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    _style_axes(axes[0], "SSP age  (Gyr)", f"median |Δm| in {ref}  (mmag)",
                f"Error by population age  (N ≈ {n_target})")
    axes[0].set_xlim(right=axes[0].get_xlim()[1] * 2.2)
    axes[1].set_yscale("log")
    _style_axes(axes[1], "log(Z/Z☉)", None, "Error by metallicity")
    axes[1].set_xlim(right=axes[1].get_xlim()[1] + 0.55)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def fig_d4000(results, path):
    """D4000 accuracy, measured directly and through the PAUS narrow bands."""
    import matplotlib.pyplot as plt

    groups = _by_scheme(results)
    keys = sorted({k for r in results["runs"] for k in r.get("d4000", {})
                   if k.startswith("D4000_n")})
    if not keys:
        return None
    keys = keys[:4]
    ncol = min(len(keys), 2)
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.7 * ncol, 4.1 * nrow), squeeze=False)

    for ax, key in zip(axes.ravel(), keys):
        for lab, runs in groups.items():
            x = np.array([r["bytes_per_galaxy"] for r in runs], dtype=float)
            y = np.array([_metric(r, ("d4000", key, "mad_frac")) for r in runs])
            ok = np.isfinite(y) & (y > 0)
            if ok.sum() == 0:
                continue
            ax.plot(x[ok], y[ok] * 100.0, "-", color=SCHEME_COLOURS[lab], linewidth=1.9,
                    marker=SCHEME_MARKERS[lab], markersize=4.2,
                    markeredgecolor="white", markeredgewidth=0.7)
            _direct_label(ax, x[ok][-1], y[ok][-1] * 100.0, lab, SCHEME_COLOURS[lab])
        ax.set_xscale("log")
        ax.set_yscale("log")
        pretty = key.replace("D4000_n/", "").replace("narrowband_z", "PAUS narrow bands, z=")
        _style_axes(ax, "bytes per galaxy", "median |ΔD4000| / D4000  (%)",
                    f"D4000$_n$ — {pretty}")
        ax.set_xlim(right=ax.get_xlim()[1] * 2.4)

    for ax in axes.ravel()[len(keys):]:
        ax.set_visible(False)
    fig.suptitle("D4000 reconstruction error (Renard et al. 2022, Eqs. 1–5)",
                 fontsize=11.5, color=_INK, x=0.05, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def fig_weight_curves(results, path, basis_dir=None, name=None):
    """The weight functions themselves — what each scheme actually asks for."""
    import matplotlib.pyplot as plt

    from galspectra.pca.basis import SpectralBasis

    if basis_dir is None:
        return None
    curves = {}
    for lab in _by_scheme(results):
        f = Path(basis_dir) / f"{name}_basis_{lab}.npz"
        if f.exists():
            b = SpectralBasis.load(f)
            curves[lab] = (b.wave, b.weights)
    if not curves:
        return None

    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    for lab, (w, wt) in curves.items():
        ax.plot(w, wt, "-", color=SCHEME_COLOURS[lab], linewidth=1.7)
        _direct_label(ax, w[-1], wt[-1], lab, SCHEME_COLOURS[lab])
    ax.axvspan(3600, 4400, color="#cccccc", alpha=0.35, zorder=0)
    ax.annotate("4000 Å break region", xy=(4000, ax.get_ylim()[1]), xytext=(0, -12),
                textcoords="offset points", fontsize=7.5, color=_MUTED, ha="center", va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style_axes(ax, "rest-frame wavelength  (Å)", "weight  (geometric mean = 1)",
                "What each weighting scheme asks the PCA to prioritise")
    ax.set_xlim(right=ax.get_xlim()[1] * 1.9)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────

def make_all_figures(results, fig_dir, name):
    import matplotlib
    matplotlib.use("Agg")

    from galspectra.photometry.filter_sets import describe_set, load_filter_set

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    _assign_styles(list(_by_scheme(results)))

    sets = results["config"]["photometry"]["sets"]
    pivots = {r["band"]: r["pivot_AA"] for r in describe_set(load_filter_set(*sets))}

    written = []
    for fn, fname in (
        (lambda p: fig_error_vs_bytes(results, p), "fig1_error_vs_bytes.png"),
        (lambda p: fig_error_by_wavelength(results, p, pivots), "fig2_error_by_wavelength.png"),
        (lambda p: fig_red_fraction(results, p), "fig3_red_fraction_shift.png"),
        (lambda p: fig_variance_trap(results, p), "fig4_variance_is_not_a_budget.png"),
        (lambda p: fig_age_metallicity(results, p), "fig5_error_by_age_metallicity.png"),
        (lambda p: fig_d4000(results, p), "fig6_d4000.png"),
        (lambda p: fig_weight_curves(
            results, p,
            basis_dir=Path(results["config"]["output"]["dir"]),
            name=name), "fig7_weight_curves.png"),
    ):
        out = fn(fig_dir / fname)
        if out is not None:
            written.append(out)
            print(f"    {out}")
    return written
