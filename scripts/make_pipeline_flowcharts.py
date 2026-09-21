"""Draw how a spectrum is computed *inside* the L-GALAXIES executable.

Produces `documents/incode_spectra_flowcharts.pdf`, three landscape pages:

    1. L-GALAXIES' own photometry  ->  Mag / MagDust / ObsMag
    2. The GALsPeCtrA in-code PCA  ->  pca_coeffs[50]
    3. The two side by side, with the places they diverge

Scope
-----
Only what runs in the C executable. Fitting the PCA basis is a separate story and is
notebook 02's subject; this chart starts from the basis already existing on disk.

Why every box carries a file:line
---------------------------------
So the chart can be checked against the source, and so it fails loudly rather than
quietly when the code moves. Every reference here was verified against the tree at
`../L-GALAXIES/LGalaxies2020_PublicRepository-master` for a build with
COMPUTE_SPECPHOT_PROPERTIES, POST_PROCESS_MAGS, COMP_PCA_COEFFICIENTS,
DETAILED_METALS_AND_MASS_RETURN and BC03 on, FULL_SPECTRA and ICL off.

The framing matters and is easy to get wrong: **neither path is an "on-the-fly SED".**
Nothing spectral is simulation state -- `pca_coeffs` lives in `struct GALAXY_OUTPUT` and
not in the runtime `struct GALAXY` -- and both paths run at snapshot-output time, back to
back, over the whole stored star-formation history. See `documents/onthefly_factcheck.md`.

Usage
-----
    python scripts/make_pipeline_flowcharts.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PDF = PROJECT_ROOT / "documents" / "incode_spectra_flowcharts.pdf"

PAGE = (11.69, 8.27)          # landscape A4, inches

# One hue per path, grey for what they share, and an accent reserved for divergences.
# Chosen to stay distinguishable in greyscale: the fills differ in lightness, not only hue.
C_LGAL   = {"fc": "#DCE6F2", "ec": "#2F5597"}
C_PCA    = {"fc": "#FBE3D6", "ec": "#C55A11"}
C_SHARED = {"fc": "#EDEDED", "ec": "#666666"}
C_TABLE  = {"fc": "#E4EFDC", "ec": "#548235"}
C_OUT    = {"fc": "#FFF2CC", "ec": "#BF8F00"}
C_WARN   = "#C00000"


# Geometry, in axes fractions of a 8.27 in page. Box heights are COMPUTED from the
# content rather than hand-set: the first version of this script hand-sized every box and
# every one of them overflowed, with the file:line drawn on top of the body text.
LINE_H  = 0.0152      # one line of body text
TITLE_H = 0.0225
REF_H   = 0.0165
PAD_T   = 0.0100
PAD_B   = 0.0075
GAP     = 0.0105      # body-to-ref breathing room
FLOOR   = 0.030       # content must stay above the footer


def box_height(title, body=None, ref=None):
    h = PAD_T + PAD_B
    if title:
        h += TITLE_H
    if body:
        h += LINE_H * len(body.split("\n"))
    if ref:
        h += REF_H + (GAP if body else 0.0)
    return h


def box(ax, x, y_top, w, title, body=None, ref=None, style=None, fs=8.2, h=None):
    """Draw a node whose top edge is at `y_top`. Returns the bottom edge.

    Height is derived from the content unless `h` is given (used to level a row).
    """
    style = style or C_SHARED
    h = h if h is not None else box_height(title, body, ref)
    ax.add_patch(FancyBboxPatch(
        (x, y_top - h), w, h, boxstyle="round,pad=0.003,rounding_size=0.010",
        facecolor=style["fc"], edgecolor=style["ec"], linewidth=1.1, zorder=2))
    cx = y = x + w / 2.0
    cursor = y_top - PAD_T
    if title:
        ax.text(cx, cursor, title, ha="center", va="top", fontsize=fs + 0.7,
                fontweight="bold", color="#111111", zorder=3)
        cursor -= TITLE_H
    if body:
        ax.text(cx, cursor, body, ha="center", va="top", fontsize=fs,
                color="#222222", linespacing=1.42, zorder=3)
        cursor -= LINE_H * len(body.split("\n"))
    if ref:
        ax.text(cx, cursor - (GAP if body else 0.0) + 0.002, ref, ha="center", va="top",
                fontsize=fs - 1.4, family="monospace", color=style["ec"], zorder=3)
    return y_top - h


def row(ax, specs, y_top, gap=0.018, x0=0.035, x1=0.965):
    """A row of boxes levelled to a common height. `specs` = (title, body, ref, style)."""
    n = len(specs)
    w = ((x1 - x0) - gap * (n - 1)) / n
    h = max(box_height(t, b, r) for t, b, r, _ in specs)
    xs = [x0 + i * (w + gap) for i in range(n)]
    for x, (t, b, r, st) in zip(xs, specs):
        box(ax, x, y_top, w, t, b, r, st, h=h)
    return y_top - h, [x + w / 2 for x in xs]


def arrow(ax, x0, y0, x1, y1, label=None, color="#555555", ls="-"):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
        linewidth=1.15, color=color, linestyle=ls,
        shrinkA=0.5, shrinkB=0.5, zorder=1))
    if label:
        ax.text((x0 + x1) / 2 + 0.010, (y0 + y1) / 2, label, ha="left", va="center",
                fontsize=7.0, color=color, style="italic", zorder=3)


def page(title, subtitle):
    fig = plt.figure(figsize=PAGE)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.986, title, ha="center", va="top", fontsize=15, fontweight="bold")
    ax.text(0.5, 0.957, subtitle, ha="center", va="top", fontsize=8.0,
            color="#444444", linespacing=1.55)
    return fig, ax


def check_fits(y_bottom, page_name):
    """Fail loudly if the flow ran off the page.

    The first version of this script sized every box by hand and page 1 clipped its last
    two nodes; nothing complained. Cheap assertion, expensive bug.
    """
    if y_bottom < FLOOR:
        raise SystemExit(
            f"{page_name}: content ends at y={y_bottom:.3f}, below the {FLOOR} floor — "
            f"it would be clipped. Shorten a box or tighten the gaps.")


def footer(ax, text):
    ax.text(0.5, 0.012, text, ha="center", va="bottom", fontsize=6.8,
            color="#888888", style="italic")


# ── The framing line that must appear on every page ─────────────────────────
FRAMING = ("Runs at snapshot-output time, over the whole stored SFH.  Nothing spectral is simulation state.  —  documents/onthefly_factcheck.md")


def page_lgalaxies(pdf):
    fig, ax = page("1 · L-GALAXIES' own photometry",
                   "How the model turns a star-formation history into Mag, MagDust and "
                   "ObsMag\n" + FRAMING)
    W, X, CX = 0.46, 0.27, 0.50
    y = 0.918
    y = box(ax, X, y, W, "prepare_galaxy_for_output()  calls  post_process_spec_mags(o)",
            None, "save.c:645", C_SHARED)
    arrow(ax, CX, y, CX, y - 0.016); y -= 0.016
    y = box(ax, X, y, W, "Copy the SFH out of the output struct",
            "o->sfh_DiskMass / sfh_BulgeMass / sfh_ICM + metal channels\n"
            "->  local  struct SFH_BIN sfh_bins[]",
            "post_process_spec_mags.c:45-58", C_SHARED)
    arrow(ax, CX, y, CX, y - 0.016); y -= 0.016
    y = box(ax, X, y, W, "for nlum = 0 .. NMAG-1   (5 bands, outer)",
            "for ll = 0 .. o->sfh_ibin   (populated SFH bins)   ·   N_AgeBin is always 1",
            "post_process_spec_mags.c:68, 88, 96-108", C_LGAL)
    y_split = y
    y, cxs = row(ax, [
        ("Bin-centre age  (code units)",
         "age = SFH_t[snap][0][ll]\n"
         "     + SFH_dt[snap][0][ll]/2 − NumToTime(snap)",
         "post_process_spec_mags.c:124,127", C_LGAL),
        ("Mass  →  units of 1e11 M☉",
         "diskmass = sfh_DiskMass × 0.1 / h\n"
         "internal unit is 1e10 M☉/h, so h cancels",
         "post_process_spec_mags.c:136", C_LGAL),
        ("Metallicity  (absolute Z)",
         "Z = Σ sfh_MetalsDiskMass[ii] / sfh_DiskMass\n"
         "no explicit clamp on this side",
         "post_process_spec_mags.c:145-148", C_LGAL),
    ], y_split - 0.032)
    for cx in cxs:
        arrow(ax, CX, y_split, cx, y_split - 0.030)
    y_merge = y
    for cx in cxs:
        arrow(ax, cx, y_merge, CX, y_merge - 0.028)
    y = box(ax, 0.135, y_merge - 0.030, 0.73,
            "Look the luminosity up in the precomputed SSP tables",
            "compute_post_process_lum()  →  find_interpolated_lum()\n"
            "BILINEAR: linear in log₁₀(age) × linear in metallicity.  Outside the\n"
            "table it snaps to the edge — L-GALAXIES' de-facto Z clamp.\n"
            "The snapshot axis is NOT interpolated: ObsMag reads an exact column.\n"
            "tables:  PhotTables/{PhotPrefix}_{SSP}_{IMF}_Phot_Table_{Sim}_Mag{band}_m{Z}.dat",
            "post_process_spec_mags.c:334 · model_spectro_photometric.c:378-459, :101",
            C_TABLE)
    arrow(ax, CX, y, CX, y - 0.016); y -= 0.016
    y = box(ax, 0.235, y, 0.53, "Accumulate luminosity per band",
            "LumDisk, LumBulge, the young-star terms YLumDisk/YLumBulge, and ObsLum* copies",
            "post_process_spec_mags.c:167-200", C_LGAL)
    arrow(ax, CX, y, CX, y - 0.016); y -= 0.016
    y = box(ax, 0.165, y, 0.67, "Apply dust",
            "disk: dust_correction_for_post_processing() — ISM + birth cloud.  bulge: birth\n"
            "cloud only.  n_H ∝ (1+z)⁻¹ using ZZ[snap].  dust_model() is dead code here.",
            "post_process_spec_mags.c:401-527, :417, :274-277", C_LGAL)
    arrow(ax, CX, y, CX, y - 0.016); y -= 0.016
    box(ax, 0.185, y, 0.63,
        "Mag[5]   MagBulge[5]   MagDust[5]   ObsMag[5]   ObsMagDust[5]",
        "lum_to_mag() returns the sentinel 99.0 when luminosity ≤ 0 — the \"< 90\" masks",
        "post_process_spec_mags.c:230-298 · model_misc.c:946-952", C_OUT)
    check_fits(y - box_height("t", "one line", "ref"), "page 1")
    footer(ax, "Verified against the L-GALAXIES tree · regenerate with "
               "scripts/make_pipeline_flowcharts.py")
    pdf.savefig(fig); plt.close(fig)


def page_galspectra(pdf):
    fig, ax = page("2 · The GALsPeCtrA in-code PCA path",
                   "How the same star-formation history becomes pca_coeffs[50]\n" + FRAMING)
    CX = 0.50
    y = 0.918
    y = box(ax, 0.24, y, 0.52,
            "…  called immediately after post_process_spec_mags(o)",
            None, "save.c:649", C_SHARED)
    arrow(ax, CX, y, CX, y - 0.032); y -= 0.032
    yb, cxs = row(ax, [
        ("Once, at start-up:  setup_pca_tables()",
         "reads SpecPhotTables/PCA/pca_coeff_grid_bc03.bin\n"
         "header: 3 × int32  →  200 ages × 6 Z × 50 PCs\n"
         "then age_gyr[], logzsol[], coeff_grid[] float64\n"
         "WHAT IT HOLDS: mean-subtracted coefficients of a\n"
         "per-wavelength z-scored SSP spectrum. The PCA mean\n"
         "and components are never shipped to C.",
         "init.c:195 · pca_sed.c:86-131", C_TABLE),
        ("compute_pca_coefficients(o)",
         "reads only o->sfh_*, o->SnapNum and SFH_t / SFH_dt —\n"
         "no data dependency on the magnitudes computed above\n\n"
         "double sum[50] = {0}\n"
         "for ll = 0 .. o->sfh_ibin: one pass over the SFH,\n"
         "all 50 coefficients per bin from one interpolation",
         "pca_sed.c:146, 152, 154", C_PCA),
    ], y)
    arrow(ax, CX, y, cxs[1], y - 0.016)
    y = yb
    y_split = y
    y, cxs3 = row(ax, [
        ("Bin-centre age  →  Gyr",
         "age_gyr = (SFH_t + SFH_dt/2\n"
         "     − NumToTime(snap))\n"
         "  × UnitTime_in_years / h / 1e9", "pca_sed.c:156-162", C_PCA),
        ("Mass  →  M☉",
         "mass = sfh_DiskMass × 1e10 / h\n"
         "disk and bulge handled in turn,\ninto the same accumulator",
         "pca_sed.c:167, 184", C_PCA),
        ("Metallicity  →  logzsol",
         "Z = Σ metals / mass,\nCLAMPED to [1e-4, 0.05]\n"
         "logzsol = log₁₀(Z / 0.02)", "pca_sed.c:172-175, 189-192", C_PCA),
    ], y_split - 0.048)
    for cx in cxs3:
        arrow(ax, CX, y_split, cx, y_split - 0.046)
    y_merge = y
    for cx in cxs3:
        arrow(ax, cx, y_merge, CX, y_merge - 0.044)
    y = box(ax, 0.185, y_merge - 0.046, 0.63,
            "bilinear_interp_pca(age_gyr, logzsol, tmp)",
            "LINEAR in age (Gyr — not log age), linear in logzsol.\n"
            "Outside the grid it CLAMPS; it never extrapolates.\n"
            "Returns one 50-vector for this (age, Z).",
            "pca_sed.c:36-71, :52-53", C_TABLE)
    arrow(ax, CX, y, CX, y - 0.032); y -= 0.032
    y = box(ax, 0.30, y, 0.40, "sum[k] += mass × tmp[k]",
            "disk and bulge accumulate into the SAME sum[]",
            "pca_sed.c:178-180, 195-197", C_PCA)
    arrow(ax, CX, y, CX, y - 0.032); y -= 0.032
    box(ax, 0.165, y, 0.67, "o->pca_coeffs[k] = (float) sum[k]",
        "A mass-weighted sum, NOT divided by total mass. Rest-frame only.\n"
        "ICM is excluded — as it is from Mag.\n"
        "Offline, Python divides by M, adds the PCA mean and components back,\n"
        "un-z-scores, and multiplies by M again to get a spectrum.",
        "pca_sed.c:200-201", C_OUT)
    check_fits(y - box_height("o->pca_coeffs[k] = (float) sum[k]",
        "a\nb\nc\nd", "pca_sed.c:200-201"), "page 2")
    footer(ax, "Verified against the L-GALAXIES and GALsPeCtrA trees")
    pdf.savefig(fig); plt.close(fig)


def page_comparison(pdf):
    fig, ax = page("3 · The two side by side",
                   "Same galaxy, same moment, same SFH array — every difference is "
                   "downstream of that")
    y = box(ax, 0.22, 0.905, 0.56,
            "One star-formation history in struct GALAXY_OUTPUT",
            "o->sfh_DiskMass[], o->sfh_BulgeMass[], their metal channels, o->sfh_ibin\n"
            "Identical bin-centre age:  SFH_t + SFH_dt/2 − NumToTime(snap)",
            "save.c:645  then  save.c:649", C_SHARED)
    ax.text(0.175, y - 0.022, "L-GALAXIES", ha="center", fontsize=11,
            fontweight="bold", color=C_LGAL["ec"])
    ax.text(0.825, y - 0.022, "GALsPeCtrA in-code", ha="center", fontsize=11,
            fontweight="bold", color=C_PCA["ec"])
    rows = [
        ("Loop shape", "5 bands  ×  SFH bins\n(band is the outer loop)",
         "SFH bins once\n(all 50 PCs per interpolation)"),
        ("Mass to the table", "× 0.1 / h   →  units of 1e11 M☉\npost_process_spec_mags.c:136",
         "× 1e10 / h  →  M☉\npca_sed.c:167"),
        ("Metallicity", "raw Z, clamped only at the table edge\n"
         "model_spectro_photometric.c:427-438",
         "Z clamped to [1e-4, 0.05] explicitly\npca_sed.c:172-175"),
        ("Interpolated in", "log₁₀(age)  ×  metallicity\n"
         "model_spectro_photometric.c:412,446",
         "LINEAR age (Gyr)  ×  logzsol\npca_sed.c:44-49"),
        ("Off the grid", "snaps to the edge entry", "clamps; never extrapolates"),
        ("Redshift", "exact per-snapshot column → ObsMag\n(never interpolated)",
         "none — coefficients are rest-frame"),
        ("Emits", "Mag, MagBulge, MagDust, ObsMag,\nObsMagDust   (5 bands each)",
         "pca_coeffs[50], unnormalised,\nPCA mean NOT included"),
    ]
    y -= 0.048
    for name, left, right in rows:
        h = max(box_height(None, left, None), box_height(None, right, None))
        box(ax, 0.020, y, 0.315, None, left, None, C_LGAL, fs=7.9, h=h)
        box(ax, 0.665, y, 0.315, None, right, None, C_PCA, fs=7.9, h=h)
        ax.text(0.5, y - h / 2, name, ha="center", va="center",
                fontsize=8.8, fontweight="bold", color=C_WARN)
        y -= h + 0.036
    ax.text(0.5, y - 0.004,
            "One further difference, and it is between the C and the Python rather than "
            "between the two charts:\n"
            "the C clamps at the grid boundary, while csp/interpolator.py builds "
            "RegularGridInterpolator(fill_value=None) and extrapolates.\n"
            "Metallicity cannot leave the grid on either side; age can — the grid runs "
            "1e-4 to 13.7 Gyr.",
            ha="center", va="top", fontsize=8, color=C_WARN, linespacing=1.6)
    footer(ax, "documents/incode_spectra_flowcharts.pdf · companion to "
               "documents/onthefly_factcheck.md")
    pdf.savefig(fig); plt.close(fig)


def main():
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        page_lgalaxies(pdf)
        page_galspectra(pdf)
        page_comparison(pdf)
        d = pdf.infodict()
        d["Title"] = "How spectra are computed inside L-GALAXIES"
        d["Subject"] = ("L-GALAXIES native photometry vs the GALsPeCtrA in-code PCA path, "
                        "verified against source")
    print(f"wrote {OUT_PDF.relative_to(PROJECT_ROOT)}  "
          f"({OUT_PDF.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
