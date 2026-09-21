"""
Flux-conserving resampling onto a logarithmic wavelength grid.

Why this module exists
----------------------
The BC03 SSP library is delivered on a *piecewise* wavelength grid: 10 Å in the UV,
20 Å across the optical, 50 Å and then 100 Å in the near-infrared. A PCA fitted on
that grid weights each wavelength bin equally, so the optical — which happens to be
finely sampled — dominates the variance purely because it contributes more columns.
A grid of constant resolving power R = λ/Δλ removes that accident: every decade of
wavelength contributes the same number of columns.

The linearity requirement
-------------------------
The whole GALsPeCtrA architecture rests on reconstruction being *linear in flux*, so
that a composite population is the mass-weighted sum of its simple populations. Any
resampling therefore has to be a linear operator with no per-spectrum term:

    f_out = M @ f_in          M fixed, independent of f_in

`resampling_matrix()` returns exactly that matrix. It is built once and applied to
every spectrum, so it cannot introduce a per-spectrum normalisation by accident.
Note that `np.interp` is *also* linear in this sense, but it is not flux-conserving:
integrating an interpolated spectrum over a passband does not give the same answer as
integrating the original. Since every metric in this project is a passband integral,
that difference matters, and this module integrates exactly instead.

Method
------
The input is treated as piecewise linear in λ between its samples — the same
assumption `np.trapezoid` makes, and the same one the synthetic photometry makes.
Each output bin holds the exact mean of that piecewise-linear function over the bin:

    f_out[j] = (1 / (e[j+1] - e[j])) * ∫_{e[j]}^{e[j+1]} f(λ) dλ

Because the integral of a piecewise-linear function is a linear functional of the node
values, this is a matrix. It is assembled from a cumulative-integral operator T, where
row k of T maps node values to ∫ from λ_0 to an arbitrary point.

A consequence worth stating: this operation cannot create information. Requesting a
resolution finer than the input grid produces a smooth interpolation of the input, not
extra spectral structure. `resolution_report()` exists to make that visible rather than
letting a config file quietly imply a resolution the library does not have.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "log_wavelength_grid",
    "cumulative_integral_matrix",
    "resampling_matrix",
    "native_resolution",
    "resolution_report",
]


def log_wavelength_grid(wave_min, wave_max, resolution):
    """Bin centres and edges of a grid with constant resolving power.

    Parameters
    ----------
    wave_min, wave_max : float
        Range to cover, in Å. Both are bin *edges*, so the returned centres lie
        strictly inside.
    resolution : float
        R = λ/Δλ. Bin edges are geometrically spaced with ratio exp(1/R).

    Returns
    -------
    centres : (N,) float64 — geometric bin centres, sqrt(e_j * e_{j+1})
    edges   : (N+1,) float64

    The number of bins is ceil(R * ln(λmax/λmin)), so the actual resolution is
    marginally higher than requested and the range is covered exactly.
    """
    wave_min = float(wave_min)
    wave_max = float(wave_max)
    if not (0 < wave_min < wave_max):
        raise ValueError(f"need 0 < wave_min < wave_max, got {wave_min}, {wave_max}")
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")

    n_bins = int(np.ceil(resolution * np.log(wave_max / wave_min)))
    if n_bins < 2:
        raise ValueError(
            f"R={resolution:g} over {wave_min:g}-{wave_max:g} Å gives {n_bins} bins"
        )

    # Spread the (sub-bin) rounding excess over the whole range rather than
    # letting the last bin overhang wave_max.
    edges = wave_min * np.exp(np.linspace(0.0, np.log(wave_max / wave_min), n_bins + 1))
    centres = np.sqrt(edges[:-1] * edges[1:])
    return centres, edges


def cumulative_integral_matrix(wave_in, points):
    """Rows mapping node values to ∫ from wave_in[0] to each point.

    Returns
    -------
    T : (len(points), len(wave_in)) float64
        T @ f  ==  [∫_{λ0}^{p} f(λ) dλ  for p in points], for f piecewise linear
        on `wave_in`.

    Points outside [wave_in[0], wave_in[-1]] are clamped to the ends, i.e. the
    function is treated as zero beyond the library range. Callers should not rely
    on that: `resampling_matrix` rejects out-of-range output grids outright.
    """
    wave_in = np.asarray(wave_in, dtype=float)
    points = np.atleast_1d(np.asarray(points, dtype=float))
    n_in = wave_in.size

    if n_in < 2:
        raise ValueError("wave_in needs at least 2 samples")
    if not np.all(np.diff(wave_in) > 0):
        raise ValueError("wave_in must be strictly increasing")

    dlam = np.diff(wave_in)                       # (n_in-1,)

    # Node-to-node trapezoid contributions, as a matrix:
    #   ∫_{λ_k}^{λ_{k+1}} f dλ = dlam[k] * (f_k + f_{k+1}) / 2
    # Cumulative sum of those gives C[k] = ∫_{λ_0}^{λ_k} f dλ.
    seg = np.zeros((n_in - 1, n_in))
    idx = np.arange(n_in - 1)
    seg[idx, idx] = dlam / 2.0
    seg[idx, idx + 1] = dlam / 2.0
    cum_nodes = np.zeros((n_in, n_in))
    cum_nodes[1:] = np.cumsum(seg, axis=0)        # (n_in, n_in)

    p = np.clip(points, wave_in[0], wave_in[-1])
    # Bracketing interval [k, k+1] for each point.
    k = np.clip(np.searchsorted(wave_in, p, side="right") - 1, 0, n_in - 2)
    t = (p - wave_in[k]) / dlam[k]                # 0 at the left node, 1 at the right

    T = cum_nodes[k].copy()                       # (n_points, n_in)

    # Partial interval: ∫_{λ_k}^{p} f dλ with f linear
    #   = (p - λ_k)/2 * ((2 - t) f_k + t f_{k+1})
    half_span = (p - wave_in[k]) / 2.0
    rows = np.arange(p.size)
    T[rows, k] += half_span * (2.0 - t)
    T[rows, k + 1] += half_span * t
    return T


def resampling_matrix(wave_in, edges_out, tol=1e-9):
    """Flux-conserving resampling operator onto the bins defined by `edges_out`.

    Parameters
    ----------
    wave_in : (N_in,) — input sample wavelengths, strictly increasing, Å
    edges_out : (N_out+1,) — output bin edges, strictly increasing, Å
    tol : float — relative slack allowed when checking that the output grid lies
        inside the input range, to absorb float round-off in grid construction.

    Returns
    -------
    M : (N_out, N_in) float64 — f_out = M @ f_in, with f_out[j] the mean of the
        piecewise-linear input over bin j.

    Raises
    ------
    ValueError if the output grid extends beyond the input range. Extrapolating a
    stellar library past its own wavelength coverage is never the right answer, and
    silently returning zeros there would put a fake absorption trough in the basis.
    """
    wave_in = np.asarray(wave_in, dtype=float)
    edges_out = np.asarray(edges_out, dtype=float)

    if not np.all(np.diff(edges_out) > 0):
        raise ValueError("edges_out must be strictly increasing")

    slack_lo = wave_in[0] * (1.0 - tol)
    slack_hi = wave_in[-1] * (1.0 + tol)
    if edges_out[0] < slack_lo or edges_out[-1] > slack_hi:
        raise ValueError(
            f"output grid {edges_out[0]:.4f}-{edges_out[-1]:.4f} Å is not contained "
            f"in the input grid {wave_in[0]:.4f}-{wave_in[-1]:.4f} Å. "
            "Narrow the requested range; do not extrapolate the library."
        )

    T = cumulative_integral_matrix(wave_in, edges_out)     # (N_out+1, N_in)
    widths = np.diff(edges_out)[:, None]
    return (T[1:] - T[:-1]) / widths


def native_resolution(wave):
    """Per-sample resolving power λ/Δλ of an arbitrary grid.

    Δλ is the local sample spacing (the mean of the two neighbouring gaps, and the
    single gap at the ends), so the answer is symmetric and does not jump at the
    boundaries between the BC03 grid's constant-Δλ blocks.
    """
    wave = np.asarray(wave, dtype=float)
    dlam = np.empty_like(wave)
    gaps = np.diff(wave)
    dlam[0] = gaps[0]
    dlam[-1] = gaps[-1]
    dlam[1:-1] = 0.5 * (gaps[:-1] + gaps[1:])
    return wave / dlam


def resolution_report(wave_in, resolution, wave_min=None, wave_max=None):
    """Compare a requested resolving power against what the input grid supplies.

    Returns a dict with the fraction of the requested range where the request
    exceeds the native resolution — i.e. where resampling interpolates rather than
    measures — and the worst-case oversampling factor.

    This is reported, not enforced. Oversampling is legitimate (it makes the PCA's
    implicit metric uniform, and passbands wider than the native sampling are
    unaffected); what is not legitimate is claiming spectral detail the library does
    not contain.
    """
    wave_in = np.asarray(wave_in, dtype=float)
    lo = wave_in[0] if wave_min is None else float(wave_min)
    hi = wave_in[-1] if wave_max is None else float(wave_max)

    sel = (wave_in >= lo) & (wave_in <= hi)
    if sel.sum() < 2:
        raise ValueError("requested range contains fewer than 2 input samples")

    r_native = native_resolution(wave_in)[sel]
    oversampled = r_native < resolution
    return {
        "requested_R": float(resolution),
        "native_R_min": float(r_native.min()),
        "native_R_median": float(np.median(r_native)),
        "native_R_max": float(r_native.max()),
        "fraction_oversampled": float(oversampled.mean()),
        "max_oversampling_factor": float(resolution / r_native.min()),
        "interpolating": bool(oversampled.any()),
    }
