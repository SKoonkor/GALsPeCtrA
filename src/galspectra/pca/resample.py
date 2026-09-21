"""
Flux-conserving resampling onto a logarithmic wavelength grid.

BC03's native grid is piecewise (10 Å in the UV, 100 Å in the IR), so a PCA
fit on it implicitly overweights the finely-sampled optical. A constant-R
grid fixes that. Resampling must stay linear in flux with no per-spectrum
term (f_out = M @ f_in, M fixed) — `resampling_matrix` returns exactly that
matrix, built once and reused for every spectrum. Unlike `np.interp`, it is
also flux-conserving: passband integrals of the resampled spectrum match the
original, which matters since every metric here is a passband integral.

Each output bin is the exact mean of the (piecewise-linear) input over that
bin. Oversampling relative to the input's native resolution does not create
information; `resolution_report` makes that visible.
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
    """Bin centres/edges of a constant-resolving-power grid.

    wave_min, wave_max : range to cover, in Å (bin edges).
    resolution : R = λ/Δλ; edges are geometrically spaced with ratio exp(1/R).

    Returns (centres, edges). n_bins = ceil(R * ln(wave_max/wave_min)), so
    actual resolution is marginally higher than requested.
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

    # spread rounding excess over the range rather than overhang wave_max
    edges = wave_min * np.exp(np.linspace(0.0, np.log(wave_max / wave_min), n_bins + 1))
    centres = np.sqrt(edges[:-1] * edges[1:])
    return centres, edges


def cumulative_integral_matrix(wave_in, points):
    """Rows mapping node values to ∫ from wave_in[0] to each point.

    T @ f == cumulative integral of f (piecewise-linear on wave_in) at each
    point. Points outside the input range are clamped to the ends;
    `resampling_matrix` rejects out-of-range output grids outright.
    """
    wave_in = np.asarray(wave_in, dtype=float)
    points = np.atleast_1d(np.asarray(points, dtype=float))
    n_in = wave_in.size

    if n_in < 2:
        raise ValueError("wave_in needs at least 2 samples")
    if not np.all(np.diff(wave_in) > 0):
        raise ValueError("wave_in must be strictly increasing")

    dlam = np.diff(wave_in)

    # trapezoid contribution per segment, cumulative-summed -> C[k] = ∫_λ0^λk f dλ
    seg = np.zeros((n_in - 1, n_in))
    idx = np.arange(n_in - 1)
    seg[idx, idx] = dlam / 2.0
    seg[idx, idx + 1] = dlam / 2.0
    cum_nodes = np.zeros((n_in, n_in))
    cum_nodes[1:] = np.cumsum(seg, axis=0)        # (n_in, n_in)

    p = np.clip(points, wave_in[0], wave_in[-1])
    k = np.clip(np.searchsorted(wave_in, p, side="right") - 1, 0, n_in - 2)  # bracketing interval
    t = (p - wave_in[k]) / dlam[k]                # 0 at left node, 1 at right

    T = cum_nodes[k].copy()

    # partial-interval integral for linear f: (p-λ_k)/2 * ((2-t)f_k + t f_{k+1})
    half_span = (p - wave_in[k]) / 2.0
    rows = np.arange(p.size)
    T[rows, k] += half_span * (2.0 - t)
    T[rows, k + 1] += half_span * t
    return T


def resampling_matrix(wave_in, edges_out, tol=1e-9):
    """Flux-conserving resampling operator: f_out = M @ f_in.

    wave_in : (N_in,) input wavelengths, strictly increasing, Å
    edges_out : (N_out+1,) output bin edges, strictly increasing, Å
    tol : relative slack when checking the output grid lies inside the input range

    f_out[j] is the mean of the piecewise-linear input over output bin j.
    Raises if edges_out extends beyond wave_in — never extrapolate the library.
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
    """Per-sample resolving power λ/Δλ (Δλ = mean of neighbouring gaps)."""
    wave = np.asarray(wave, dtype=float)
    dlam = np.empty_like(wave)
    gaps = np.diff(wave)
    dlam[0] = gaps[0]
    dlam[-1] = gaps[-1]
    dlam[1:-1] = 0.5 * (gaps[:-1] + gaps[1:])
    return wave / dlam


def resolution_report(wave_in, resolution, wave_min=None, wave_max=None):
    """Compare a requested resolving power to the input grid's native one.

    Returns the fraction of the range where the request exceeds native
    resolution (interpolating, not measuring) and the worst oversampling factor.
    Reported, not enforced.
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
