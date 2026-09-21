"""
Diagonal weighting schemes for the spectral PCA.

Weights must be a fixed function of wavelength only — never per-spectrum — or
reconstruction stops being linear in flux and a composite population is no
longer the mass-weighted sum of its SSPs. The failure is silent (each SSP
still reconstructs fine; only galaxies come out wrong), so `build_weights`
enforces (N_wave,) shape and refuses the per-spectrum schemes `pca/preprocess.py`
still offers, by name.

`inverse_std` reproduces `preprocess.normalize_seds(method="std")` — the
committed basis's weighting — described here as what it structurally is.
"""

from __future__ import annotations

import numpy as np

__all__ = ["SCHEMES", "REFUSED_SCHEMES", "build_weights", "check_weights"]

# Schemes that are safe because they depend on wavelength only.
SCHEMES = ("uniform", "inverse_std", "inverse_rms", "inverse_mean", "piecewise")

# Schemes offered by pca/preprocess.py that must never reach a basis fit.
REFUSED_SCHEMES = {
    "l2": (
        "'l2' divides each spectrum by its own L2 norm. That is a per-spectrum "
        "operation, so reconstruction stops being linear in flux and a composite "
        "population is no longer the mass-weighted sum of its SSPs. The error does "
        "not show up per-SSP — only at the galaxy level."
    ),
    "5500A": (
        "'5500A' divides each spectrum by its own value at 5500 Å. Same failure mode "
        "as 'l2': per-spectrum, therefore non-linear in flux, therefore silently "
        "wrong for composite populations."
    ),
    "log": (
        "PCA on log(flux) is not linear in flux: exp(Σ log) is a product, not a sum, "
        "so mass-weighted composition is impossible. Never run the basis on logs."
    ),
    "peak": (
        "Normalising by each spectrum's peak is per-spectrum, therefore non-linear "
        "in flux."
    ),
}

_EPS = 1e-300  # only guards true zeros; never large enough to bias a real weight


def check_weights(w, n_wave):
    """Validate a weight vector; returns it as contiguous float64.

    The shape check makes a per-spectrum weight structurally impossible rather
    than a convention to remember.
    """
    w = np.ascontiguousarray(np.asarray(w, dtype=float))
    if w.ndim != 1:
        raise ValueError(
            f"weights must be 1-D over wavelength, got shape {w.shape}. A 2-D weight "
            "would be per-spectrum and would break linearity in flux."
        )
    if w.size != n_wave:
        raise ValueError(f"weights have length {w.size}, expected {n_wave}")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights contain non-finite values")
    if np.any(w <= 0):
        n_bad = int(np.sum(w <= 0))
        raise ValueError(
            f"{n_bad} weights are <= 0. A zero weight makes that wavelength "
            "unrecoverable (the basis is divided by w), so it must be excluded from "
            "the wavelength range instead."
        )
    return w


def _piecewise_factors(wave, regions):
    """Per-wavelength multiplier from a list of {min, max, factor} regions.

    Regions are applied in order and may overlap; later entries win. Wavelengths
    matched by no region keep a factor of 1.
    """
    factors = np.ones(wave.size, dtype=float)
    for r in regions:
        lo = float(r.get("min", -np.inf))
        hi = float(r.get("max", np.inf))
        fac = float(r["factor"])
        if fac <= 0:
            raise ValueError(f"piecewise factor must be positive, got {fac}")
        factors[(wave >= lo) & (wave <= hi)] = fac
    return factors


def build_weights(scheme, X, wave, base="uniform", regions=None, normalise=True):
    """Construct a diagonal weight vector over wavelength.

    scheme : 'uniform' (w=1), 'inverse_std' (w=1/std, the incumbent),
        'inverse_rms' (w=1/rms), 'inverse_mean' (w=1/mean), or 'piecewise'
        (a `base` scheme times per-region factors from `regions`).
    X : (N_ssp, N_wave) SSP library the weights are derived from.
    wave : (N_wave,) Å, needed by 'piecewise'.
    normalise : rescale so the geometric mean weight is 1 (cosmetic — an
        overall scalar on w cancels exactly in fit -> reconstruct).

    Returns (w, meta): w is (N_wave,) float64; meta is provenance for the basis file.
    """
    if scheme in REFUSED_SCHEMES:
        raise ValueError(
            f"weighting scheme '{scheme}' is refused.\n  {REFUSED_SCHEMES[scheme]}\n"
            f"  Permitted schemes: {', '.join(SCHEMES)}"
        )
    if scheme not in SCHEMES:
        raise ValueError(f"unknown weighting scheme '{scheme}'. Choose from {SCHEMES}")

    X = np.asarray(X, dtype=float)
    wave = np.asarray(wave, dtype=float)
    if X.ndim != 2 or X.shape[1] != wave.size:
        raise ValueError(f"X has shape {X.shape}, incompatible with {wave.size} wavelengths")

    n_wave = wave.size
    meta = {"scheme": scheme, "normalise": bool(normalise)}

    if scheme == "uniform":
        w = np.ones(n_wave)
    elif scheme == "inverse_std":
        w = 1.0 / (X.std(axis=0) + _EPS)
    elif scheme == "inverse_rms":
        w = 1.0 / (np.sqrt(np.mean(X ** 2, axis=0)) + _EPS)
    elif scheme == "inverse_mean":
        w = 1.0 / (np.abs(X.mean(axis=0)) + _EPS)
    elif scheme == "piecewise":
        if not regions:
            raise ValueError("scheme 'piecewise' requires a non-empty `regions` list")
        if base == "piecewise":
            raise ValueError("piecewise cannot use itself as a base")
        w_base, base_meta = build_weights(base, X, wave, normalise=False)
        w = w_base * _piecewise_factors(wave, regions)
        meta["base"] = base
        meta["base_meta"] = base_meta
        meta["regions"] = list(regions)

    if not np.all(np.isfinite(w)) or np.any(w <= 0):
        # a dead bin (zero across the library) is a range problem, not a weighting one
        bad = ~np.isfinite(w) | (w <= 0)
        raise ValueError(
            f"{int(bad.sum())} wavelength bins gave a non-positive or non-finite "
            f"weight under '{scheme}' (first at {wave[bad][0]:.1f} Å). These bins are "
            "identically zero across the library; narrow the wavelength range."
        )

    if normalise:
        w = w / np.exp(np.mean(np.log(w)))

    w = check_weights(w, n_wave)
    meta["w_min"] = float(w.min())
    meta["w_max"] = float(w.max())
    meta["w_dynamic_range"] = float(w.max() / w.min())
    return w, meta
