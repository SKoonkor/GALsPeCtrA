"""
Diagonal weighting schemes for the spectral PCA.

The architectural constraint
----------------------------
Reconstruction must stay *linear in flux*, because a composite stellar population is
built as the mass-weighted sum of simple populations:

    L_gal(λ) = Σ_i m_i · S(λ; age_i, Z_i)

If the SSP spectra are transformed by anything that depends on the individual
spectrum — an L2 norm, a value at a reference wavelength, a logarithm — that identity
no longer holds, and the failure is silent: each SSP still reconstructs beautifully,
and only the *galaxies* come out wrong. That is the hardest class of bug to find,
because every unit-level check passes.

A weight that is a fixed function of wavelength alone is safe. It rescales the
columns of the design matrix, changing which wavelengths the PCA works hardest to
reproduce, and it is divided back out of the basis vectors afterwards
(`basis.SpectralBasis.fit`), so the stored basis lives in flux units and the
composition identity survives exactly.

So: **weights are (N_wave,), never (N_ssp,) and never (N_ssp, 1).** Every function
here enforces that shape, and `build_weights` refuses the two per-spectrum schemes
that `pca/preprocess.py` still offers, by name, with an explanation.

Relation to the existing code
-----------------------------
`preprocess.normalize_seds(method="std")` divides each wavelength bin by its standard
deviation across the SSP grid. That *is* a diagonal weighting — it was simply never
described as one. It appears here as `inverse_std`, and reproduces the committed
basis exactly. `uniform` is the natural null hypothesis it should be compared against,
which has never been done.
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
    """Validate a weight vector. Returns it as a contiguous float64 array.

    The shape check is the load-bearing one: it makes a per-spectrum weight — the
    thing that breaks the architecture — a structural impossibility rather than a
    convention someone has to remember.
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

    Parameters
    ----------
    scheme : str — one of SCHEMES.
        'uniform'      w = 1. The null hypothesis: fit absolute flux errors.
        'inverse_std'  w = 1/std_over_SSPs. The incumbent (preprocess method='std').
                       Equalises how much each bin *varies* across the library, so
                       the PCA spends components where the library disagrees with
                       itself rather than where it is bright.
        'inverse_rms'  w = 1/sqrt(mean(X²)). Equalises *amplitude*, so a fixed
                       fractional error costs the same everywhere. This is the
                       scheme that should help the faint blue continuum of red
                       galaxies, which is where the measured bias lives.
        'inverse_mean' w = 1/mean(X). As inverse_rms but using the mean; kept
                       because it is what a reader assumes 'fractional' means.
        'piecewise'    a `base` scheme multiplied by per-region factors, to buy
                       accuracy in a named wavelength range at a stated cost
                       elsewhere.
    X : (N_ssp, N_wave) — the SSP library the weights are derived from.
    wave : (N_wave,) — wavelengths in Å, needed by 'piecewise'.
    base : str — base scheme for 'piecewise'.
    regions : list of {'min','max','factor'} — required for 'piecewise'.
    normalise : bool — rescale so the geometric mean weight is 1. This is cosmetic;
        an overall scalar on w cancels exactly in fit → reconstruct, but it keeps
        the reported weight curves comparable between schemes.

    Returns
    -------
    w : (N_wave,) float64
    meta : dict — provenance for the basis file
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
        # A dead wavelength bin (identically zero across the whole library) is a
        # property of the range, not of the weighting. Say so precisely.
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
