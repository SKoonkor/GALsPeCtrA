"""
A faithful port of the quadrature L-GALAXIES uses to build its photometric tables.

Why this exists
---------------
`SpecPhotTables/PhotTables/` and `SpecPhotTables/FullSEDs/` are two products in the
same L-GALAXIES distribution that disagree, in *g−r*, by ~0.005–0.009 mag after the
filter convolution convention is accounted for (`documents/filter_convention.md`).
That residual was attributed to a BaSeL-vs-STELIB library difference, and that
explanation was retired when the residual turned out to be **larger outside** the
empirical library's wavelength range than inside it.

There is no library difference. `setup_Spec_LumTables_onthefly()` in
`code/model_spectro_photometric.c:179` **writes the PhotTables**, and its own comment
says it needs only "the SEDs in SpecPhotDir/FullSEDs/ … and the filter curves in
SpecPhotDir/Filters/" — the same two inputs GALsPeCtrA integrates. So the two products
are two implementations of one integral, and the disagreement is quadrature.

This module reproduces the C implementation exactly, defects included, so the
difference can be measured instead of argued about. It is a **diagnostic**, not a
recommendation: see `documents/lgalaxies_quadrature.md`.

What the C code does, and where
-------------------------------
Reading the chain `setup_Spec_LumTables_onthefly` → `create_grid` → `interpolate` →
`integrate` → `get_AbsAB_magnitude`, three things differ from a textbook synthetic
magnitude. Each is reproduced here behind its own switch, so the decomposition can
attribute the residual rather than merely close it.

1. **The in-band average is taken in dλ, not dν.**
   `model_spectro_photometric.c:284` forms ``f_ν · T`` and `:294-296` integrates that
   against ``T``, both over a wavelength grid. The weight is therefore ∝ ``T``, where
   energy weighting is ∝ ``T/λ²`` and photon counting ∝ ``T/λ``. It sits one power of
   λ beyond photon counting — which is why photon came closer to the tables than
   energy did, and still fell short.

2. **`integrate()` never multiplies by the step size** (`model_misc.c:3955`). It
   returns Simpson-*weighted sums of values*: ``I = (f₀+f_N)/3 + (2/3)Σf_odd +
   (4/3)Σf_even``. Two consequences. The grid is BC03's native wavelength grid, whose
   spacing runs from 4 to 100 Å, so without a Δλ this is not a wavelength integral at
   all — it is a point-count-weighted mean, and regions where the library happens to
   be finely sampled are over-weighted. And the weights are swapped relative to
   Simpson's rule, which puts 4/3 on odd and 2/3 on even points. A constant step size
   would cancel between numerator and denominator; a varying one does not, and neither
   does the weight pattern unless f_ν is flat across the band.

3. **`interpolate()` does not interpolate** (`..._onthefly_misc.c:186`). It locates the
   bracketing index with a Numerical-Recipes `locate()` and takes
   ``FluxOnGrid[i] = flux[kk]`` — the neighbouring sample, not a weighted blend. The
   filter curve is therefore stepped onto the SSP grid.

All three are worst where the SSP grid is coarsest, which is outside 3200–9500 Å. That
is precisely the observation that falsified the library hypothesis.

Faithfulness
------------
`_locate` reproduces the Numerical Recipes routine as called — with `n = nlambda-1`
passed against a 0-indexed C array, index arithmetic included — because the point is to
match the C code, not to fix it. Do not tidy these functions; if they stop matching the
tables the diagnostic is worthless.
"""

import numpy as np

from .synthetic import C_ANG_S, flam_to_fnu

__all__ = [
    "MEASURES",
    "lgal_integrate",
    "lgal_integration_weights",
    "lgal_sample_on_grid",
    "lgal_band_matrix",
    "lgal_magnitudes_from_matrix",
    "compute_lgal_magnitudes",
]

#: Which measure the in-band average is taken in.
#:
#:   ``"dnu"``       ∫f_ν T dν   / ∫T dν     — energy weighting
#:   ``"dnu_over_nu"`` ∫f_ν T dν/ν / ∫T dν/ν — photon counting (GALsPeCtrA's default)
#:   ``"dlambda"``   ∫f_ν T dλ   / ∫T dλ     — what the C code does
MEASURES = ("dnu", "dnu_over_nu", "dlambda")

#: 4π(10 pc)² · 3631 Jy in cgs, from `get_AbsAB_magnitude` (`..._onthefly_misc.c:41-47`).
_DISTANCE_CM = 10.0 * 3.08568025e18
_ZEROPOINT = -2.5 * np.log10(4.0 * np.pi * _DISTANCE_CM * _DISTANCE_CM * 3631.0 * 1.0e-23)


def _locate(xx, n, x):
    """Numerical Recipes `locate`, as called from `interpolate` (`model_misc.c:3923`).

    Ported literally, including the caller's mixing of NR's 1-based indexing with a
    0-based C array — `interpolate` passes `nlambda-1` as `n`. The resulting index is
    the left neighbour under that convention, which is what makes the filter sampling
    a step rather than a blend.
    """
    jl, ju = 0, n + 1
    ascnd = xx[n] >= xx[1]
    while ju - jl > 1:
        jm = (ju + jl) >> 1
        if (x >= xx[jm]) == ascnd:
            jl = jm
        else:
            ju = jm
    if x == xx[1]:
        return 1
    if x == xx[n]:
        return n - 1
    return jl


def lgal_sample_on_grid(grid, lam, flux):
    """`interpolate()` from `..._onthefly_misc.c:186` — a step sample, not an interpolation.

    Values outside the curve's own range become 0, matching the C ("outside filter
    range, transmission is 0").
    """
    grid = np.asarray(grid, dtype=float)
    lam = np.asarray(lam, dtype=float)
    flux = np.asarray(flux, dtype=float)
    n_lam = lam.size
    out = np.zeros(grid.size, dtype=float)

    for i, x in enumerate(grid):
        if x < lam[0] or x > lam[-1]:
            continue
        nn = _locate(lam, n_lam - 1, x)
        # kk = min(max(nn - (m-1)//2, 1), nlambda + 1 - m) with m = 2, so (m-1)//2 = 0
        kk = min(max(nn, 1), n_lam - 1)
        out[i] = flux[kk]
    return out


def lgal_integration_weights(n, x=None):
    """The coefficient vector `integrate()` applies (`model_misc.c:3955`).

    Returned as a vector rather than folded into a sum, because the quadrature is
    linear in the values — which is what lets `lgal_band_matrix` evaluate a whole SSP
    library with one matrix product instead of 53,000 Python loops.

    With `x` given, returns trapezoidal weights over that grid instead, restoring the
    step size the C omits. The loop bounds and the 2/3 ÷ 4/3 assignment below are
    copied from the C, including the swap relative to Simpson's rule (which puts 4/3
    on odd points, not even). **Do not correct them** — this exists to be wrong in the
    same way the tables are.
    """
    if n < 2:
        return np.zeros(max(n, 0), dtype=float)

    if x is not None:
        x = np.asarray(x, dtype=float)
        w = np.zeros(n, dtype=float)
        dx = np.diff(x)
        w[:-1] += dx / 2.0
        w[1:] += dx / 2.0
        return w

    w = np.zeros(n, dtype=float)
    for i in range(max(0, n // 2 - 2)):
        w[2 * i + 1] += 2.0 / 3.0
    for i in range(max(0, n // 2 - 1)):
        w[2 * i] += 4.0 / 3.0
    w[0] += 1.0 / 3.0
    w[n - 1] += 1.0 / 3.0
    return w


def lgal_integrate(values, dx=None):
    """`integrate()` from `model_misc.c:3955`: Simpson-like weights, **no step size**.

    Pass `dx` (the grid, not a spacing) to restore a trapezoidal integral instead.
    """
    f = np.asarray(values, dtype=float)
    return float(lgal_integration_weights(f.size, dx) @ f)


def _measure_weight(lam, measure):
    """The λ-space factor that converts ∫·dλ into the requested measure.

    ∫X dν ∝ ∫X λ⁻² dλ and ∫X dν/ν ∝ ∫X λ⁻¹ dλ, so a single multiplicative factor turns
    one into another and the three conventions differ only by a power of λ.
    """
    if measure == "dlambda":
        return np.ones_like(lam)
    if measure == "dnu_over_nu":
        return 1.0 / lam
    if measure == "dnu":
        return 1.0 / lam ** 2
    raise ValueError(f"unknown measure {measure!r}; choose from {MEASURES}")


def lgal_band_matrix(wave_ang, filters, *,
                     measure="dlambda",
                     use_step_size=False,
                     step_filter=True,
                     redshift=0.0):
    """The same quadrature as a matrix: `mean_f_nu = (f_lam @ P.T) / denom`.

    L-GALAXIES' integral is still *linear in f_λ* — the defects change the weights, not
    the linearity — so a whole SSP library goes through in one matrix product. This is
    the same trick `pca.metrics.BandOperator` uses for the production conventions, and
    it is what makes the 221 × 6 × 40 decomposition tractable.

    Returns
    -------
    names : list[str]        bands that overlap the grid, in input order
    P     : (n_band, n_wave) rows of the numerator functional
    denom : (n_band,)        the ∫T term each row is divided by
    """
    wave_ang = np.asarray(wave_ang, dtype=float)
    obs = (1.0 + redshift) * wave_ang
    lam2_over_c = (1.0 + redshift) * wave_ang ** 2 / C_ANG_S   # f_λ → f_ν, as in the C

    names, rows, denoms = [], [], []
    for name, (wave_f, trans_f) in filters.items():
        lo, hi = float(wave_f[0]), float(wave_f[-1])
        sel = (obs >= min(lo, hi)) & (obs <= max(lo, hi))
        if sel.sum() < 2:
            continue

        grid = obs[sel]
        if step_filter:
            trans_on_grid = lgal_sample_on_grid(grid, wave_f, trans_f)
        else:
            trans_on_grid = np.interp(grid, wave_f, trans_f, left=0.0, right=0.0)

        weighted_trans = trans_on_grid * _measure_weight(grid, measure)
        w = lgal_integration_weights(grid.size, grid if use_step_size else None)

        den = float(w @ weighted_trans)
        if den <= 0:
            continue

        row = np.zeros(wave_ang.size, dtype=float)
        row[sel] = w * weighted_trans * lam2_over_c[sel]

        names.append(name)
        rows.append(row)
        denoms.append(den)

    P = np.asarray(rows, dtype=float) if rows else np.zeros((0, wave_ang.size))
    return names, P, np.asarray(denoms, dtype=float)


def lgal_magnitudes_from_matrix(f_lam, P, denom):
    """AB magnitudes from `lgal_band_matrix`, for one spectrum or a stack of them."""
    f_lam = np.atleast_2d(np.asarray(f_lam, dtype=float))
    mean_f_nu = (f_lam @ P.T) / denom
    with np.errstate(divide="ignore", invalid="ignore"):
        mags = -2.5 * np.log10(mean_f_nu) - _ZEROPOINT
    return np.where(np.isfinite(mags), mags, np.nan)


def compute_lgal_magnitudes(wave_ang, f_lam, filters, *,
                            measure="dlambda",
                            use_step_size=False,
                            step_filter=True,
                            redshift=0.0):
    """AB magnitudes through L-GALAXIES' own quadrature.

    Parameters
    ----------
    wave_ang, f_lam : (N,) — the SSP on its **native** grid. Do not resample first;
        the grid spacing is part of what is being measured.
    filters : dict {name: (wave_AA, transmission)}
    measure : one of `MEASURES`. `"dlambda"` is the C behaviour.
    use_step_size : False reproduces `integrate()`'s missing Δλ. True substitutes a
        trapezoidal integral, isolating that defect.
    step_filter : True reproduces `interpolate()`'s step sampling. False uses
        `np.interp`, as production does.
    redshift : carried for completeness; the tables in use are z = 0.

    Returns
    -------
    dict {band: AB magnitude}, np.nan where the filter does not overlap the grid.

    The defect switches exist so `scripts/photometry_convention_check.py` can turn them
    on one at a time and attribute the residual, rather than only closing it.
    """
    wave_ang = np.asarray(wave_ang, dtype=float)
    f_lam = np.asarray(f_lam, dtype=float)

    # `read_InputSSP_spectra` (`..._onthefly_initialize.c:194`) stores f_ν, scaled by
    # 1e11 for the mass unit; the scale is a constant and folds into the zero point.
    f_nu = flam_to_fnu(wave_ang, f_lam) * (1.0 + redshift)

    mags = {}
    for name, (wave_f, trans_f) in filters.items():
        # create_grid: the filter's span, the spectrum's binning.
        lo, hi = float(wave_f[0]), float(wave_f[-1])
        obs = (1.0 + redshift) * wave_ang
        sel = (obs >= min(lo, hi)) & (obs <= max(lo, hi))
        if sel.sum() < 2:
            mags[name] = np.nan
            continue

        grid = obs[sel]
        sed_on_grid = f_nu[sel]

        if step_filter:
            trans_on_grid = lgal_sample_on_grid(grid, wave_f, trans_f)
        else:
            trans_on_grid = np.interp(grid, wave_f, trans_f, left=0.0, right=0.0)

        # The measure enters as a power of λ on both numerator and denominator, exactly
        # as the convention does in `synthetic.band_weight`.
        w = _measure_weight(grid, measure)
        weighted_trans = trans_on_grid * w

        dx = grid if use_step_size else None
        num = lgal_integrate(sed_on_grid * weighted_trans, dx)
        den = lgal_integrate(weighted_trans, dx)

        if den <= 0 or num <= 0:
            mags[name] = np.nan
            continue

        mags[name] = -2.5 * (np.log10(num) - np.log10(den)) - _ZEROPOINT

    return mags
