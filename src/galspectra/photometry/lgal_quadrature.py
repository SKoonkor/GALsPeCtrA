"""
A faithful port of the quadrature L-GALAXIES uses to build its photometric tables.

`SpecPhotTables/PhotTables/` and `SpecPhotTables/FullSEDs/` disagree by
~0.005-0.009 mag in g-r after the filter convention is accounted for
(`documents/filter_convention.md`) — not a library difference (BaSeL vs
STELIB was tested and falsified: the residual is larger *outside* STELIB's
range, not inside it), but quadrature: both come from
`setup_Spec_LumTables_onthefly()` (`code/model_spectro_photometric.c:179`),
whose own comment says it needs only FullSEDs and Filters — the same two
inputs GALsPeCtrA integrates.

This module reproduces that C implementation exactly, three defects
included, as a **diagnostic** (not a recommendation — see
`documents/lgalaxies_quadrature.md`), each behind its own switch so the
residual can be attributed rather than only closed:

1. **In-band average taken in dλ, not dν** (`model_spectro_photometric.c:284,
   294-296`): weight ∝ T, one power of λ beyond photon counting.
2. **`integrate()` never multiplies by the step size** (`model_misc.c:3955`):
   a point-count-weighted mean on BC03's 4-100 Å uneven grid, with Simpson
   weights swapped (4/3 on odd points, not even).
3. **`interpolate()` doesn't interpolate** (`..._onthefly_misc.c:186`): steps
   the filter onto the nearest SSP grid point via Numerical Recipes `locate()`
   rather than blending.

All three are worst outside 3200-9500 Å, which is what falsified the library
hypothesis. **Do not tidy `_locate` or the Simpson weights in
`lgal_integration_weights`** — they exist to match the C bug-for-bug; fixing
them makes the diagnostic worthless.
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
    """Numerical Recipes `locate`, as called from `interpolate` (model_misc.c:3923).

    Ported literally, mixed 1-based/0-based indexing included — this is what
    makes the filter sampling a step, not a blend.
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
    """`interpolate()` (..._onthefly_misc.c:186) — a step sample, not an interpolation.

    Values outside the curve's own range become 0, matching the C.
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
    """The coefficient vector `integrate()` applies (model_misc.c:3955).

    A vector, not a folded sum, because linearity lets `lgal_band_matrix`
    evaluate a whole library with one matrix product. With `x` given, returns
    trapezoidal weights instead (restores the step size the C omits).

    The loop bounds and 2/3 vs 4/3 assignment below are copied from the C,
    including the swap relative to Simpson's rule. Do not correct them —
    this exists to be wrong in the same way the tables are.
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
    """`integrate()` (model_misc.c:3955): Simpson-like weights, no step size.

    Pass `dx` (the grid, not a spacing) to restore a trapezoidal integral.
    """
    f = np.asarray(values, dtype=float)
    return float(lgal_integration_weights(f.size, dx) @ f)


def _measure_weight(lam, measure):
    """λ-space factor converting ∫·dλ into the requested measure.

    ∫X dν ∝ ∫X λ⁻² dλ, ∫X dν/ν ∝ ∫X λ⁻¹ dλ — the three conventions differ
    only by a power of λ.
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
    """The same quadrature as a matrix: mean_f_nu = (f_lam @ P.T) / denom.

    The defects change the weights, not the linearity in f_λ, so a whole SSP
    library goes through one matrix product (as `pca.metrics.BandOperator`
    does for the production conventions) — what makes the 221×6×40
    decomposition tractable.

    Returns names (bands that overlap, in input order), P (n_band, n_wave),
    denom (n_band,).
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

    wave_ang, f_lam : (N,) the SSP on its native grid — do not resample first,
        the grid spacing is part of what is being measured.
    measure : one of MEASURES; 'dlambda' is the C behaviour.
    use_step_size : False reproduces integrate()'s missing Δλ.
    step_filter : True reproduces interpolate()'s step sampling.

    Returns {band: AB magnitude}, np.nan where no overlap. The defect
    switches let `scripts/photometry_convention_check.py` attribute the
    residual to each one rather than only closing it.
    """
    wave_ang = np.asarray(wave_ang, dtype=float)
    f_lam = np.asarray(f_lam, dtype=float)

    # read_InputSSP_spectra (..._onthefly_initialize.c:194) scales f_ν by 1e11;
    # a constant, folds into the zero point.
    f_nu = flam_to_fnu(wave_ang, f_lam) * (1.0 + redshift)

    mags = {}
    for name, (wave_f, trans_f) in filters.items():
        lo, hi = float(wave_f[0]), float(wave_f[-1])  # create_grid: filter span
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

        # measure enters as a power of λ on both num/denom, like synthetic.band_weight
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
