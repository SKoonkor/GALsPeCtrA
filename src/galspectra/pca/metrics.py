"""
Error metrics for spectral compression: photometry, D4000, and storage cost.

Everything here is built around one observation: **AB band flux is a linear
functional of f_λ.** Writing

    m_AB = -2.5 log10( ∫ f_ν T dν / ∫ T dν ) - 48.6

the numerator is linear in f_ν, and f_ν = f_λ λ²/c is a diagonal map from f_λ, so the
whole chain from spectrum to band flux is a single fixed matrix P:

    band_flux = flux_lambda @ P.T

That has three consequences worth the effort of exploiting:

* **Speed.** 1,200 SSPs or 10,000 galaxies × 60 bands becomes one matrix product,
  not a Python loop over `compute_ab_magnitudes`.
* **Consistency.** Truth and reconstruction go through the identical operator, so a
  difference in magnitudes is a difference in spectra and nothing else.
* **Composition.** Because P is linear and reconstruction is linear, the band flux of
  a composite population is the mass-weighted sum of the band fluxes of its SSPs.
  Galaxy-level photometry can be evaluated without ever materialising a galaxy
  spectrum.

`BandOperator` reproduces `galspectra.photometry.synthetic.compute_ab_magnitudes`
exactly — same trapezoid-in-frequency convention on the same grid — so the fast path
and the production path cannot disagree. `tests/test_pca_basis_linearity.py` pins that.

D4000 follows the same pattern. Both the direct spectral definition and Renard et
al. (2022)'s narrow-band reconstruction are ratios of linear functionals, so each is
a small matrix and both are evaluated the same way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = [
    "C_ANG_S",
    "CONVENTIONS",
    "DEFAULT_CONVENTION",
    "BandOperator",
    "D4000_DEFINITIONS",
    "d4000_operator",
    "d4000_narrowband_operator",
    "mag_difference",
    "summarise_mmag",
    "region_flux_errors",
    "WAVELENGTH_REGIONS",
    "bytes_for",
]

from galspectra.photometry.synthetic import (  # noqa: E402
    CONVENTIONS,
    DEFAULT_CONVENTION,
    band_weight,
)

C_ANG_S = 2.99792458e18  # speed of light, Å/s — matches photometry/synthetic.py


# ─────────────────────────────────────────────────────────────────────────────
# Photometry as a matrix
# ─────────────────────────────────────────────────────────────────────────────

def _trapezoid_weights(x):
    """Weights v such that v @ g == np.trapezoid(g, x) for strictly monotonic x."""
    x = np.asarray(x, dtype=float)
    v = np.empty_like(x)
    v[0] = 0.5 * (x[1] - x[0])
    v[-1] = 0.5 * (x[-1] - x[-2])
    v[1:-1] = 0.5 * (x[2:] - x[:-2])
    return v


class BandOperator:
    """Turns f_λ on a fixed grid into mean in-band f_ν, for a set of filters.

    Parameters
    ----------
    wave : (N_wave,) — the SED grid, Å, strictly increasing
    filters : dict {name: (wave_filter_AA, transmission)}
    min_coverage : float
        Reject a band whose transmission-weighted overlap with the SED grid falls
        below this fraction of its total. A partially-covered band produces a
        magnitude that looks plausible and is wrong, which is worse than a NaN.
    convention : 'photon' (default) or 'energy'
        The filter convolution convention, from
        `galspectra.photometry.synthetic`. Both are linear in f_λ, so the
        operator stays a matrix either way — only the weight changes.

        **This default is imported, never restated.** It has to be the same
        object `compute_ab_magnitudes` uses: if the fast matrix path and the
        production path could disagree about the convention, every number this
        harness produces would silently stop being comparable with the validated
        pipeline, and nothing would fail. `tests/test_pca_basis_linearity.py`
        pins that the two agree.

    Attributes
    ----------
    names : list[str] — bands that passed the coverage test, in input order
    P : (N_band, N_wave) — band_flux = flux @ P.T
    coverage : dict {name: fraction} — for every input band, including rejects
    """

    def __init__(self, wave, filters, min_coverage=0.99,
                 convention=DEFAULT_CONVENTION):
        wave = np.asarray(wave, dtype=float)
        if not np.all(np.diff(wave) > 0):
            raise ValueError("wave must be strictly increasing")

        # Work in frequency-ascending order, matching compute_ab_magnitudes, which
        # sorts by ν before integrating. On a λ-ascending grid that is a reversal.
        nu = C_ANG_S / wave
        order = np.argsort(nu)
        v_nu_sorted = _trapezoid_weights(nu[order])
        v_nu = np.empty_like(v_nu_sorted)
        v_nu[order] = v_nu_sorted

        lam2_over_c = wave ** 2 / C_ANG_S       # f_λ -> f_ν

        rows, names, coverage = [], [], {}
        for name, (wf, tf) in filters.items():
            wf = np.asarray(wf, dtype=float)
            tf = np.asarray(tf, dtype=float)
            if not np.all(np.diff(wf) > 0):
                o = np.argsort(wf)
                wf, tf = wf[o], tf[o]

            # Coverage: how much of ∫T dλ falls inside the SED grid's span.
            total = np.trapezoid(tf, wf)
            inside = (wf >= wave[0]) & (wf <= wave[-1])
            frac = 0.0
            if total > 0 and inside.sum() >= 2:
                frac = float(np.trapezoid(tf[inside], wf[inside]) / total)
            coverage[name] = frac
            if frac < min_coverage:
                continue

            T_on_sed = np.interp(wave, wf, tf, left=0.0, right=0.0)
            # The convention enters only through the weight function; the
            # quadrature is unchanged, matching production exactly.
            w_on_sed = band_weight(nu, T_on_sed, convention)
            denom = float(np.sum(w_on_sed * v_nu))
            if denom <= 0:
                coverage[name] = 0.0
                continue

            rows.append(lam2_over_c * w_on_sed * v_nu / denom)
            names.append(name)

        if not rows:
            raise ValueError(
                "no filter met the coverage requirement on this wavelength grid "
                f"({wave[0]:.0f}-{wave[-1]:.0f} Å)"
            )

        self.wave = wave
        self.names = names
        self.P = np.asarray(rows)
        self.coverage = coverage
        self.min_coverage = float(min_coverage)
        self.convention = convention
        self.rejected = [n for n, f in coverage.items() if f < min_coverage]

    def __len__(self):
        return len(self.names)

    def band_flux(self, flux):
        """Mean in-band f_ν. `flux` is (N_wave,) or (N, N_wave) in f_λ."""
        flux = np.asarray(flux, dtype=float)
        return flux @ self.P.T

    def magnitudes(self, flux, zero_point=-48.6):
        """AB magnitudes. Non-positive band fluxes become NaN, as in production."""
        fnu = self.band_flux(flux)
        with np.errstate(divide="ignore", invalid="ignore"):
            mags = -2.5 * np.log10(np.where(fnu > 0, fnu, np.nan)) + zero_point
        return mags

    def index(self, name):
        return self.names.index(name)

    def subset(self, names):
        """A new operator restricted to `names`, preserving their order."""
        idx = [self.index(n) for n in names]
        new = object.__new__(BandOperator)
        new.wave = self.wave
        new.names = list(names)
        new.P = self.P[idx]
        new.coverage = {n: self.coverage[n] for n in names}
        new.min_coverage = self.min_coverage
        new.convention = self.convention
        new.rejected = []
        return new


# ─────────────────────────────────────────────────────────────────────────────
# D4000
# ─────────────────────────────────────────────────────────────────────────────

# Rest-frame window edges in Å.
D4000_DEFINITIONS = {
    # Balogh et al. (1999) narrow definition — less sensitive to reddening
    "D4000_n": {"blue": (3850.0, 3950.0), "red": (4000.0, 4100.0)},
    # Bruzual (1983) original wide definition
    "D4000_w": {"blue": (3750.0, 3950.0), "red": (4050.0, 4250.0)},
}


def _window_mean_fnu_row(wave, lo, hi):
    """Row r with r @ f_λ == mean f_ν over [lo, hi].

    ⟨F_ν⟩ = ∫_lo^hi f_ν dλ / (hi - lo), with f_ν = f_λ λ²/c.  The integral is taken
    over the SED grid with trapezoid weights restricted to the window, so a window
    narrower than a few bins degrades gracefully rather than silently returning the
    nearest single sample.
    """
    from galspectra.pca.resample import cumulative_integral_matrix

    wave = np.asarray(wave, dtype=float)
    if lo < wave[0] or hi > wave[-1]:
        raise ValueError(f"window {lo}-{hi} Å outside grid {wave[0]:.0f}-{wave[-1]:.0f} Å")
    # Integrate f_ν = f_λ · λ²/c, so integrate the *scaled* node values.
    T = cumulative_integral_matrix(wave, [lo, hi])
    row = (T[1] - T[0]) * (wave ** 2 / C_ANG_S) / (hi - lo)
    return row


def d4000_operator(wave, definition="D4000_n"):
    """(2, N_wave) matrix whose rows give ⟨F_ν⟩ in the blue and red windows.

    D4000 = (row_red @ f) / (row_blue @ f)  — Renard et al. (2022) Eq. 1.
    """
    if definition not in D4000_DEFINITIONS:
        raise ValueError(f"unknown D4000 definition '{definition}'. "
                         f"Choose from {list(D4000_DEFINITIONS)}")
    win = D4000_DEFINITIONS[definition]
    return np.vstack([
        _window_mean_fnu_row(wave, *win["blue"]),
        _window_mean_fnu_row(wave, *win["red"]),
    ])


def d4000_from_operator(op, flux):
    """D4000 = red/blue from a (2, N_wave) operator."""
    vals = np.asarray(flux, dtype=float) @ op.T
    blue, red = vals[..., 0], vals[..., 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(blue > 0, red / blue, np.nan)


def d4000_narrowband_operator(wave, filters, redshift, definition="D4000_n",
                              sigma=None, min_r=1e-3):
    """Renard et al. (2022) narrow-band D4000, as a (2, N_wave) operator.

    Implements Eqs. 3–5 for a source at `redshift`, evaluated against a *rest-frame*
    spectrum. The redshift enters by blueshifting each band's response onto the rest
    frame (λ_rest = λ_obs / (1+z)); the (1+z) and luminosity-distance factors that
    convert rest-frame f_ν to observed f_ν are a single multiplicative constant and
    cancel in the red/blue ratio, so no cosmology is needed.

        Δλ_i = ∫ R_i dλ                                              (Eq. 5)
        r_i  = ∫_window R_i dλ / ∫ R_i dλ                            (Eq. 4)
        ⟨F_ν⟩ = Σ_i Δλ_i r_i² σ_i⁻² F_ν,i / Σ_i Δλ_i r_i² σ_i⁻²      (Eq. 3)

    Parameters
    ----------
    wave : (N_wave,) — rest-frame SED grid, Å
    filters : dict {name: (wave_obs_AA, transmission)} — observed-frame responses
    redshift : float
    sigma : dict {name: float} or None
        Per-band photometric uncertainties. `None` gives every band equal weight,
        which is the right choice for noiseless model spectra: the σ⁻² factors then
        cancel identically between numerator and denominator. Supplying real PAUS
        uncertainties changes the weighting and is the correct thing to do when
        comparing to data, so the argument exists.
    min_r : float
        Bands contributing less than this fraction of their response to the window
        are dropped. Without it, a band with r ~ 1e-8 still enters the sum with
        weight r² and only adds noise.

    Returns
    -------
    (2, N_wave) operator, rows = blue then red, or raises if either window has no
    contributing band (which is the honest outcome when the windows fall outside the
    instrument's wavelength range at this redshift).
    """
    win = D4000_DEFINITIONS[definition]
    wave = np.asarray(wave, dtype=float)
    zp1 = 1.0 + float(redshift)

    rest_filters = {}
    for name, (wf, tf) in filters.items():
        rest_filters[name] = (np.asarray(wf, dtype=float) / zp1, np.asarray(tf, dtype=float))

    # Band fluxes come from the same operator production photometry uses.
    band_op = BandOperator(wave, rest_filters, min_coverage=0.99)

    rows = []
    detail = {}
    for side in ("blue", "red"):
        lo, hi = win[side]
        weights, used = [], []
        for name in band_op.names:
            wf, tf = rest_filters[name]
            dlam = float(np.trapezoid(tf, wf))                       # Eq. 5
            if dlam <= 0:
                continue
            inside = (wf >= lo) & (wf <= hi)
            if inside.sum() < 2:
                continue
            r = float(np.trapezoid(tf[inside], wf[inside]) / dlam)   # Eq. 4
            if r < min_r:
                continue
            s = 1.0 if sigma is None else float(sigma.get(name, 1.0))
            weights.append(dlam * r ** 2 / s ** 2)                   # Eq. 3 numerator weight
            used.append(name)

        if not used:
            raise ValueError(
                f"no band contributes to the {side} D4000 window "
                f"({lo:.0f}-{hi:.0f} Å rest) at z={redshift:g}. "
                "The 4000 Å break is outside this filter set at this redshift."
            )
        wgt = np.asarray(weights)
        wgt = wgt / wgt.sum()
        idx = [band_op.index(n) for n in used]
        rows.append(wgt @ band_op.P[idx])
        detail[side] = {"bands": used, "weights": wgt.tolist()}

    op = np.vstack(rows)
    op_detail = detail
    return op, op_detail


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def mag_difference(truth_flux, recon_flux, band_op):
    """Δm = m_recon − m_truth in **millimagnitudes**, shape (..., N_band).

    Computed from the flux ratio rather than by differencing two magnitudes, so it
    is well defined wherever both band fluxes are positive and does not lose
    precision when the two magnitudes are nearly equal.
    """
    f_t = band_op.band_flux(truth_flux)
    f_r = band_op.band_flux(recon_flux)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((f_t > 0) & (f_r > 0), f_r / f_t, np.nan)
        return -2.5 * np.log10(ratio) * 1000.0


def summarise_mmag(dmag, axis=0):
    """Robust and non-robust summaries of a Δmag array, in mmag.

    Both are reported because they answer different questions: the median absolute
    error is what a typical galaxy suffers, and the 99th percentile is what decides
    whether a catalogue has a tail that will show up in a luminosity function.
    """
    d = np.asarray(dmag, dtype=float)
    absd = np.abs(d)
    with np.errstate(invalid="ignore"):
        return {
            "bias": np.nanmedian(d, axis=axis),
            "mean": np.nanmean(d, axis=axis),
            "mad": np.nanmedian(absd, axis=axis),
            "rms": np.sqrt(np.nanmean(d ** 2, axis=axis)),
            "p68": np.nanpercentile(absd, 68.0, axis=axis),
            "p95": np.nanpercentile(absd, 95.0, axis=axis),
            "p99": np.nanpercentile(absd, 99.0, axis=axis),
            "max": np.nanmax(absd, axis=axis),
            "n_valid": np.sum(np.isfinite(d), axis=axis),
        }


# Named rest-frame regions for the by-wavelength breakdown. The boundaries are
# chosen where the *physics* changes, not on round numbers: the Lyman and Balmer
# limits, the 4000 Å break, and the onset of the Rayleigh-Jeans tail.
WAVELENGTH_REGIONS = {
    "far-UV (<1216 Å)": (0.0, 1216.0),
    "near-UV (1216-3646 Å)": (1216.0, 3646.0),
    "blue optical (3646-5500 Å)": (3646.0, 5500.0),
    "red optical (5500-9000 Å)": (5500.0, 9000.0),
    "near-IR (9000-24000 Å)": (9000.0, 24000.0),
}


def region_flux_errors(wave, truth, recon, regions=None):
    """Median and 99th-percentile |Δf/f| within each named wavelength region.

    Fractional flux error, not magnitude error, because these regions are not
    passbands — no instrument integrates over "near-UV" — and a fractional flux
    error is the quantity that propagates into whatever passband a reader cares
    about.
    """
    regions = regions or WAVELENGTH_REGIONS
    wave = np.asarray(wave, dtype=float)
    truth = np.atleast_2d(np.asarray(truth, dtype=float))
    recon = np.atleast_2d(np.asarray(recon, dtype=float))

    out = {}
    for label, (lo, hi) in regions.items():
        sel = (wave >= lo) & (wave < hi)
        if sel.sum() == 0:
            continue
        t = truth[:, sel]
        r = recon[:, sel]
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(t > 0, np.abs(r - t) / t, np.nan)
        out[label] = {
            "n_bins": int(sel.sum()),
            "median_frac": float(np.nanmedian(frac)),
            "p99_frac": float(np.nanpercentile(frac, 99.0)),
            "max_frac": float(np.nanmax(frac)),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Storage accounting
# ─────────────────────────────────────────────────────────────────────────────

def bytes_for(n_values, dtype=np.float32):
    """Bytes to store `n_values` numbers per galaxy."""
    return int(n_values * np.dtype(dtype).itemsize)
