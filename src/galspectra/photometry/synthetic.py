"""
Synthetic photometry.

Convolves an SED (flux vs wavelength) with filter transmission curves to
produce AB magnitudes:

    m_AB = -2.5 log10( ∫ f_ν(ν) T(ν) dν / ∫ T(ν) dν ) - 48.6

f_ν must be a flux (erg/s/Hz/cm²) at the detector, not a luminosity. BC03/FSPS
SEDs are erg/s/Å per Msun formed, so scale before calling compute_ab_magnitudes:

    sed_at_10pc = sed_per_msun * M_star_msun / (4 * pi * (10 pc in cm)^2)

Omitting this gives magnitudes ~70 mag too bright.
"""

import numpy as np

C_ANG_S = 2.99792458e18  # speed of light in Å/s

# ─────────────────────────────────────────────────────────────────────────────
# Filter convolution convention: energy (⟨f_ν⟩ = ∫f_ν T dν / ∫T dν) weights by
# transmission; photon (⟨f_ν⟩ = ∫f_ν T dν/ν / ∫T dν/ν) weights by 1/(hν), as a
# CCD does. They agree only for a source flat in f_ν; the difference grows for
# red spectra, hence old stellar populations.
#
# `photon` is the default (adopted 7 Aug 2026, chosen by measurement against
# L-GALAXIES' own PhotTables — see documents/filter_convention.md). `energy`
# is kept, not deprecated: every product before that date used it, and
# data/pre_photon_energy_convention/ preserves them for comparison.
#
# `lgal_native` (see lgal_quadrature.py) replaces the quadrature with
# L-GALAXIES' own, defects included — a diagnostic only, never for science.
CONVENTIONS = ("energy", "photon")
DEFAULT_CONVENTION = "photon"

#: Not in CONVENTIONS: it replaces the quadrature rather than reweighting it,
#: and must stay out of anything that iterates the legitimate choices.
DIAGNOSTIC_CONVENTIONS = ("lgal_native",)


def band_weight(nu_hz, trans, convention=DEFAULT_CONVENTION):
    """Weight w(ν) for ⟨f_ν⟩ = ∫ f_ν w dν / ∫ w dν.

    nu_hz, trans : (N,) frequency grid (Hz) and filter transmission.
    convention : 'energy' or 'photon'.
    """
    if convention == "energy":
        return trans
    if convention == "photon":
        return trans / nu_hz
    raise ValueError(
        f"unknown filter convention {convention!r}; choose from {CONVENTIONS}"
    )


def flam_to_fnu(wave_ang, f_lam):
    """Convert f_λ (erg/s/Å per Msun) to f_ν (erg/s/Hz per Msun)."""
    return f_lam * wave_ang**2 / C_ANG_S


def compute_ab_magnitudes(wave_ang, f_lam, filters, convention=DEFAULT_CONVENTION):
    """Compute synthetic AB magnitudes for a set of filters.

    wave_ang : (N_wave,) Å
    f_lam : (N_wave,) flux per unit wavelength (Lsun/Å or erg/s/Å per Msun)
    filters : {name: (wave_filter_AA, transmission)}
    convention : 'photon' (default), 'energy', or the 'lgal_native' diagnostic
        (see CONVENTIONS above) — never use lgal_native for science.

    Returns {filter_name: AB_magnitude}, np.nan where the filter doesn't
    overlap the SED.
    """
    if convention in DIAGNOSTIC_CONVENTIONS:
        # different quadrature entirely; imported here to keep it off the production import path
        from .lgal_quadrature import compute_lgal_magnitudes
        return compute_lgal_magnitudes(wave_ang, f_lam, filters)

    f_nu = flam_to_fnu(wave_ang, f_lam)
    nu_hz = C_ANG_S / wave_ang        # Hz, decreasing since wave increases

    mags = {}
    for name, (wave_f, trans_f) in filters.items():
        trans_on_sed = np.interp(wave_ang, wave_f, trans_f, left=0.0, right=0.0)
        if trans_on_sed.sum() == 0:
            mags[name] = np.nan
            continue

        idx = np.argsort(nu_hz)  # ascending ν (wave increases -> ν decreases)
        nu_s  = nu_hz[idx]
        fnu_s = f_nu[idx]
        T_s   = trans_on_sed[idx]

        # convention enters only via the weight; quadrature is identical either way
        w_s = band_weight(nu_s, T_s, convention)

        numerator   = np.trapezoid(fnu_s * w_s, nu_s)
        denominator = np.trapezoid(w_s, nu_s)

        if denominator <= 0 or numerator <= 0:
            mags[name] = np.nan
            continue

        # -48.6 zero-point assumes f_ν in erg/s/Hz/cm²; colours are independent
        # of any absolute-scale offset
        mags[name] = -2.5 * np.log10(numerator / denominator) - 48.6

    return mags


def compute_colours(mags, colour_pairs):
    """Colours from a magnitude dict: {'band1-band2': mag1 - mag2}.

    colour_pairs : list of (band1, band2) tuples, e.g. [('g', 'r')].
    """
    colours = {}
    for b1, b2 in colour_pairs:
        key = f"{b1}-{b2}"
        m1, m2 = mags.get(b1, np.nan), mags.get(b2, np.nan)
        colours[key] = m1 - m2
    return colours
