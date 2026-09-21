"""
Synthetic photometry.

Convolves an SED (flux vs wavelength) with filter transmission curves to
produce AB magnitudes.

AB magnitude formula:
    m_AB = -2.5 log10( ∫ f_ν(ν) T(ν) dν / ∫ T(ν) dν ) - 48.6

where:
    f_ν  = flux density in erg/s/Hz/cm²  (flux AT the detector, NOT a luminosity)
    T(ν) = filter transmission as a function of frequency

The -48.6 zero-point requires f_ν in erg/s/Hz/cm². BC03 and FSPS SEDs are
delivered in erg/s/Å per Msun initially formed — a specific luminosity, not a flux.
Before calling compute_ab_magnitudes, callers must scale the SED to a physical flux:

    sed_at_10pc = sed_per_msun * M_star_msun / (4 * pi * (10 pc in cm)^2)

This places the galaxy at 10 pc, giving absolute AB magnitudes comparable to
L-GALAXIES Mag/MagDust fields.  Omitting this scaling causes magnitudes ~70 mag
too bright (too negative) because erg/s/Å is many orders of magnitude larger than
erg/s/Å/cm² at any realistic distance.
"""

import numpy as np

C_ANG_S = 2.99792458e18  # speed of light in Å/s

# ─────────────────────────────────────────────────────────────────────────────
# Filter convolution convention
#
# The in-band average of f_ν is a weighted mean over the passband, and there are
# two conventions for the weight:
#
#   energy   ⟨f_ν⟩ = ∫ f_ν T dν / ∫ T dν
#   photon   ⟨f_ν⟩ = ∫ f_ν T dν/ν / ∫ T dν/ν
#
# They differ because a detector that counts *photons* weights each frequency by
# 1/(hν) relative to one that integrates *energy*. A CCD counts photons. The two
# agree exactly only for a source flat in f_ν across the band; for anything with
# colour inside the passband they differ, and the difference grows with how red
# the spectrum is — which is why this matters for old stellar populations and not
# for young ones.
#
# **`photon` is the default, adopted 7 August 2026.** It was chosen by measurement,
# not by argument: integrating the same BC03 `FullSEDs` under both conventions and
# comparing against L-GALAXIES' own `PhotTables` over 221 ages × 6 metallicities ×
# 40 bands, photon counting is closer in 30 of 40 bands and in every SDSS band,
# halving the *g−r* offset for old populations. `documents/filter_convention.md`
# has the measurement and the migration record.
#
# `energy` is kept selectable and is not deprecated. It is what every GALsPeCtrA
# product written before 7 August 2026 used, `data/pre_photon_energy_convention/`
# holds those products, and `tests/test_pca_basis_linearity.py` pins that this code
# still reproduces them. Do not remove it: the comparison between the two is a
# published claim and a referee may want to re-run it.
#
# A third name, `lgal_native`, selects L-GALAXIES' own quadrature rather than a
# weighting: see `lgal_quadrature.py`. It is a **diagnostic** — it deliberately
# reproduces three defects in the C code that writes `PhotTables`, so that the
# disagreement between that product and `FullSEDs` can be measured. It is not a
# physically correct magnitude and must never become the default.
CONVENTIONS = ("energy", "photon")
DEFAULT_CONVENTION = "photon"

#: Selectable through `compute_ab_magnitudes`, but not a member of `CONVENTIONS`,
#: because it is not a weighting choice within the same quadrature — it replaces the
#: quadrature. Keeping it out of `CONVENTIONS` also keeps it out of anything that
#: iterates the legitimate choices.
DIAGNOSTIC_CONVENTIONS = ("lgal_native",)


def band_weight(nu_hz, trans, convention=DEFAULT_CONVENTION):
    """Weight w(ν) for the in-band average ⟨f_ν⟩ = ∫ f_ν w dν / ∫ w dν.

    Parameters
    ----------
    nu_hz : (N,) — frequency grid, Hz
    trans : (N,) — filter transmission on that grid
    convention : 'energy' or 'photon'

    Both conventions share the same quadrature, so a comparison between them is a
    comparison of weight functions and nothing else.
    """
    if convention == "energy":
        return trans
    if convention == "photon":
        return trans / nu_hz
    raise ValueError(
        f"unknown filter convention {convention!r}; choose from {CONVENTIONS}"
    )


def flam_to_fnu(wave_ang, f_lam):
    """
    Convert f_λ (erg/s/Å per Msun) to f_ν (erg/s/Hz per Msun).

    Parameters
    ----------
    wave_ang : (N,) — wavelength in Å
    f_lam    : (N,) — flux per unit wavelength

    Returns
    -------
    f_nu : (N,) — flux per unit frequency
    """
    return f_lam * wave_ang**2 / C_ANG_S


def compute_ab_magnitudes(wave_ang, f_lam, filters, convention=DEFAULT_CONVENTION):
    """
    Compute synthetic AB magnitudes for a set of filters.

    Parameters
    ----------
    wave_ang : (N_wave,) array — wavelength in Å
    f_lam    : (N_wave,) array — spectral flux per unit wavelength
               (Lsun/Å per Msun, or erg/s/Å per Msun — see module docstring)
    filters  : dict {name: (wave_filter_Ang, transmission)}
               from galspectra.photometry.filters.load_sdss_filters() etc.
    convention : 'photon' (default) or 'energy' — see `CONVENTIONS`.
               'photon' is what a CCD measures and what reproduces L-GALAXIES'
               own tables; 'energy' reproduces GALsPeCtrA products written
               before 7 August 2026.
               'lgal_native' is a **diagnostic** that replaces the quadrature with
               L-GALAXIES' own, defects included — see `lgal_quadrature.py`. Do not
               use it for science.

    Returns
    -------
    mags : dict {filter_name: AB_magnitude (float)}
          Returns np.nan for bands where the filter does not overlap the SED.
    """
    if convention in DIAGNOSTIC_CONVENTIONS:
        # Not a weighting but a different quadrature entirely, so it cannot share the
        # loop below. Imported here to keep the diagnostic off the production import
        # path.
        from .lgal_quadrature import compute_lgal_magnitudes
        return compute_lgal_magnitudes(wave_ang, f_lam, filters)

    f_nu = flam_to_fnu(wave_ang, f_lam)

    # Convert wavelength to frequency (Hz) for integration
    # ν = c / λ, and dν = -c/λ² dλ → |dν/dλ| = c/λ²
    nu_hz = C_ANG_S / wave_ang        # Hz; decreasing in order since wave is increasing

    mags = {}
    for name, (wave_f, trans_f) in filters.items():
        # Interpolate filter onto the SED wavelength grid
        trans_on_sed = np.interp(wave_ang, wave_f, trans_f, left=0.0, right=0.0)

        # Check overlap
        if trans_on_sed.sum() == 0:
            mags[name] = np.nan
            continue

        # Integrate using the trapezoidal rule in frequency space
        # Working with ν increasing: flip arrays since wave increases → ν decreases
        idx = np.argsort(nu_hz)
        nu_s  = nu_hz[idx]
        fnu_s = f_nu[idx]
        T_s   = trans_on_sed[idx]

        # The convention enters only through the weight; the quadrature is
        # identical either way, so a difference between them is a difference in
        # physics rather than in numerics.
        w_s = band_weight(nu_s, T_s, convention)

        numerator   = np.trapezoid(fnu_s * w_s, nu_s)
        denominator = np.trapezoid(w_s, nu_s)

        if denominator <= 0 or numerator <= 0:
            mags[name] = np.nan
            continue

        # AB magnitude; the -48.6 zero-point assumes f_ν in erg/s/Hz/cm²
        # For SSP/CSP in Lsun/Å units, the absolute scale will include a
        # distance + unit conversion offset.  Within a single backend the
        # relative magnitudes (colours) are independent of this offset.
        mags[name] = -2.5 * np.log10(numerator / denominator) - 48.6

    return mags


def compute_colours(mags, colour_pairs):
    """
    Compute colours from a magnitude dictionary.

    Parameters
    ----------
    mags : dict {name: float} from compute_ab_magnitudes()
    colour_pairs : list of (band1, band2) tuples, e.g. [('g', 'r'), ('u', 'r')]

    Returns
    -------
    colours : dict {'band1-band2': float}
    """
    colours = {}
    for b1, b2 in colour_pairs:
        key = f"{b1}-{b2}"
        m1, m2 = mags.get(b1, np.nan), mags.get(b2, np.nan)
        colours[key] = m1 - m2
    return colours
