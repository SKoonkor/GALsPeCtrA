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


def compute_ab_magnitudes(wave_ang, f_lam, filters):
    """
    Compute synthetic AB magnitudes for a set of filters.

    Parameters
    ----------
    wave_ang : (N_wave,) array — wavelength in Å
    f_lam    : (N_wave,) array — spectral flux per unit wavelength
               (Lsun/Å per Msun, or erg/s/Å per Msun — see module docstring)
    filters  : dict {name: (wave_filter_Ang, transmission)}
               from galspectra.photometry.filters.load_sdss_filters() etc.

    Returns
    -------
    mags : dict {filter_name: AB_magnitude (float)}
          Returns np.nan for bands where the filter does not overlap the SED.
    """
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

        numerator   = np.trapezoid(fnu_s * T_s, nu_s)
        denominator = np.trapezoid(T_s, nu_s)

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
