"""
nebular/emission.py

Convert an ionizing photon rate Q(H0) [photons s⁻¹] into a nebular SED
(emission lines + nebular continuum) in physical units on a user-supplied
wavelength grid.

Case B recombination at T_e = 10^4 K (Osterbrock 2006):
  L(Hα) = 1.37 × 10⁻¹² × f_ion × Q(H0)   [erg s⁻¹]

All other lines are fixed ratios relative to Hα (solar metallicity, typical HII
region conditions).  For metallicity-dependent ratios, replace NEBULAR_LINES
with Byler et al. (2017) CLOUDY tables.

Nebular continuum (free-free + free-bound) is approximated by a template
proportional to Q(H0) following Byler et al. (2017), normalised to
Q(H0) = 10^49 photons s⁻¹.
"""

import numpy as np

# ── Emission line catalogue ───────────────────────────────────────────────
# Format: name -> (wavelength_Å, luminosity_ratio_relative_to_Halpha)
# Ratios from Osterbrock (2006), case B, T_e = 1e4 K, solar Z.
# Lyman-alpha included assuming f_esc,Ly = 0 (all Ly photons destroyed/converted).

NEBULAR_LINES = {
    "Lyalpha": (1216.0, 23.5),    # Lyman-α  (case A recombination line)
    "OII_3727": (3727.0,  0.44),  # [OII] doublet (blended)
    "OII_3729": (3729.0,  0.44),  # second component (will merge with 3727)
    "Hgamma":  (4340.0,  0.47 * 0.47),  # Hγ
    "Hbeta":   (4861.0,  0.350),  # Hβ
    "OIII_4959": (4959.0, 0.14),  # [OIII]
    "OIII_5007": (5007.0, 0.43),  # [OIII] (strong line)
    "Halpha":  (6563.0,  1.000),  # Hα — reference
    "NII_6583": (6583.0, 0.250),  # [NII]
    "SII_6717": (6717.0, 0.100),  # [SII]
    "SII_6731": (6731.0, 0.080),
}

# Hα luminosity per ionizing photon: L(Hα) = HAL_PER_PHOTON × Q(H0)
_HAL_PER_PHOTON = 1.37e-12   # erg photon⁻¹  (Kennicutt 1998)


def nebular_sed(wave, qh0_total, f_ion=1.0, line_fwhm=5.0,
                include_continuum=True):
    """
    Compute a nebular SED for a given Q(H0).

    Parameters
    ----------
    wave : (N_wave,) array
        Wavelength grid in Å.
    qh0_total : float
        Total ionizing photon rate Q(H0) in photons s⁻¹ (already scaled by
        stellar mass and summed over SFH bins).
    f_ion : float, optional
        Fraction of ionizing photons that produce nebular emission (1 − escape
        fraction).  Default 1.0 (no escape).
    line_fwhm : float, optional
        Gaussian FWHM for emission line broadening in Å.  Default 5 Å.
    include_continuum : bool, optional
        Include nebular free-free + free-bound continuum.  Default True.

    Returns
    -------
    sed_neb : (N_wave,) array
        Nebular SED in erg s⁻¹ Å⁻¹  (NOT yet divided by 4πd² — same units
        as the BC03 per-Msun SEDs before the _FOUR_PI_10PC_SQ normalisation).
    """
    wave  = np.asarray(wave, dtype=np.float64)
    sed   = np.zeros_like(wave)

    if qh0_total <= 0:
        return sed

    # ── Emission lines ────────────────────────────────────────────────────
    l_ha   = _HAL_PER_PHOTON * f_ion * qh0_total   # erg s⁻¹
    sigma  = line_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # FWHM → σ in Å

    for name, (lam0, ratio) in NEBULAR_LINES.items():
        if lam0 < wave[0] or lam0 > wave[-1]:
            continue
        l_line = l_ha * ratio   # erg s⁻¹
        # Gaussian profile normalised so integral = l_line (in erg s⁻¹ Å⁻¹)
        profile = np.exp(-0.5 * ((wave - lam0) / sigma) ** 2)
        profile /= (sigma * np.sqrt(2.0 * np.pi))
        sed += l_line * profile

    # ── Nebular continuum ─────────────────────────────────────────────────
    if include_continuum:
        sed += _nebular_continuum(wave, qh0_total * f_ion)

    return sed


# ── Nebular continuum template ────────────────────────────────────────────

def _nebular_continuum(wave, qh0):
    """
    Approximate nebular free-free + free-bound continuum.

    Template: power-law approximation in UV–optical, normalised to
    L_cont(Hβ) = 1.58e-13 × Q(H0) erg s⁻¹ Å⁻¹ at 4861 Å
    (Byler et al. 2017, their Eq. 9).

    Shape follows e^{-hν/kT} / ν² Gaunt factor ∝ λ² exp(hc/λkT) in the
    Rayleigh-Jeans limit; we use a simpler λ^{0.6} polynomial fit valid
    from 900 to 10000 Å (adequate for the current wavelength range).
    """
    # Normalisation at Hβ (4861 Å)
    l_cont_hbeta = 1.58e-13 * qh0      # erg s⁻¹ Å⁻¹
    # Spectral shape: flat to slightly rising longward of Lyman break
    shape = (wave / 4861.0) ** 0.6
    # Zero below Lyman limit (912 Å) — ionising photons are absorbed
    shape[wave < 912.0] = 0.0
    return l_cont_hbeta * shape


# ── CSP nebular helper ────────────────────────────────────────────────────

def csp_nebular_sed(wave, ages_gyr, logzsol, mass_weights, qh0_interp,
                   f_ion=1.0, line_fwhm=5.0, age_max_neb_gyr=0.1):
    """
    Compute the total nebular SED for a composite stellar population.

    Only SFH bins with age < age_max_neb_gyr contribute (ionizing photon
    output from stars older than ~100 Myr is negligible).

    Parameters
    ----------
    wave : (N_wave,) array — wavelength grid in Å
    ages_gyr : (N_bins,) array — lookback ages in Gyr
    logzsol : (N_bins,) array — log10(Z/Zsun) per bin
    mass_weights : (N_bins,) array — stellar mass in Msun per bin
    qh0_interp : callable(age_gyr, Z) -> Q(H0) per Msun
        From `build_qh0_interpolator`.
    f_ion : float — ionising photon absorption fraction (default 1.0)
    line_fwhm : float — Gaussian FWHM in Å (default 5 Å)
    age_max_neb_gyr : float — age cut for nebular contribution (default 0.1 Gyr)

    Returns
    -------
    sed_neb : (N_wave,) erg s⁻¹ Å⁻¹  (same units as BC03 SEDs)
    qh0_total : float  — total Q(H0) [photons s⁻¹] used
    """
    qh0_total = 0.0
    for age, logz, mass in zip(ages_gyr, logzsol, mass_weights):
        if mass <= 0 or age > age_max_neb_gyr:
            continue
        Z = 0.02 * (10.0 ** logz)
        qh0_total += mass * float(np.squeeze(qh0_interp(age, Z)))

    return nebular_sed(wave, qh0_total, f_ion=f_ion, line_fwhm=line_fwhm), qh0_total
