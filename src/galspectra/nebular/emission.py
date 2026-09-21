"""
nebular/emission.py

Converts an ionizing photon rate Q(H0) into a nebular SED (lines + continuum).

Case B recombination at T_e = 1e4 K (Osterbrock 2006):
  L(Hα) = 1.37e-12 × f_ion × Q(H0)   [erg s⁻¹]

Other lines are fixed ratios to Hα (solar Z, typical HII region) — for
metallicity-dependent ratios, replace NEBULAR_LINES with Byler et al. (2017)
CLOUDY tables. Continuum (free-free + free-bound) is a Byler et al. (2017)
template proportional to Q(H0).
"""

import numpy as np

# name -> (wavelength_Å, ratio to Hα). Osterbrock (2006), case B, T_e=1e4 K, solar Z.
# Lyman-α assumes f_esc,Ly = 0.

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
    """Nebular SED for a given Q(H0).

    qh0_total : photons/s, already mass-scaled and summed over SFH bins.
    f_ion : fraction of ionizing photons producing emission (1 - f_esc).
    line_fwhm : Gaussian FWHM in Å.

    Returns sed_neb (N_wave,), erg/s/Å — same units as BC03 per-Msun SEDs,
    NOT yet divided by 4πd².
    """
    wave  = np.asarray(wave, dtype=np.float64)
    sed   = np.zeros_like(wave)

    if qh0_total <= 0:
        return sed

    l_ha   = _HAL_PER_PHOTON * f_ion * qh0_total   # erg s⁻¹
    sigma  = line_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # FWHM -> σ

    for name, (lam0, ratio) in NEBULAR_LINES.items():
        if lam0 < wave[0] or lam0 > wave[-1]:
            continue
        l_line = l_ha * ratio
        profile = np.exp(-0.5 * ((wave - lam0) / sigma) ** 2)
        profile /= (sigma * np.sqrt(2.0 * np.pi))  # normalised so integral = l_line
        sed += l_line * profile

    if include_continuum:
        sed += _nebular_continuum(wave, qh0_total * f_ion)

    return sed


# ── Nebular continuum template ────────────────────────────────────────────

def _nebular_continuum(wave, qh0):
    """Approximate nebular free-free + free-bound continuum.

    Normalised to L_cont(Hβ) = 1.58e-13 × Q(H0) erg/s/Å at 4861 Å (Byler et
    al. 2017, Eq. 9); shape is a λ^0.6 fit to the Rayleigh-Jeans-limit form,
    valid 900-10000 Å.
    """
    l_cont_hbeta = 1.58e-13 * qh0  # normalisation at Hβ, 4861 Å
    shape = (wave / 4861.0) ** 0.6
    shape[wave < 912.0] = 0.0  # zero below Lyman limit (ionising photons absorbed)
    return l_cont_hbeta * shape


# ── CSP nebular helper ────────────────────────────────────────────────────

def csp_nebular_sed(wave, ages_gyr, logzsol, mass_weights, qh0_interp,
                   f_ion=1.0, line_fwhm=5.0, age_max_neb_gyr=0.1):
    """Total nebular SED for a composite stellar population.

    Only bins with age < age_max_neb_gyr contribute (older ionizing output
    is negligible). qh0_interp : callable(age_gyr, Z) -> Q(H0) per Msun, from
    build_qh0_interpolator.

    Returns (sed_neb, qh0_total): (N_wave,) erg/s/Å, and total Q(H0) [photons/s].
    """
    qh0_total = 0.0
    for age, logz, mass in zip(ages_gyr, logzsol, mass_weights):
        if mass <= 0 or age > age_max_neb_gyr:
            continue
        Z = 0.02 * (10.0 ** logz)
        qh0_total += mass * float(np.squeeze(qh0_interp(age, Z)))

    return nebular_sed(wave, qh0_total, f_ion=f_ion, line_fwhm=line_fwhm), qh0_total
