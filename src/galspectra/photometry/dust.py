"""
L-GALAXIES dust attenuation model.

Implements the two-component dust model from Delucia et al. (2007), which is used
in L-GALAXIES 2020 to compute MagDust from intrinsic SSP luminosities.

Components:
  1. ISM (diffuse disk):  Mathis, Mezger & Panagia (1983) extinction law,
     metallicity-dependent opacity, slab geometry with random inclination.
  2. Birth clouds:        extra attenuation for disk stars < 10 Myr,
     power-law τ ∝ λ^-0.7; fixed τ=0.5 for young bulge stars.

Reference: model_dust.c in LGalaxies2020_PublicRepository-master/code/
"""

import numpy as np

# ── Mathis, Mezger & Panagia (1983) extinction curve ──────────────────────────
# 42-point table: wavelength in μm, A_λ/A_V, and dust albedo.
# Copied verbatim from L-GALAXIES code/model_dust.c  get_extinction().
_MATHIS_LAMBDA_MUM = np.array([
    0.091, 0.10, 0.13, 0.143, 0.18, 0.20, 0.21, 0.216, 0.23, 0.25,
    0.346, 0.435, 0.55, 0.7, 0.9, 1.2, 1.8, 2.2, 2.4, 3.4,
    4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 20.0, 25.0, 30.0, 40.0,
    50.0, 60.0, 70.0, 80.0, 100.0, 150.0, 200.0, 300.0, 400.0, 600.0,
    800.0, 1000.0,
])

_MATHIS_AV = np.array([
    5.720, 4.650, 2.960, 2.700, 2.490, 2.780, 3.000, 3.120, 2.860, 2.350,
    1.580, 1.320, 1.000, 0.750, 0.480, 0.280, 0.160, 0.122, 0.093, 0.038,
    0.024, 0.018, 0.014, 0.013, 0.072, 0.030, 0.065, 0.062, 0.032, 0.017,
    0.014, 0.012, 9.7e-3, 8.5e-3, 6.5e-3, 3.7e-3, 2.5e-3, 1.1e-3, 6.7e-4, 2.5e-4,
    1.4e-4, 7.3e-5,
])

_MATHIS_ALBEDO = np.array([
    0.42, 0.43, 0.45, 0.45, 0.53, 0.56, 0.56, 0.56, 0.63, 0.63,
    0.71, 0.67, 0.63, 0.56, 0.50, 0.37, 0.25, 0.22, 0.15, 0.058,
    0.046, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00,
])

# Constants from h_variables.h
_EXP_TAU_BC_BULGE = 0.5    # effective surviving fraction of young bulge luminosity
_MUCENTER         = 0.3    # birth-cloud inclination factor mean
_MUWIDTH          = 0.2    # birth-cloud inclination factor width
_VBAND_MUM        = 0.55   # V-band reference wavelength in μm


def mathis_extinction(wave_ang):
    """
    Interpolate (A_λ/A_V, Albedo) from the Mathis 1983 table.

    Parameters
    ----------
    wave_ang : array_like — wavelength in Å

    Returns
    -------
    alav   : ndarray — A_λ/A_V (same shape as wave_ang)
    albedo : ndarray — dust albedo (same shape as wave_ang)
    """
    wave_mum = np.asarray(wave_ang, dtype=float) * 1e-4   # Å → μm
    alav   = np.interp(wave_mum, _MATHIS_LAMBDA_MUM, _MATHIS_AV,
                       left=_MATHIS_AV[0], right=0.0)
    albedo = np.interp(wave_mum, _MATHIS_LAMBDA_MUM, _MATHIS_ALBEDO,
                       left=_MATHIS_ALBEDO[0], right=0.0)
    return alav, albedo


def ism_optical_depth(wave_ang, Zg_solar, n_h, cos_incl):
    """
    Compute per-wavelength ISM optical depth τ_λ.

    Parameters
    ----------
    wave_ang  : (N,) — wavelength in Å
    Zg_solar  : float — gas metallicity in solar units (= Z_gas / 0.02)
    n_h       : float — H column density in units of [2.1×10²¹ atoms/cm²]
    cos_incl  : float — cos(inclination), clamped to [0.2, 1]

    Returns
    -------
    tau : (N,) — ISM optical depth τ_λ (before inclination; already includes sec(i))
    """
    alav, albedo = mathis_extinction(wave_ang)
    wave_mum = np.asarray(wave_ang, dtype=float) * 1e-4

    # Metallicity scaling: s = 1.35 for λ < 0.2 μm, 1.6 for λ ≥ 0.2 μm
    Zg = max(float(Zg_solar), 1e-6)
    metal_scale = np.where(wave_mum < 0.2,
                           Zg ** 1.35 * (1.0 - albedo) ** 0.5,
                           Zg ** 1.6  * (1.0 - albedo) ** 0.5)
    alav_eff = alav * metal_scale

    cosinc = max(float(cos_incl), 0.2)
    sec = 1.0 / cosinc
    return alav_eff * n_h * sec


def ism_attenuation(tau):
    """
    Slab-geometry attenuation factor a_λ = (1 − exp(−τ)) / τ.

    Parameters
    ----------
    tau : array_like — optical depth

    Returns
    -------
    a_lam : ndarray — attenuation factor in [0, 1]
    """
    tau = np.asarray(tau, dtype=float)
    a = np.where(tau > 0, (1.0 - np.exp(-tau)) / tau, 1.0)
    return a


def hydrogen_column_density(cold_gas_msun, cold_gas_radius_mpc_h, hubble_h=0.673,
                             redshift=0.0):
    """
    Compute the mean H column density N_H in [2.1×10²¹ atoms/cm²] units.

    Uses the same formula as L-GALAXIES model_dust.c, adapted for the .npy sample
    where ColdGas is in Msun (not 10^10 Msun/h) and ColdGasRadius is in Mpc/h.

    Parameters
    ----------
    cold_gas_msun       : float — cold gas mass in Msun
    cold_gas_radius_mpc_h : float — disk scale (3× scale length) in Mpc/h
    hubble_h            : float — Hubble parameter h
    redshift            : float — galaxy redshift (for (1+z)^-1 correction)

    Returns
    -------
    n_h : float — column density in [2.1×10²¹ atoms/cm²] units
    """
    if cold_gas_msun <= 0 or cold_gas_radius_mpc_h <= 0:
        return 0.0

    # Convert to code units: 10^10 Msun/h
    cold_gas_code = cold_gas_msun * hubble_h / 1e10

    # N_H in [10^10 Msun/h / (Mpc/h)^2] units, then divide by 3252.37 to get
    # [2.1×10^21 atoms/cm^2] units  (0.94 = 2.83/3 from disk profile integration)
    n_h = cold_gas_code / (np.pi * (cold_gas_radius_mpc_h * 0.94) ** 2 * 1.4) / 3252.37

    # Redshift evolution
    n_h *= (1.0 + redshift) ** (-1.0)
    return max(float(n_h), 0.0)


def draw_mu(rng=None):
    """
    Draw birth-cloud inclination parameter μ from Gaussian(0.3, 0.2), clamped to [0.1, 1.0].
    """
    if rng is None:
        rng = np.random.default_rng()
    for _ in range(1000):
        mu = rng.normal(_MUCENTER, _MUWIDTH)
        if 0.1 <= mu <= 1.0:
            return mu
    return _MUCENTER   # fallback


def birthcloud_optical_depth(wave_ang, tau_v_ism, mu):
    """
    Compute birth cloud optical depth τ_BC_λ for young disk stars.

    τ_BC_λ = τ_V_BC × (λ / 0.55 μm)^(-0.7)
    τ_V_BC = τ_V_ISM × (1/μ − 1)

    Parameters
    ----------
    wave_ang  : (N,) — wavelength in Å
    tau_v_ism : float — V-band ISM optical depth (with sec(i) already applied)
    mu        : float — birth-cloud inclination factor, drawn from Gaussian

    Returns
    -------
    tau_bc : (N,) — birth cloud optical depth per wavelength
    """
    tau_v_bc = tau_v_ism * (1.0 / mu - 1.0)
    wave_mum = np.asarray(wave_ang, dtype=float) * 1e-4
    return tau_v_bc * (wave_mum / _VBAND_MUM) ** (-0.7)


def apply_dust_to_seds(wave_ang,
                       sed_disk_old, sed_disk_young,
                       sed_bulge_old, sed_bulge_young,
                       cold_gas_msun, cold_gas_radius_mpc_h,
                       metals_cold_gas,
                       cos_incl,
                       redshift=0.0,
                       hubble_h=0.673,
                       rng=None):
    """
    Apply the complete L-GALAXIES two-component dust model to a galaxy SED.

    Implements the formula from model_dust.c (OUTPUT_REST_MAGS branch):
        SED_dust = SED_bulge_old
                 + SED_bulge_young × ExpTauBCBulge            (= 0.5)
                 + (SED_disk_old + SED_disk_young) × a_λ      (ISM on all disk)
                 − SED_disk_young × a_λ × (1 − exp(−τ_BC_λ)) (extra BC on young disk)

    Parameters
    ----------
    wave_ang              : (N_wave,) — wavelength in Å
    sed_disk_old          : (N_wave,) — old disk SED (age ≥ 10 Myr)
    sed_disk_young        : (N_wave,) — young disk SED (age < 10 Myr)
    sed_bulge_old         : (N_wave,) — old bulge SED
    sed_bulge_young       : (N_wave,) — young bulge SED
    cold_gas_msun         : float — ColdGas in Msun (from .npy sample)
    cold_gas_radius_mpc_h : float — ColdGasRadius in Mpc/h
    metals_cold_gas       : array of length ≥1 — MetalsColdGas in Msun (3 channels)
    cos_incl              : float — CosInclination (from galaxy sample)
    redshift              : float — galaxy redshift
    hubble_h              : float — Hubble h
    rng                   : numpy Generator or None — for μ sampling

    Returns
    -------
    sed_dust : (N_wave,) — total dust-attenuated SED (same units as inputs)
    """
    wave_ang = np.asarray(wave_ang, dtype=float)

    # Metallicity normalised to solar
    total_metals = np.asarray(metals_cold_gas, dtype=float).sum()
    if cold_gas_msun > 0:
        Zg_solar = (total_metals / cold_gas_msun) / 0.02
    else:
        Zg_solar = 0.0

    # Column density
    n_h = hydrogen_column_density(cold_gas_msun, cold_gas_radius_mpc_h,
                                  hubble_h=hubble_h, redshift=redshift)

    if n_h <= 0 or Zg_solar <= 0:
        # No dust
        return sed_disk_old + sed_disk_young + sed_bulge_old + sed_bulge_young

    # ISM attenuation factor a_λ
    tau_lam = ism_optical_depth(wave_ang, Zg_solar, n_h, cos_incl)
    a_lam   = ism_attenuation(tau_lam)

    # V-band τ (for birth cloud calibration) — evaluate at 0.55 μm
    alav_v, albedo_v = mathis_extinction(np.array([5500.0]))  # 5500 Å = V band
    alav_v = alav_v[0]; albedo_v = albedo_v[0]
    Zg = max(float(Zg_solar), 1e-6)
    alav_v_eff = alav_v * Zg ** 1.6 * (1.0 - albedo_v) ** 0.5
    cosinc = max(float(cos_incl), 0.2)
    tau_v_ism = alav_v_eff * n_h / cosinc   # V-band τ with sec already applied

    # Birth cloud for young disk
    mu = draw_mu(rng)
    tau_bc = birthcloud_optical_depth(wave_ang, tau_v_ism, mu)

    # Total dust SED
    sed_disk_total = sed_disk_old + sed_disk_young
    sed_dust = (sed_bulge_old
                + sed_bulge_young * _EXP_TAU_BC_BULGE
                + sed_disk_total * a_lam
                - sed_disk_young * a_lam * (1.0 - np.exp(-tau_bc)))
    return sed_dust
