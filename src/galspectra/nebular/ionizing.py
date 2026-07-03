"""
nebular/ionizing.py

Pre-compute Q(H0) — the hydrogen-ionizing photon rate per solar mass —
from BC03 FullSED files for each (age, Z) grid point.

Q(H0) [photons s⁻¹ Msun⁻¹] = ∫_{91Å}^{912Å} F_λ × λ / (h c) dλ

where the integration is over the Lyman continuum (photon energy > 13.6 eV).

BC03 FullSED flux units: erg s⁻¹ Å⁻¹ Msun⁻¹  (verified from file values)

Physical constants
------------------
h  = 6.626 × 10⁻²⁷ erg s
c  = 2.998 × 10¹⁸  Å s⁻¹
=> h×c = 1.986 × 10⁻⁸  erg Å
=> Q(H0) = (1 / h×c) × ∫ F_λ × λ dλ  =  5.035 × 10⁷ × ∫ F_λ × λ dλ

Calibration check: Q(H0) at age≈0, Z_sun = 4.8 × 10⁴⁶ photons/s/Msun
  (consistent with Kennicutt 1998 and Byler et al. 2017).
"""

from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# ── Physical constants ────────────────────────────────────────────────────
_HC_ERG_ANG = 6.626e-27 * 2.998e18  # h×c in erg Å  (= 1.986e-8 erg Å)
_CONV       = 1.0 / _HC_ERG_ANG     # 5.035e7 photons / (erg Å²) — converts F_λ×λ integral

# Lyman-continuum wavelength limits (Å)
_LYMC_MIN = 91.0
_LYMC_MAX = 912.0


def compute_qh0_grid(bc03_dir, age_grid_gyr, Z_grid):
    """
    Compute Q(H0) for a Cartesian (age, Z) grid by integrating each BC03 SSP
    over the Lyman continuum (91–912 Å).

    Uses BC03Library with wave_min=91, wave_max=912 to load only the ionizing UV.

    Parameters
    ----------
    bc03_dir : str or Path
        Directory with BC03_Chabrier_FullSED_m*.dat files.
    age_grid_gyr : (N_age,) array
        Linear ages in Gyr (same grid used for the SSP SED grid).
    Z_grid : (N_Z,) array
        Linear metallicities (e.g. BC03_METALLICITIES).

    Returns
    -------
    qh0 : (N_age, N_Z) array  —  photons s⁻¹ Msun⁻¹
    age_grid_gyr : echoed back for convenience
    Z_grid       : echoed back for convenience
    """
    from galspectra.sps.bc03_backend import BC03Library

    bc03_dir = Path(bc03_dir)
    lib = BC03Library(bc03_dir, wave_min=_LYMC_MIN, wave_max=_LYMC_MAX)

    n_age = len(age_grid_gyr)
    n_Z   = len(Z_grid)
    qh0   = np.zeros((n_age, n_Z))

    for j, Z in enumerate(Z_grid):
        # Snap to nearest BC03 grid point to avoid float-key mismatches
        from galspectra.sps.bc03_backend import _nearest_Z
        Z_bc03 = _nearest_Z(float(Z))
        ages_bc03, wave, flux_grid = lib.get(Z_bc03)

        # wave is on BC03 native grid, 91–912 Å
        # flux_grid shape: (N_age_bc03, N_wave)
        # Integrate: Q(H0) = CONV × ∫ F_λ × λ dλ  (trapezoid rule)
        integrand = flux_grid * wave[np.newaxis, :]   # (N_age_bc03, N_wave)
        q_bc03    = _CONV * np.trapezoid(integrand, wave, axis=1)  # (N_age_bc03,)

        # Interpolate onto our requested age grid
        for i, age in enumerate(age_grid_gyr):
            age_clamped = np.clip(age, ages_bc03[0], ages_bc03[-1])
            qh0[i, j]   = np.interp(age_clamped, ages_bc03, q_bc03)

        print(f"  Z = {Z:.4f} done  "
              f"(Q(H0) range: {q_bc03.min():.2e} – {q_bc03.max():.2e})")

    return qh0, age_grid_gyr, Z_grid


def build_qh0_interpolator(qh0_file):
    """
    Load a pre-computed Q(H0) grid and return a callable interpolator.

    Parameters
    ----------
    qh0_file : str or Path
        Path to `data/qh0_grid_bc03.npz`

    Returns
    -------
    interpolator : callable(age_gyr, Z) -> Q(H0) in photons s⁻¹ Msun⁻¹
        Bilinear interpolation in (log age, log Z); clamped at grid boundaries.
    """
    data    = np.load(qh0_file)
    qh0     = data["qh0"]           # (N_age, N_Z)
    ages    = data["age_grid_gyr"]  # (N_age,)
    Z_arr   = data["Z_grid"]        # (N_Z,)

    # Build in log-log space for smoother interpolation
    log_ages = np.log10(ages)
    log_Z    = np.log10(Z_arr)
    log_qh0  = np.log10(np.maximum(qh0, 1e-100))  # guard against log(0)

    interp = RegularGridInterpolator(
        (log_ages, log_Z), log_qh0,
        method="linear", bounds_error=False, fill_value=None,
    )

    def _get(age_gyr, Z):
        """Interpolate Q(H0) [photons/s/Msun] at age (Gyr) and metallicity Z."""
        age = np.atleast_1d(np.float64(age_gyr))
        z   = np.atleast_1d(np.float64(Z))
        la  = np.log10(np.clip(age, ages[0], ages[-1]))
        lz  = np.log10(np.clip(z, Z_arr[0], Z_arr[-1]))
        pts = np.column_stack([la.ravel(), lz.ravel()])
        return 10.0 ** interp(pts).reshape(age.shape)

    return _get
