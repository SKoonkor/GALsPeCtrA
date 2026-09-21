"""
nebular/ionizing.py

Pre-computes Q(H0), the H-ionizing photon rate per solar mass, from BC03
FullSED files over the Lyman continuum (91-912 Å, photon energy > 13.6 eV):

    Q(H0) [photons/s/Msun] = ∫_91^912 F_λ λ / (hc) dλ = 5.035e7 × ∫ F_λ λ dλ

BC03 flux is erg/s/Å/Msun. Calibration check: Q(H0) at age~0, Z_sun =
4.8e46 photons/s/Msun (consistent with Kennicutt 1998, Byler et al. 2017).
"""

from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator

_HC_ERG_ANG = 6.626e-27 * 2.998e18  # h×c in erg Å (= 1.986e-8 erg Å)
_CONV       = 1.0 / _HC_ERG_ANG     # 5.035e7 photons/(erg Å²), converts ∫F_λ×λ

_LYMC_MIN = 91.0   # Lyman-continuum limits, Å
_LYMC_MAX = 912.0


def compute_qh0_grid(bc03_dir, age_grid_gyr, Z_grid):
    """Q(H0) on a Cartesian (age, Z) grid, integrating each BC03 SSP over
    the Lyman continuum (91-912 Å).

    bc03_dir : directory with BC03_Chabrier_FullSED_m*.dat files.
    age_grid_gyr : (N_age,) linear Gyr. Z_grid : (N_Z,) linear metallicities.

    Returns (qh0 (N_age, N_Z) photons/s/Msun, age_grid_gyr, Z_grid) — the
    last two echoed back for convenience.
    """
    from galspectra.sps.bc03_backend import BC03Library

    bc03_dir = Path(bc03_dir)
    lib = BC03Library(bc03_dir, wave_min=_LYMC_MIN, wave_max=_LYMC_MAX)

    n_age = len(age_grid_gyr)
    n_Z   = len(Z_grid)
    qh0   = np.zeros((n_age, n_Z))

    for j, Z in enumerate(Z_grid):
        from galspectra.sps.bc03_backend import _nearest_Z
        Z_bc03 = _nearest_Z(float(Z))  # snap to nearest BC03 grid point
        ages_bc03, wave, flux_grid = lib.get(Z_bc03)

        # Q(H0) = CONV × ∫ F_λ × λ dλ, trapezoid rule; flux_grid is (N_age_bc03, N_wave)
        integrand = flux_grid * wave[np.newaxis, :]
        q_bc03    = _CONV * np.trapezoid(integrand, wave, axis=1)

        for i, age in enumerate(age_grid_gyr):  # onto the requested age grid
            age_clamped = np.clip(age, ages_bc03[0], ages_bc03[-1])
            qh0[i, j]   = np.interp(age_clamped, ages_bc03, q_bc03)

        print(f"  Z = {Z:.4f} done  "
              f"(Q(H0) range: {q_bc03.min():.2e} – {q_bc03.max():.2e})")

    return qh0, age_grid_gyr, Z_grid


def build_qh0_interpolator(qh0_file):
    """Load a pre-computed Q(H0) grid; returns callable(age_gyr, Z) -> Q(H0).

    Bilinear interpolation in (log age, log Z), clamped at grid boundaries.
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
