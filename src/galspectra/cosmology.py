"""
Cosmology: one home for the parameters and the distances.

Values are **L-GALAXIES' own**, from
``input/input_MR_W1_PLANCK_LGals2020_DM.par`` (lines 50-52), so GALsPeCtrA
and the simulation it post-processes cannot disagree about the universe.

``hubble_h = 0.673`` had six independent copies (``lgalaxies/sfh.py``,
``photometry/dust.py`` twice, ``process_lgalaxies.py``,
``build_galaxy_table.py``, ``verify_pca_onthefly.py``) — not a per-tree
constant, so it lives here rather than in ``trees.py``. Migrating those call
sites is separate.

Distances come from ``astropy.cosmology.FlatLambdaCDM``, not a port of
L-GALAXIES' own integrator — implement correctly, then measure the gap
rather than silently inherit it. ``lgal_luminosity_distance_mpc`` reproduces
``lum_distance()`` (``..._onthefly_misc.c:76``) for exactly that measurement;
it's a diagnostic, not for science.

**Doesn't affect the observer-frame magnitude comparison**: ``ObsMag`` is
absolute (``h_galaxy_output.h:234``); distance only enters under
L-GALAXIES' ``APP`` flag, which is off.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "HUBBLE_H",
    "OMEGA_M",
    "OMEGA_LAMBDA",
    "SPEED_OF_LIGHT_KM_S",
    "TEN_PC_CM",
    "cosmology",
    "luminosity_distance_mpc",
    "distance_modulus",
    "lgal_luminosity_distance_mpc",
]

#: input/input_MR_W1_PLANCK_LGals2020_DM.par:50-52 — the Planck cosmology the
#: Millennium runs were scaled to. Imported, never restated.
HUBBLE_H = 0.673
OMEGA_M = 0.315
OMEGA_LAMBDA = 0.685

SPEED_OF_LIGHT_KM_S = 299792.458

#: L-GALAXIES' own constant, cm/s (code/h_params.h:9) — used only by the
#: diagnostic port below, which must match the C exactly.
_LGAL_SPEEDOFLIGHT = 2.9979e10

#: 10 pc in cm, the reference distance for absolute magnitudes — repeated
#: here (not re-derived) so it can't drift from the rest-frame path's copy.
TEN_PC_CM = 3.0856775814913673e19


_COSMO = None


def cosmology():
    """The astropy cosmology object, built once.

    Imported lazily so that the package does not require astropy for the rest-frame
    path, which has no need of distances.
    """
    global _COSMO
    if _COSMO is None:
        from astropy.cosmology import FlatLambdaCDM
        _COSMO = FlatLambdaCDM(H0=100.0 * HUBBLE_H, Om0=OMEGA_M)
    return _COSMO


def luminosity_distance_mpc(z):
    """Luminosity distance in Mpc (not Mpc/h). Scalar or array."""
    z = np.asarray(z, dtype=float)
    d = cosmology().luminosity_distance(np.atleast_1d(z)).value
    return d if z.ndim else float(d[0])


def distance_modulus(z):
    """m - M = 5 log10(d_L / 10 pc), d_L in Mpc.

    Defined only for z > 0; at z=0 a galaxy is at the reference distance, so
    the modulus is 0 by construction (matching the rest-frame path).
    """
    z = np.asarray(z, dtype=float)
    d_mpc = np.atleast_1d(luminosity_distance_mpc(z))
    out = np.where(d_mpc > 0, 5.0 * np.log10(np.maximum(d_mpc, 1e-30) * 1e5), 0.0)
    return out if z.ndim else float(out[0])


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic: L-GALAXIES' own integrator
# ─────────────────────────────────────────────────────────────────────────────

def lgal_luminosity_distance_mpc(redshift, n_points=1000):
    """`lum_distance()` (model_spectro_photometric_onthefly_misc.c:76), ported
    faithfully, defects included, so the gap from a correct calculation can be
    measured. Do not use for science.

    Three defects, all kept:
    * ``x[i] = h*(i-1)`` puts the first sample at **negative redshift** (-h), not 0.
    * The odd-index loop starts at k=-1, reading ``x[-1]`` — out of bounds in
      C. Python wraps ``x[-1]`` to the last element, unlike the C, so this
      port uses 0.0 there instead — **the one place it cannot be faithful**.
    * ``f[0]`` is never assigned before ``I[0] = (f[0]+f[1])/3``.

    Weights are 4/3 on odd, 2/3 on even — opposite `integrate()` in
    `model_misc.c` (`lgal_quadrature.py`), which is why both are worth measuring.
    """
    if redshift <= 0:
        return 0.0

    h = redshift / (n_points - 1)
    x = np.array([h * (i - 1) for i in range(n_points)])

    def integrand(xx):
        return 1.0 / np.sqrt((1.0 + xx) ** 2 * (1.0 + OMEGA_M * xx)
                             - xx * OMEGA_LAMBDA * (2.0 + xx))

    s_odd = 0.0
    for i in range(n_points // 2):
        k = 2 * i - 1
        s_odd += 0.0 if k < 0 else integrand(x[k])   # C reads x[-1]; see docstring
    i1 = s_odd * 4.0 / 3.0

    s_even = 0.0
    for i in range(n_points // 2 - 1):
        s_even += integrand(x[2 * i])
    i2 = s_even * 2.0 / 3.0

    f1 = integrand(x[0])
    i0 = (0.0 + f1) / 3.0        # f[0] never assigned in the C

    integral = h * (i0 + i1 + i2)
    dl = integral / 1000.0
    # SPEEDOFLIGHT is cm/s; /100 -> m/s, /1000 above + H0 division -> Mpc (ported verbatim)
    return dl * (1.0 + redshift) * (_LGAL_SPEEDOFLIGHT / 100.0) / (HUBBLE_H * 100.0)
