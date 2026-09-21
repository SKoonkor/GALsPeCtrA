"""
Cosmology: one home for the parameters and the distances.

Scope
-----
The values here are **L-GALAXIES' own**, read from
``input/input_MR_W1_PLANCK_LGals2020_DM.par`` (lines 50-52) so that GALsPeCtrA and the
simulation it post-processes cannot disagree about the universe they are in.

Why this module exists
----------------------
``hubble_h = 0.673`` had six independent copies — ``lgalaxies/sfh.py``,
``photometry/dust.py`` (twice), ``process_lgalaxies.py``, ``build_webapp_bundle.py`` and
``verify_pca_onthefly.py``. It is not a per-tree constant, so it does not belong in
``trees.py``; it is cosmology, and this is where it goes. Migrating those call sites is a
separate, mechanical change.

On matching L-GALAXIES
----------------------
Distances come from ``astropy.cosmology.FlatLambdaCDM``, not from a port of the
simulation's own integrator. That is a deliberate choice, and the same one taken over the
filter convolution: implement it correctly, then **measure** the difference against the
model rather than silently inheriting it.

``lgal_luminosity_distance`` reproduces L-GALAXIES' ``lum_distance()``
(``code/model_spectro_photometric_onthefly_misc.c:76``) so the gap can be quantified. It
is a diagnostic and should not be used for science. Its defects are documented there.

**This does not affect the observer-frame magnitude comparison.** ``ObsMag`` is an
*absolute* observer-frame magnitude (``h_galaxy_output.h:234``); the distance only enters
under L-GALAXIES' ``APP`` flag, which is off. So validating the redshifting path against
``ObsMag`` is unaffected by any disagreement about distance.
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

#: From `input/input_MR_W1_PLANCK_LGals2020_DM.par:50-52` — the Planck cosmology the
#: Millennium runs were scaled to. Changing these silently changes every distance, so
#: they live in one place and are imported, never restated.
HUBBLE_H = 0.673
OMEGA_M = 0.315
OMEGA_LAMBDA = 0.685

SPEED_OF_LIGHT_KM_S = 299792.458

#: L-GALAXIES' own constant, in cm/s (`code/h_params.h:9`). Used only by the diagnostic
#: port below, which must consume exactly the value the C consumes.
_LGAL_SPEEDOFLIGHT = 2.9979e10

#: 10 pc in cm — the reference distance for absolute magnitudes. The rest-frame path
#: already uses this constant; it is repeated here so the two cannot drift.
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
    """m - M = 5 log10(d_L / 10 pc), with d_L in Mpc.

    Defined only for z > 0. At z = 0 a galaxy is *at* the reference distance and the
    modulus is 0 by construction, which is what the rest-frame path already assumes.
    """
    z = np.asarray(z, dtype=float)
    d_mpc = np.atleast_1d(luminosity_distance_mpc(z))
    out = np.where(d_mpc > 0, 5.0 * np.log10(np.maximum(d_mpc, 1e-30) * 1e5), 0.0)
    return out if z.ndim else float(out[0])


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic: L-GALAXIES' own integrator
# ─────────────────────────────────────────────────────────────────────────────

def lgal_luminosity_distance_mpc(redshift, n_points=1000):
    """`lum_distance()` from `model_spectro_photometric_onthefly_misc.c:76`, ported.

    Reproduced faithfully, defects included, so the difference from a correct
    calculation can be measured. Do not use for science.

    Three things in the original are almost certainly unintended, and all three are kept:

    * ``x[i] = h*(i-1)`` makes the first sample sit at **negative redshift**, ``-h``,
      rather than at 0;
    * the odd-index loop starts at ``k = 2*i-1 = -1``, so it reads ``x[-1]`` — out of
      bounds in C. Python's ``x[-1]`` wraps to the last element, which is not what the C
      does, so this port uses 0.0 there rather than pretending to reproduce undefined
      behaviour. **This is the one place the port cannot be faithful**, and it is
      recorded rather than hidden;
    * ``f[0]`` is never assigned before ``I[0] = (f[0]+f[1])/3``, so the endpoint term
      carries only one of its two values.

    The weights are 4/3 on odd and 2/3 on even samples — the opposite assignment to
    `integrate()` in `model_misc.c`, which is what makes it worth measuring both.
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
        s_odd += 0.0 if k < 0 else integrand(x[k])   # see docstring: C reads x[-1]
    i1 = s_odd * 4.0 / 3.0

    s_even = 0.0
    for i in range(n_points // 2 - 1):
        s_even += integrand(x[2 * i])
    i2 = s_even * 2.0 / 3.0

    f1 = integrand(x[0])
    i0 = (0.0 + f1) / 3.0        # f[0] is never assigned in the C

    integral = h * (i0 + i1 + i2)
    dl = integral / 1000.0
    # The C's SPEEDOFLIGHT is in cm/s; /100 makes it m/s and the /1000 above plus the
    # H0 division carry it to Mpc. Ported verbatim rather than rederived.
    return dl * (1.0 + redshift) * (_LGAL_SPEEDOFLIGHT / 100.0) / (HUBBLE_H * 100.0)
