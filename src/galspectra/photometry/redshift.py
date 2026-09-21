"""
Placing a rest-frame spectrum at a redshift, and photometering it there.

Two operations, kept separate on purpose
----------------------------------------
**The spectral stretch** moves the spectrum in wavelength and dilutes it:

    lambda_obs = lambda_rest (1 + z)
    f_lambda_obs(lambda_obs) = f_lambda_rest(lambda_rest) / (1 + z)

This is where the physics is. It changes *colours*, because the filters no longer sample
the same rest-frame wavelengths, and it is what a k-correction undoes.

**The distance scaling** divides by 4 pi d_L^2 and is a single number per galaxy. It moves
every band by the same amount, so it changes brightness and not colour.

They are separate functions because they fail differently. A sign or (1+z) error in the
stretch produces wrong colours that look like astrophysics; an error in the distance
produces an offset so large it is obvious. Testing them together would let the first hide.

Why the (1+z) divisor is what it is
-----------------------------------
Energy conservation over a stretched band: the same photons arrive spread over
(1+z) times the wavelength interval, so the flux *density* per unit wavelength drops by
(1+z). Expressed in f_nu the factor is a multiplication instead — the module works in
f_lambda throughout and converts only inside the photometry, matching
`synthetic.compute_ab_magnitudes`.

Validating against L-GALAXIES
-----------------------------
`ObsMag` in `GALAXY_OUTPUT` is an **observer-frame absolute** magnitude
(`code/h_galaxy_output.h:234`) — the spectrum is redshifted, but it is still referred to
10 pc. The distance modulus only enters under L-GALAXIES' `APP` compile flag, which is
off. So `observed_frame_absolute_magnitudes` is the function that corresponds to `ObsMag`,
and it is the one to compare; `apparent_magnitudes` adds the modulus on top.

That split is deliberate: comparing against `ObsMag` tests the stretch, which is the part
that can plausibly be wrong, without the distance term muddying it.
"""

from __future__ import annotations

import numpy as np

from .synthetic import DEFAULT_CONVENTION, compute_ab_magnitudes

__all__ = [
    "observed_frame",
    "observed_frame_absolute_magnitudes",
    "apparent_magnitudes",
]


def observed_frame(wave_rest_ang, f_lam_rest, z):
    """Redshift a spectrum. Returns (wave_obs, f_lam_obs).

    Parameters
    ----------
    wave_rest_ang : (N,) rest-frame wavelength, Angstrom, increasing
    f_lam_rest    : (N,) or (M, N) rest-frame flux density per unit wavelength
    z             : float, >= 0

    No distance scaling — see the module docstring. At z = 0 this is the identity, which
    is asserted in the tests rather than assumed.
    """
    if z < 0:
        raise ValueError(f"redshift must be >= 0, got {z}")
    wave_rest_ang = np.asarray(wave_rest_ang, dtype=float)
    f_lam_rest = np.asarray(f_lam_rest, dtype=float)
    return wave_rest_ang * (1.0 + z), f_lam_rest / (1.0 + z)


def observed_frame_absolute_magnitudes(wave_rest_ang, f_lam_rest, filters, z,
                                       convention=DEFAULT_CONVENTION):
    """AB magnitudes of the redshifted spectrum, still referred to 10 pc.

    This is the quantity L-GALAXIES calls `ObsMag`. It carries the k-correction — the
    filters sample bluer rest-frame wavelengths as z rises — but not the distance.
    """
    wave_obs, f_lam_obs = observed_frame(wave_rest_ang, f_lam_rest, z)
    return compute_ab_magnitudes(wave_obs, f_lam_obs, filters, convention=convention)


def apparent_magnitudes(wave_rest_ang, f_lam_rest, filters, z,
                        convention=DEFAULT_CONVENTION):
    """Observer-frame apparent magnitudes: the above plus the distance modulus.

    Undefined at z = 0, where a galaxy sits at the reference distance and "apparent"
    has no meaning; `distance_modulus` returns 0 there and this reduces to the absolute
    magnitude rather than raising, which is the least surprising behaviour for a caller
    sweeping a redshift grid that starts at zero.
    """
    from ..cosmology import distance_modulus

    mags = observed_frame_absolute_magnitudes(
        wave_rest_ang, f_lam_rest, filters, z, convention=convention)
    dm = distance_modulus(z)
    return {band: (m + dm if np.isfinite(m) else m) for band, m in mags.items()}
