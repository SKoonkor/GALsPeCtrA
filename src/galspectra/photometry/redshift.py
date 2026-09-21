"""
Placing a rest-frame spectrum at a redshift, and photometering it there.

Two operations, kept separate because they fail differently: the spectral
stretch (λ_obs = λ_rest(1+z), f_λ_obs = f_λ_rest/(1+z)) changes colours and is
where a sign or (1+z) error would look like real astrophysics; the distance
scaling (÷ 4πd_L²) is one number per galaxy and changes brightness only, so an
error there is an obvious offset instead.

Validating against L-GALAXIES: `ObsMag` (`GALAXY_OUTPUT`,
code/h_galaxy_output.h:234) is observer-frame **absolute**, still referred to
10 pc — the distance modulus is off by default (`APP` compile flag). So
`observed_frame_absolute_magnitudes` is what corresponds to `ObsMag`, and is
the function to compare against it; `apparent_magnitudes` adds the modulus.
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
    """Redshift a spectrum. Returns (wave_obs, f_lam_obs); no distance scaling.

    wave_rest_ang : (N,) Å, increasing. f_lam_rest : (N,) or (M, N). z >= 0.
    Identity at z=0 (asserted in tests).
    """
    if z < 0:
        raise ValueError(f"redshift must be >= 0, got {z}")
    wave_rest_ang = np.asarray(wave_rest_ang, dtype=float)
    f_lam_rest = np.asarray(f_lam_rest, dtype=float)
    return wave_rest_ang * (1.0 + z), f_lam_rest / (1.0 + z)


def observed_frame_absolute_magnitudes(wave_rest_ang, f_lam_rest, filters, z,
                                       convention=DEFAULT_CONVENTION):
    """AB magnitudes of the redshifted spectrum, still at 10 pc — L-GALAXIES'
    `ObsMag`. Carries the k-correction, not the distance."""
    wave_obs, f_lam_obs = observed_frame(wave_rest_ang, f_lam_rest, z)
    return compute_ab_magnitudes(wave_obs, f_lam_obs, filters, convention=convention)


def apparent_magnitudes(wave_rest_ang, f_lam_rest, filters, z,
                        convention=DEFAULT_CONVENTION):
    """Apparent magnitudes: observed_frame_absolute_magnitudes + distance modulus.

    At z=0, distance_modulus returns 0 (reduces to absolute mag rather than
    raising — the least surprising choice for a caller sweeping z from 0).
    """
    from ..cosmology import distance_modulus

    mags = observed_frame_absolute_magnitudes(
        wave_rest_ang, f_lam_rest, filters, z, convention=convention)
    dm = distance_modulus(z)
    return {band: (m + dm if np.isfinite(m) else m) for band, m in mags.items()}
