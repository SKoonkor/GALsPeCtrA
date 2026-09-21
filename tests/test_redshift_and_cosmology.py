"""Redshifting and cosmology. The spectral stretch and distance scaling are
tested separately."""

from __future__ import annotations

import numpy as np
import pytest

from galspectra import cosmology as cos
from galspectra.photometry.redshift import (
    apparent_magnitudes,
    observed_frame,
    observed_frame_absolute_magnitudes,
)


# ─────────────────────────────────────────────────────────────────────────────
# Cosmology
# ─────────────────────────────────────────────────────────────────────────────

def test_parameters_match_the_simulation():
    """Cosmology constants match L-GALAXIES' own parameter file."""
    assert cos.HUBBLE_H == 0.673
    assert cos.OMEGA_M == 0.315
    assert cos.OMEGA_LAMBDA == 0.685
    assert abs(cos.OMEGA_M + cos.OMEGA_LAMBDA - 1.0) < 1e-12, "assumed flat"


def test_luminosity_distance_is_monotonic_and_zero_at_zero():
    z = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 5.0])
    d = cos.luminosity_distance_mpc(z)
    assert d[0] == 0.0
    assert np.all(np.diff(d) > 0)


def test_distance_modulus_matches_its_definition():
    """m - M = 5 log10(d_L / 10 pc), checked against the distance rather than restated."""
    for z in (0.1, 1.0, 3.0):
        d_mpc = cos.luminosity_distance_mpc(z)
        assert abs(cos.distance_modulus(z) - 5.0 * np.log10(d_mpc * 1e5)) < 1e-9


def test_lgalaxies_own_integrator_agrees_to_half_a_percent():
    """L-GALAXIES' own lum_distance() agrees with astropy to better than 0.5%."""
    for z in (0.5, 1.0, 2.0, 5.0):
        ours = cos.luminosity_distance_mpc(z)
        theirs = cos.lgal_luminosity_distance_mpc(z)
        assert abs(theirs / ours - 1.0) < 0.005, f"z={z}: ratio {theirs / ours}"


# ─────────────────────────────────────────────────────────────────────────────
# The spectral stretch
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def spectrum():
    wave = np.geomspace(1000.0, 25000.0, 500)
    flux = (wave / 5500.0) ** -1.5
    return wave, flux


def test_z_zero_is_the_identity(spectrum):
    """z=0 is the identity, exactly."""
    wave, flux = spectrum
    w, f = observed_frame(wave, flux, 0.0)
    assert np.array_equal(w, wave)
    assert np.array_equal(f, flux)


def test_round_trip_recovers_the_rest_frame(spectrum):
    wave, flux = spectrum
    z = 1.7
    w, f = observed_frame(wave, flux, z)
    assert np.allclose(w / (1.0 + z), wave, rtol=0, atol=1e-9)
    assert np.allclose(f * (1.0 + z), flux, rtol=1e-14)


def test_the_stretch_conserves_energy(spectrum):
    """∫f_λ dλ is invariant under the redshift stretch."""
    wave, flux = spectrum
    total_rest = np.trapezoid(flux, wave)
    for z in (0.5, 2.0, 5.0):
        w, f = observed_frame(wave, flux, z)
        assert abs(np.trapezoid(f, w) / total_rest - 1.0) < 1e-12


def test_negative_redshift_is_refused(spectrum):
    wave, flux = spectrum
    with pytest.raises(ValueError, match="must be >= 0"):
        observed_frame(wave, flux, -0.1)


def test_stretch_handles_a_stack(spectrum):
    """(M, N) in, (M, N) out — the harness photometers many galaxies at once."""
    wave, flux = spectrum
    stack = np.vstack([flux, 2.0 * flux, 0.5 * flux])
    w, f = observed_frame(wave, stack, 1.0)
    assert f.shape == stack.shape
    assert np.allclose(f, stack / 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Photometry through the stretch
# ─────────────────────────────────────────────────────────────────────────────

def _flat_filters():
    """A tophat in f_nu-flat space, wide enough to stay on the grid at z = 2."""
    w = np.linspace(4000.0, 6000.0, 200)
    return {"flat": (w, np.ones_like(w))}


def test_apparent_is_absolute_plus_the_distance_modulus():
    """apparent_magnitudes == observed_frame_absolute_magnitudes + distance_modulus."""
    wave = np.geomspace(1000.0, 25000.0, 600)
    flux = (wave / 5500.0) ** -1.0
    filters = _flat_filters()
    z = 1.3

    absolute = observed_frame_absolute_magnitudes(wave, flux, filters, z)
    apparent = apparent_magnitudes(wave, flux, filters, z)
    dm = cos.distance_modulus(z)
    for band in absolute:
        assert abs((apparent[band] - absolute[band]) - dm) < 1e-9


def test_a_flat_f_nu_source_brightens_by_exactly_2p5_log10_1pz():
    """A flat-f_nu source brightens by exactly -2.5 log10(1+z) under redshifting."""
    wave = np.geomspace(1000.0, 40000.0, 2000)
    C = 2.99792458e18
    f_lam = C / wave ** 2                       # f_lambda for constant f_nu = 1

    filters = _flat_filters()
    ref = observed_frame_absolute_magnitudes(wave, f_lam, filters, 0.0)["flat"]
    for z in (0.5, 1.0, 2.0):
        m = observed_frame_absolute_magnitudes(wave, f_lam, filters, z)["flat"]
        expected = ref - 2.5 * np.log10(1.0 + z)
        assert abs(m - expected) < 1e-9, f"z={z}: {m} vs expected {expected}"
