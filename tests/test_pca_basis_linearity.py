"""
Structural guarantees of the spectral basis: reconstruction is linear in
flux, and the mean term is scaled by total formed mass. Runs on a small
synthetic library where possible; tests needing the real library skip if absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from galspectra.pca.basis import SpectralBasis
from galspectra.pca.metrics import BandOperator, d4000_operator, d4000_from_operator
from galspectra.pca.resample import (
    log_wavelength_grid,
    resampling_matrix,
    cumulative_integral_matrix,
)
from galspectra.pca.weighting import REFUSED_SCHEMES, SCHEMES, build_weights, check_weights

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SED_GRID = PROJECT_ROOT / "data" / "sed_grid_bc03.npz"
LGAL_FILTERS = (
    PROJECT_ROOT.parent / "L-GALAXIES" / "LGalaxies2020_PublicRepository-master"
    / "SpecPhotTables" / "Filters"
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_library():
    """A small, strictly positive, smoothly varying stand-in for an SSP grid."""
    rng = np.random.default_rng(20260806)
    wave = np.geomspace(1200.0, 22000.0, 260)
    temps = np.geomspace(2500.0, 40000.0, 40)
    slopes = np.linspace(-0.6, 0.6, 5)

    rows = []
    for T in temps:
        x = 1.4388e8 / (wave * T)          # hc/(λ k T) with λ in Å
        planck = 1.0 / (wave ** 5 * np.expm1(np.clip(x, 1e-6, 500.0)))
        for s in slopes:
            rows.append(planck * (wave / 5500.0) ** s)
    X = np.asarray(rows)
    X = X / X.max()
    X *= (1.0 + 1e-6 * rng.standard_normal(X.shape))   # break exact degeneracy
    return wave, np.abs(X)


@pytest.fixture(scope="module")
def real_library():
    if not SED_GRID.exists():
        pytest.skip(f"{SED_GRID} not present")
    d = np.load(SED_GRID, allow_pickle=True)
    return (np.asarray(d["wave"], dtype=float),
            np.asarray(d["seds"], dtype=float),
            np.asarray(d["params"], dtype=float),
            list(d["param_names"]))


# ─────────────────────────────────────────────────────────────────────────────
# Guardrail 1 — linear in flux
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("scheme", ["uniform", "inverse_std", "inverse_rms", "inverse_mean"])
def test_composition_is_exact_to_machine_precision(synthetic_library, scheme):
    """recon(Σ mᵢcᵢ, Σ mᵢ) == Σ mᵢ · recon(cᵢ, 1), to machine precision."""
    wave, X = synthetic_library
    basis = SpectralBasis.fit(wave, X, 20, weight_scheme=scheme)
    coeffs = basis.transform(X)

    rng = np.random.default_rng(7)
    masses = rng.uniform(1e6, 1e11, size=X.shape[0])

    C, M = basis.compose(coeffs, masses)
    lhs = basis.reconstruct(C, M)
    rhs = (masses[:, None] * basis.reconstruct(coeffs, np.ones(X.shape[0]))).sum(axis=0)

    scale = np.abs(rhs).max()
    assert np.abs(lhs - rhs).max() / scale < 1e-12, (
        f"composition is not linear under '{scheme}': "
        f"max relative discrepancy {np.abs(lhs - rhs).max() / scale:.3e}"
    )


def test_composition_holds_for_arbitrary_subsets(synthetic_library):
    """Linearity holds for any subset, not just the whole library."""
    wave, X = synthetic_library
    basis = SpectralBasis.fit(wave, X, 25, weight_scheme="inverse_rms")
    coeffs = basis.transform(X)
    rng = np.random.default_rng(11)
    masses = rng.uniform(0.0, 1e10, size=X.shape[0])

    whole = basis.reconstruct(*basis.compose(coeffs, masses))
    split = np.zeros_like(whole)
    for sl in (slice(0, 40), slice(40, 130), slice(130, None)):
        split += basis.reconstruct(*basis.compose(coeffs[sl], masses[sl]))

    assert np.abs(whole - split).max() / np.abs(whole).max() < 1e-12


def test_reconstruction_scales_linearly_with_mass(synthetic_library):
    """Doubling the mass must exactly double the flux — no hidden normalisation."""
    wave, X = synthetic_library
    basis = SpectralBasis.fit(wave, X, 15, weight_scheme="inverse_std")
    c = basis.transform(X[:1])[0]
    a = basis.reconstruct(c * 3.0, 3.0)
    b = 3.0 * basis.reconstruct(c, 1.0)
    assert np.allclose(a, b, rtol=0, atol=1e-9 * np.abs(b).max())


def test_weights_are_divided_back_out(synthetic_library):
    """The stored basis is in flux units: different weightings, same reconstruction."""
    wave, X = synthetic_library
    n = min(X.shape) - 1
    a = SpectralBasis.fit(wave, X, n, weight_scheme="uniform")
    b = SpectralBasis.fit(wave, X, n, weight_scheme="inverse_rms")
    ra = a.reconstruct(a.transform(X), np.ones(X.shape[0]))
    rb = b.reconstruct(b.transform(X), np.ones(X.shape[0]))
    assert np.abs(ra - rb).max() / np.abs(X).max() < 1e-8


def test_overall_weight_scale_cancels(synthetic_library):
    """Scaling every weight by a constant doesn't change the reconstruction."""
    wave, X = synthetic_library
    a = SpectralBasis.fit(wave, X, 12, weight_scheme="inverse_rms",
                          weight_kwargs={"normalise": True})
    b = SpectralBasis.fit(wave, X, 12, weight_scheme="inverse_rms",
                          weight_kwargs={"normalise": False})

    k = b.weights[0] / a.weights[0]
    assert np.allclose(b.weights, k * a.weights, rtol=1e-12)
    assert np.allclose(b.mu_eff, a.mu_eff, rtol=1e-10)

    ones = np.ones(X.shape[0])
    ra = a.reconstruct(a.transform(X), ones)
    rb = b.reconstruct(b.transform(X), ones)
    assert np.abs(ra - rb).max() < 1e-9 * np.abs(X).max()


# ─────────────────────────────────────────────────────────────────────────────
# Guardrail 1 — the refusals
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("scheme", sorted(REFUSED_SCHEMES))
def test_per_spectrum_schemes_are_refused(synthetic_library, scheme):
    """l2, 5500A, log and peak must be refused by name, with a reason."""
    wave, X = synthetic_library
    with pytest.raises(ValueError, match="refused"):
        build_weights(scheme, X, wave)
    with pytest.raises(ValueError):
        SpectralBasis.fit(wave, X, 5, weight_scheme=scheme)


def test_two_dimensional_weights_are_rejected():
    """A per-spectrum weight is structurally impossible, not merely discouraged."""
    with pytest.raises(ValueError, match="1-D over wavelength"):
        check_weights(np.ones((10, 4)), 4)


def test_non_positive_weights_are_rejected():
    with pytest.raises(ValueError, match="<= 0"):
        check_weights(np.array([1.0, 0.0, 2.0]), 3)


def test_every_permitted_scheme_yields_positive_finite_weights(synthetic_library):
    wave, X = synthetic_library
    for scheme in SCHEMES:
        kw = ({"base": "inverse_rms", "regions": [{"min": 3000, "max": 5000, "factor": 2.0}]}
              if scheme == "piecewise" else {})
        w, meta = build_weights(scheme, X, wave, **kw)
        assert w.shape == (wave.size,)
        assert np.all(np.isfinite(w)) and np.all(w > 0)
        assert meta["scheme"] == scheme


# ─────────────────────────────────────────────────────────────────────────────
# Guardrail 2 — the mean term is carried by mass
# ─────────────────────────────────────────────────────────────────────────────

def test_dropping_the_mass_term_is_catastrophic(synthetic_library):
    """Omitting M from the mean term changes the answer by far more than 1%."""
    wave, X = synthetic_library
    basis = SpectralBasis.fit(wave, X, 20, weight_scheme="inverse_std")
    coeffs = basis.transform(X)
    rng = np.random.default_rng(3)
    masses = rng.uniform(1e8, 1e10, size=X.shape[0])

    C, M = basis.compose(coeffs, masses)
    correct = basis.reconstruct(C, M)
    broken = basis.reconstruct_without_mass(C)

    rel = np.median(np.abs(broken - correct) / np.abs(correct))
    assert rel > 0.01, (
        f"dropping the total formed mass changed the reconstruction by only {rel:.3%}; "
        "the mass term is supposed to be load-bearing"
    )


def test_mass_term_is_needed_even_for_a_single_ssp_of_non_unit_mass(synthetic_library):
    wave, X = synthetic_library
    basis = SpectralBasis.fit(wave, X, 10, weight_scheme="uniform")
    c = basis.transform(X[:1])[0]
    m = 5.0e9
    correct = basis.reconstruct(c * m, m)
    broken = basis.reconstruct_without_mass(c * m)
    assert np.median(np.abs(broken - correct) / np.abs(correct)) > 0.01


def test_bytes_accounting_counts_the_mass(synthetic_library):
    """The honest storage cost is N_pc + 1 numbers, not N_pc."""
    wave, X = synthetic_library
    basis = SpectralBasis.fit(wave, X, 50, weight_scheme="inverse_std")
    assert basis.bytes_per_galaxy(np.float32) == 51 * 4
    assert basis.bytes_per_galaxy(np.float32, store_mass=False) == 50 * 4


# ─────────────────────────────────────────────────────────────────────────────
# Truncation, persistence
# ─────────────────────────────────────────────────────────────────────────────

def test_truncation_matches_a_refit(synthetic_library):
    """Truncating a fit equals refitting at that N (PCA components are nested)."""
    wave, X = synthetic_library
    full = SpectralBasis.fit(wave, X, 30, weight_scheme="inverse_rms")
    trunc = full.truncate(8)
    refit = SpectralBasis.fit(wave, X, 8, weight_scheme="inverse_rms")

    ones = np.ones(X.shape[0])
    a = trunc.reconstruct(full.transform(X)[:, :8], ones)
    b = refit.reconstruct(refit.transform(X), ones)
    assert np.abs(a - b).max() / np.abs(X).max() < 1e-8


def test_save_load_round_trip(synthetic_library, tmp_path):
    wave, X = synthetic_library
    basis = SpectralBasis.fit(wave, X, 12, weight_scheme="inverse_rms")
    path = basis.save(tmp_path / "basis.npz")
    loaded = SpectralBasis.load(path)

    ones = np.ones(X.shape[0])
    a = basis.reconstruct(basis.transform(X), ones)
    b = loaded.reconstruct(loaded.transform(X), ones)
    assert np.array_equal(basis.components, loaded.components)
    assert np.allclose(a, b, rtol=0, atol=1e-12 * np.abs(a).max())


def test_load_rejects_an_unversioned_file(tmp_path):
    """A file with no format-version marker is rejected."""
    p = tmp_path / "bogus.npz"
    np.savez(p, wave=np.arange(3.0), mu_eff=np.zeros(3), components=np.zeros((1, 3)),
             weights=np.ones(3))
    with pytest.raises(ValueError, match="unrecognised basis format"):
        SpectralBasis.load(p)


# ─────────────────────────────────────────────────────────────────────────────
# Resampling
# ─────────────────────────────────────────────────────────────────────────────

def test_resampling_is_a_linear_operator():
    wave = np.geomspace(1000.0, 20000.0, 900)
    _, edges = log_wavelength_grid(1100.0, 19000.0, 250)
    M = resampling_matrix(wave, edges)
    rng = np.random.default_rng(5)
    a, b = rng.uniform(0.1, 3.0, size=(2, wave.size))
    lhs = M @ (2.5 * a - 1.3 * b)
    rhs = 2.5 * (M @ a) - 1.3 * (M @ b)
    assert np.abs(lhs - rhs).max() < 1e-12 * np.abs(rhs).max()


def test_resampling_conserves_flux():
    wave = np.geomspace(1000.0, 20000.0, 1500)
    _, edges = log_wavelength_grid(wave[0], wave[-1], 200)
    M = resampling_matrix(wave, edges)
    rng = np.random.default_rng(9)
    f = rng.uniform(0.05, 4.0, size=wave.size)
    total_out = float(np.sum((M @ f) * np.diff(edges)))
    total_in = float(np.trapezoid(f, wave))
    assert abs(total_out - total_in) / total_in < 1e-12


def test_resampling_is_exact_for_a_linear_spectrum():
    wave = np.linspace(2000.0, 18000.0, 3000)
    f = 7.0 - 1.5e-4 * wave
    _, edges = log_wavelength_grid(2100.0, 17000.0, 180)
    out = resampling_matrix(wave, edges) @ f
    exact = 7.0 - 1.5e-4 * 0.5 * (edges[:-1] + edges[1:])
    assert np.abs(out - exact).max() < 1e-10


def test_resampling_refuses_to_extrapolate():
    wave = np.geomspace(3000.0, 9000.0, 200)
    _, edges = log_wavelength_grid(1000.0, 9000.0, 100)
    with pytest.raises(ValueError, match="not contained"):
        resampling_matrix(wave, edges)


def test_cumulative_integral_matches_trapezoid():
    wave = np.geomspace(1000.0, 20000.0, 800)
    rng = np.random.default_rng(13)
    f = rng.uniform(0.1, 2.0, size=wave.size)
    T = cumulative_integral_matrix(wave, [wave[0], wave[-1]])
    assert abs((T[1] - T[0]) @ f - np.trapezoid(f, wave)) < 1e-9 * abs(np.trapezoid(f, wave))


# ─────────────────────────────────────────────────────────────────────────────
# Photometry: the fast path must equal the production path
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not LGAL_FILTERS.exists(), reason="L-GALAXIES filter directory absent")
def test_band_operator_reproduces_production_magnitudes(real_library):
    """BandOperator agrees with compute_ab_magnitudes, the production photometry."""
    from galspectra.photometry.filter_sets import load_filter_set
    from galspectra.photometry.synthetic import compute_ab_magnitudes

    wave, X, _params, _names = real_library
    filters = load_filter_set("lgal_sdss")
    op = BandOperator(wave, filters)

    for i in (0, 137, 600, 1199):
        ref = compute_ab_magnitudes(wave, X[i], filters)
        fast = op.magnitudes(X[i])
        for band in op.names:
            assert abs(ref[band] - fast[op.index(band)]) < 1e-10, (
                f"band {band}, SSP {i}: production {ref[band]!r} vs operator "
                f"{fast[op.index(band)]!r}"
            )


def test_band_flux_composes_linearly(synthetic_library):
    """Band flux of a sum equals the sum of band fluxes."""
    from galspectra.photometry.filter_sets import load_filter_set

    wave, X = synthetic_library
    try:
        filters = load_filter_set("megacam")
    except FileNotFoundError:
        pytest.skip("MegaCam curves absent")
    op = BandOperator(wave, filters, min_coverage=0.90)

    rng = np.random.default_rng(17)
    m = rng.uniform(0, 1e9, size=X.shape[0])
    lhs = op.band_flux(m @ X)
    rhs = m @ op.band_flux(X)
    assert np.abs(lhs - rhs).max() / np.abs(rhs).max() < 1e-12


def test_band_operator_rejects_uncovered_bands(synthetic_library):
    """A band hanging off the end of the grid must be dropped, not silently truncated."""
    from galspectra.photometry.filter_sets import load_filter_set

    wave, _X = synthetic_library
    try:
        filters = load_filter_set("megacam", "2mass_ks")
    except FileNotFoundError:
        pytest.skip("filter curves absent")
    narrow = wave[(wave > 3000) & (wave < 9000)]
    op = BandOperator(narrow, filters, min_coverage=0.99)
    assert "2MASS_Ks" in op.rejected
    assert "2MASS_Ks" not in op.names


# ─────────────────────────────────────────────────────────────────────────────
# D4000
# ─────────────────────────────────────────────────────────────────────────────

def test_d4000_is_a_ratio_of_linear_functionals(synthetic_library):
    """D4000 of a composite follows from the composite's window fluxes."""
    wave, X = synthetic_library
    op = d4000_operator(wave, "D4000_n")
    rng = np.random.default_rng(23)
    m = rng.uniform(0, 1e8, size=X.shape[0])

    combined = d4000_from_operator(op, m @ X)
    windows = m @ (X @ op.T)
    assert abs(combined - windows[1] / windows[0]) < 1e-10 * abs(combined)


def test_d4000_windows_must_be_inside_the_grid():
    wave = np.geomspace(5000.0, 9000.0, 100)
    with pytest.raises(ValueError, match="outside grid"):
        d4000_operator(wave, "D4000_n")


@pytest.mark.skipif(not SED_GRID.exists(), reason="SED grid absent")
def test_d4000_of_real_ssps_is_physical(real_library):
    """Old populations must have a stronger break than young ones."""
    wave, X, params, _names = real_library
    op = d4000_operator(wave, "D4000_n")
    d = d4000_from_operator(op, X)
    ages = params[:, 0]
    solar = np.isclose(params[:, 1], 0.0)
    young = solar & (ages < 0.05)
    old = solar & (ages > 5.0)
    assert np.median(d[old]) > np.median(d[young]) + 0.5
    assert np.all(d[np.isfinite(d)] > 0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Regression pin: agreement with L-GALAXIES' own magnitudes
# ─────────────────────────────────────────────────────────────────────────────

BUNDLE = PROJECT_ROOT / "data" / "galaxy_table_MR.npz"

# per-band pin: (max allowed |median Δ|, max allowed MAD) against native L-GALAXIES
_MAG_PIN = {
    "u": (0.030, 0.006),
    "g": (0.035, 0.012),
    "r": (0.010, 0.004),
    "i": (0.008, 0.004),
    "z": (0.003, 0.002),
}


@pytest.mark.skipif(not BUNDLE.exists(), reason="galaxy table absent")
@pytest.mark.parametrize("band", sorted(_MAG_PIN))
def test_intrinsic_magnitude_agreement_pin(band):
    d = np.load(BUNDLE, allow_pickle=True)
    native = np.asarray(d[f"Mag_{band}"], dtype=float)
    synth = np.asarray(d[f"synth_mag_{band}"], dtype=float)
    # L-GALAXIES writes 99.0, not NaN, for "no magnitude".
    ok = np.isfinite(native) & np.isfinite(synth) & (np.abs(native) < 90.0)
    assert ok.sum() > 1000, f"only {ok.sum()} valid galaxies in band {band}"

    delta = synth[ok] - native[ok]
    med = float(np.median(delta))
    mad = float(np.median(np.abs(delta - med)))
    max_med, max_mad = _MAG_PIN[band]
    assert abs(med) <= max_med, f"{band}: median Δ = {med:+.4f}, pin allows |Δ| <= {max_med}"
    assert mad <= max_mad, f"{band}: MAD = {mad:.4f}, pin allows <= {max_mad}"


@pytest.mark.skipif(not BUNDLE.exists(), reason="galaxy table absent")
def test_stored_coefficients_are_not_self_contained():
    """`pca_coeffs` alone cannot reconstruct a galaxy — the mass is a separate number."""
    d = np.load(BUNDLE, allow_pickle=True)
    assert "pca_coeffs" in d
    assert "M_total" in d, (
        "the bundle carries M_total separately; if that ever changes, the "
        "'50 floats per galaxy' figure and this test both need revisiting"
    )
    m = np.asarray(d["M_total"], dtype=float)
    assert np.median(m[m > 0]) > 1e8, "M_total does not look like a stellar mass in Msun"


# ─────────────────────────────────────────────────────────────────────────────
# The filter convolution convention
# ─────────────────────────────────────────────────────────────────────────────

ARCHIVE = PROJECT_ROOT / "data" / "pre_photon_energy_convention"


def test_lgal_native_is_a_diagnostic_and_never_a_default():
    """`lgal_native` stays out of CONVENTIONS and out of DEFAULT_CONVENTION."""
    from galspectra.photometry import synthetic
    assert "lgal_native" not in synthetic.CONVENTIONS
    assert "lgal_native" in synthetic.DIAGNOSTIC_CONVENTIONS
    assert synthetic.DEFAULT_CONVENTION not in synthetic.DIAGNOSTIC_CONVENTIONS


@pytest.mark.skipif(not LGAL_FILTERS.exists(), reason="L-GALAXIES filter directory absent")
def test_lgal_native_reproduces_the_c_quadrature_matrix_form(real_library):
    """lgal_band_matrix (matrix form) agrees with compute_lgal_magnitudes (literal port)."""
    from galspectra.photometry.filter_sets import load_filter_set
    from galspectra.photometry.lgal_quadrature import (
        compute_lgal_magnitudes, lgal_band_matrix, lgal_magnitudes_from_matrix,
    )

    wave, X, _params, _names = real_library
    filters = load_filter_set("lgal_sdss")
    names, P, denom = lgal_band_matrix(wave, filters)
    assert names, "no band survived the grid"

    for i in (0, 137, 600, 1199):
        direct = compute_lgal_magnitudes(wave, X[i], filters)
        fast = lgal_magnitudes_from_matrix(X[i], P, denom)[0]
        for j, band in enumerate(names):
            assert abs(direct[band] - fast[j]) < 1e-10


@pytest.mark.skipif(not LGAL_FILTERS.exists(), reason="L-GALAXIES filter directory absent")
def test_lgal_native_differs_from_production_in_the_measured_direction(real_library):
    """lgal_native differs from production by a few mmag — neither zero nor tenths."""
    from galspectra.photometry.filter_sets import load_filter_set
    from galspectra.photometry.synthetic import compute_ab_magnitudes

    wave, X, params, _names = real_library
    filters = load_filter_set("lgal_sdss")
    old = np.where((params[:, 0] > 5.0) & np.isclose(params[:, 1], 0.0))[0]
    assert old.size

    i = int(old[0])
    prod = compute_ab_magnitudes(wave, X[i], filters)
    lgal = compute_ab_magnitudes(wave, X[i], filters, convention="lgal_native")

    # Colours, so the differing zero points cancel.
    delta = abs((prod["g"] - prod["r"]) - (lgal["g"] - lgal["r"]))
    assert 1e-4 < delta < 0.1, f"g-r differs by {delta:.5f}, outside the measured range"


def test_default_convention_is_photon():
    """DEFAULT_CONVENTION is 'photon'; 'energy' stays selectable."""
    from galspectra.photometry.synthetic import CONVENTIONS, DEFAULT_CONVENTION
    assert DEFAULT_CONVENTION == "photon"
    assert "energy" in CONVENTIONS


def test_the_two_photometry_entry_points_share_one_default():
    """BandOperator and compute_ab_magnitudes share one DEFAULT_CONVENTION object."""
    from galspectra.photometry import synthetic
    from galspectra.pca import metrics
    assert metrics.DEFAULT_CONVENTION is synthetic.DEFAULT_CONVENTION


@pytest.mark.skipif(not LGAL_FILTERS.exists(), reason="L-GALAXIES filter directory absent")
@pytest.mark.parametrize("convention", ["energy", "photon"])
def test_band_operator_matches_production_under_both_conventions(real_library, convention):
    """Whichever convention is selected, the two paths must still agree exactly."""
    from galspectra.photometry.filter_sets import load_filter_set
    from galspectra.photometry.synthetic import compute_ab_magnitudes

    wave, X, _params, _names = real_library
    filters = load_filter_set("lgal_sdss")
    op = BandOperator(wave, filters, convention=convention)

    for i in (0, 137, 600, 1199):
        ref = compute_ab_magnitudes(wave, X[i], filters, convention=convention)
        fast = op.magnitudes(X[i])
        for band in op.names:
            assert abs(ref[band] - fast[op.index(band)]) < 1e-10


@pytest.mark.skipif(not LGAL_FILTERS.exists(), reason="L-GALAXIES filter directory absent")
def test_photon_counting_is_bluer_for_old_populations(real_library):
    """Photon counting gives a bluer g-r than energy weighting for old populations."""
    from galspectra.photometry.filter_sets import load_filter_set
    from galspectra.photometry.synthetic import compute_ab_magnitudes

    wave, X, params, _names = real_library
    filters = load_filter_set("lgal_sdss")

    ages, logz = params[:, 0], params[:, 1]
    old = np.where((ages > 5.0) & np.isclose(logz, 0.0))[0]
    assert old.size, "no old solar-metallicity SSP in the library"

    i = int(old[0])
    e = compute_ab_magnitudes(wave, X[i], filters, convention="energy")
    p = compute_ab_magnitudes(wave, X[i], filters, convention="photon")
    assert (e["g"] - e["r"]) > (p["g"] - p["r"]) + 0.005


@pytest.mark.skipif(not (ARCHIVE / "lgalaxies_sed_coeffs_bc03.npz").exists(),
                    reason="pre-migration archive absent")
def test_migration_moved_photometry_and_nothing_else():
    """pca_coeffs is bit-identical across the energy->photon migration; magnitudes shift."""
    old = np.load(ARCHIVE / "lgalaxies_sed_coeffs_bc03.npz", allow_pickle=True)
    new = np.load(PROJECT_ROOT / "data" / "lgalaxies_sed_coeffs_bc03.npz", allow_pickle=True)

    assert str(new["convention"]) == "photon"
    assert np.array_equal(old["pca_coeffs"], new["pca_coeffs"]), (
        "PCA coefficients moved across a photometry-only change")

    shifts = {}
    for b in "ugriz":
        o = np.asarray(old[f"synth_mag_{b}"], dtype=float)
        n = np.asarray(new[f"synth_mag_{b}"], dtype=float)
        ok = np.isfinite(o) & np.isfinite(n)
        shifts[b] = float(np.median(n[ok] - o[ok]))
        assert -0.03 < shifts[b] < 0.0, f"{b}: unexpected shift {shifts[b]:+.4f}"
    assert shifts["g"] < shifts["z"], "the shift should be largest in g and smallest in z"
