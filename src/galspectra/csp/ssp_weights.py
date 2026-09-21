"""
Star-formation histories as a mass-weight matrix over the SSP library.

Bilinear interpolation is itself linear in the library, so
L_gal(λ) = Σ_bins m_bin · S(λ; age_bin, Z_bin) reduces to a single matrix W
of shape (N_gal, N_ssp):

    galaxy spectra     = W @ X
    galaxy band fluxes = W @ (X @ P.T)
    W.sum(axis=1)      = total formed stellar mass (M_total)

Truth and any candidate basis then share one code path (W @ X vs W @ X̂), so
per-galaxy error is exactly the per-SSP error propagated through W.

Deviates from `csp/interpolator.py`, which extrapolates off the age/Z grid
(`fill_value=None`): this module clamps instead and reports the count. Nil
effect on the Millennium-I z=0 sample (every bin lies inside the grid).
"""

from __future__ import annotations

import numpy as np

__all__ = ["SSPGridIndex", "bilinear_weights", "build_weight_matrix"]


class SSPGridIndex:
    """Maps (age, logzsol) onto row indices of a rectangular SSP library.

    params : (N_ssp, 2) library parameter table
    param_names : must contain 'tage' and 'logzsol'

    Requires a complete rectangular grid (checked) — a missing corner would
    make bilinear interpolation silently wrong.
    """

    def __init__(self, params, param_names):
        names = [str(n) for n in param_names]
        try:
            a_col = names.index("tage")
            z_col = names.index("logzsol")
        except ValueError as exc:
            raise ValueError(f"param_names must contain 'tage' and 'logzsol', got {names}") from exc

        params = np.asarray(params, dtype=float)
        ages = params[:, a_col]
        zs = params[:, z_col]

        self.ages = np.unique(ages)
        self.zsol = np.unique(zs)
        n_a, n_z = self.ages.size, self.zsol.size
        if n_a * n_z != params.shape[0]:
            raise ValueError(
                f"library is not a complete grid: {n_a} ages x {n_z} metallicities "
                f"= {n_a * n_z}, but there are {params.shape[0]} spectra"
            )

        ia = np.searchsorted(self.ages, ages)
        iz = np.searchsorted(self.zsol, zs)
        self.row_of = np.full((n_a, n_z), -1, dtype=np.int64)
        self.row_of[ia, iz] = np.arange(params.shape[0])
        if np.any(self.row_of < 0):
            raise ValueError("library grid has holes; bilinear interpolation is unsafe")

        self.n_ssp = int(params.shape[0])

    def __repr__(self):
        return (f"SSPGridIndex({self.ages.size} ages "
                f"{self.ages.min():.4g}-{self.ages.max():.4g} Gyr, "
                f"{self.zsol.size} metallicities "
                f"{self.zsol.min():.3f}-{self.zsol.max():.3f} logzsol)")


def _bracket(grid, values):
    """Lower index and interpolation fraction, clamped to the grid.

    Returns (i, t, n_clamped); points outside the grid clamp to the nearest
    edge (t=0 or 1) rather than extrapolating.
    """
    n = grid.size
    if n < 2:
        raise ValueError("need at least 2 grid points to interpolate")
    values = np.asarray(values, dtype=float)
    n_clamped = int(np.sum((values < grid[0]) | (values > grid[-1])))
    i = np.clip(np.searchsorted(grid, values, side="right") - 1, 0, n - 2)
    t = (values - grid[i]) / (grid[i + 1] - grid[i])
    return i, np.clip(t, 0.0, 1.0), n_clamped


def bilinear_weights(index, ages, zsol):
    """Four (row, weight) pairs per input point.

    Returns rows (N,4) int64 library indices, wts (N,4) float64 weights
    (each row sums to 1), and n_clamped per axis.
    """
    ia, ta, n_ca = _bracket(index.ages, ages)
    iz, tz, n_cz = _bracket(index.zsol, zsol)

    rows = np.stack([
        index.row_of[ia, iz],
        index.row_of[ia + 1, iz],
        index.row_of[ia, iz + 1],
        index.row_of[ia + 1, iz + 1],
    ], axis=1)
    wts = np.stack([
        (1.0 - ta) * (1.0 - tz),
        ta * (1.0 - tz),
        (1.0 - ta) * tz,
        ta * tz,
    ], axis=1)
    return rows, wts, {"age": n_ca, "logzsol": n_cz}


def build_weight_matrix(index, sfh_records, components=("disk", "bulge"), dtype=np.float64):
    """Assemble the (N_gal, N_ssp) mass-weight matrix for a galaxy sample.

    sfh_records : dicts as returned by `galspectra.lgalaxies.extract_sfh`
        ('age_Gyr', '<component>_mass', 'logzsol_<component>' per component).
    components : disk and bulge are tracked separately (different metallicities).

    Returns (W, info): W.sum(axis=1) is total formed stellar mass per galaxy;
    info carries clamping counts and zero-mass bins skipped (dropped rather
    than interpolated at zero weight, since L-GALAXIES writes an undefined
    metallicity there).
    """
    n_gal = len(sfh_records)
    W = np.zeros((n_gal, index.n_ssp), dtype=dtype)

    n_clamped = {"age": 0, "logzsol": 0}
    n_skipped = 0
    n_bins_used = 0

    for g, rec in enumerate(sfh_records):
        ages_all, zs_all, mass_all = [], [], []
        for comp in components:
            m = np.asarray(rec[f"{comp}_mass"], dtype=float)
            keep = m > 0
            if not keep.any():
                continue
            ages_all.append(np.asarray(rec["age_Gyr"], dtype=float)[keep])
            zs_all.append(np.asarray(rec[f"logzsol_{comp}"], dtype=float)[keep])
            mass_all.append(m[keep])
            n_skipped += int((~keep).sum())

        if not mass_all:
            continue

        ages = np.concatenate(ages_all)
        zs = np.concatenate(zs_all)
        mass = np.concatenate(mass_all)
        n_bins_used += mass.size

        rows, wts, nc = bilinear_weights(index, ages, zs)
        n_clamped["age"] += nc["age"]
        n_clamped["logzsol"] += nc["logzsol"]

        np.add.at(W[g], rows.ravel(), (wts * mass[:, None]).ravel())

    return W, {
        "n_galaxies": n_gal,
        "n_bins_used": n_bins_used,
        "n_zero_mass_bins_skipped": n_skipped,
        "n_clamped": n_clamped,
        "n_ssp": index.n_ssp,
    }
