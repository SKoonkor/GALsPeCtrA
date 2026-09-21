"""
Star-formation histories expressed as a mass-weight matrix over the SSP library.

The idea
--------
Building a composite population means evaluating

    L_gal(λ) = Σ_bins m_bin · S(λ; age_bin, Z_bin)

where S is the library, bilinearly interpolated in (age, Z). Bilinear interpolation
is itself linear in the library, so S(age, Z) is a weighted sum of at most four
library entries. Substituting that back gives

    L_gal(λ) = Σ_ssp W_gal,ssp · X_ssp(λ)          W = mass-weighted bilinear weights

so an entire galaxy sample becomes a single matrix W of shape (N_gal, N_ssp), and
**every** derived quantity is one matrix product away:

    galaxy spectra     = W @ X
    galaxy band fluxes = W @ (X @ P.T)
    row sums of W      = total formed stellar mass

The last point matters for this project specifically: `W.sum(axis=1)` *is* M_total,
so the mass term that the stored `pca_coeffs` product currently omits falls out of
the same construction that produces the coefficients.

Why this is worth doing rather than looping
-------------------------------------------
It is not only faster. It makes the truth and every candidate basis share one code
path: the truth is `W @ X`, a reconstruction is `W @ X̂`, and the difference between
them can only come from `X̂ − X`. There is no opportunity for the two sides of the
comparison to drift apart through separate loops, and the per-galaxy error is the
per-SSP error propagated through exactly the weights that define the galaxy.

Deviation from the production path, stated plainly
--------------------------------------------------
`csp/interpolator.py` builds `RegularGridInterpolator(..., fill_value=None)`, which
**extrapolates** off the age and metallicity grid rather than failing. This module
**clamps** instead, and reports how many bins were clamped. For the Millennium-I
z=0 sample the difference is nil — every SFH bin lies inside the grid, and the
metallicities are already clipped upstream — but clamping is the right behaviour for
an offline experiment and the count makes any future sample's exposure visible.
"""

from __future__ import annotations

import numpy as np

__all__ = ["SSPGridIndex", "bilinear_weights", "build_weight_matrix"]


class SSPGridIndex:
    """Maps (age, logzsol) onto row indices of a rectangular SSP library.

    Parameters
    ----------
    params : (N_ssp, 2) — the library's parameter table
    param_names : sequence of str — must contain 'tage' and 'logzsol'

    The library must be a complete rectangular grid; a missing corner would make
    bilinear interpolation silently wrong, so it is checked.
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

    Returns (i, t, n_clamped) with the point lying between grid[i] and grid[i+1]
    and t in [0, 1]. Points outside the grid land on the nearest edge with t at 0
    or 1, so the result is the boundary value rather than an extrapolation.
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

    Returns
    -------
    rows : (N, 4) int64 — library row indices
    wts  : (N, 4) float64 — bilinear weights, each row summing to 1
    n_clamped : dict — how many points fell outside each axis
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

    Parameters
    ----------
    index : SSPGridIndex
    sfh_records : sequence of dicts as returned by
        `galspectra.lgalaxies.extract_sfh` — each with 'age_Gyr', and
        '<component>_mass' / 'logzsol_<component>' for each requested component.
    components : which stellar components to sum. Disk and bulge are tracked
        separately by L-GALAXIES because they carry different metallicities; both
        contribute to the total light.

    Returns
    -------
    W : (N_gal, N_ssp) — W.sum(axis=1) is the total formed stellar mass per galaxy
    info : dict — clamping counts and the number of bins with non-positive mass

    Zero-mass bins are dropped rather than contributing a zero-weighted
    interpolation, so a bin with an undefined metallicity (which L-GALAXIES writes
    when no stars formed) cannot poison the result.
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
