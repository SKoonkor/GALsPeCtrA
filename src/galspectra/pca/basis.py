"""
A spectral basis that is explicitly linear in flux and explicit about mass.

The two structural guarantees
-----------------------------
**1. Linear in flux.** The stored basis lives in flux units. Weighting is applied
before the decomposition and divided back out of the basis vectors afterwards, so
what comes back is an honest set of spectra, not a set of spectra-in-weighted-units
that only reconstruct correctly if the caller remembers the weights.

**2. The mean term is carried by mass, not hidden.** A PCA of centred data
reconstructs as `mean + c·B`. For a single SSP of unit mass that is right. For a
composite population it is wrong unless the mean term is scaled by the total formed
mass, because

    Σ_i m_i (μ + c_i·B)  =  (Σ_i m_i) μ  +  (Σ_i m_i c_i)·B

The existing production code gets this right, but implicitly: it divides the
coefficients by the total mass before reconstructing and multiplies back afterwards
(the `/M … ×M` sandwich in `scripts/process_lgalaxies.py`). It works, and it is
fragile — the mass never appears in the stored product, so `pca_coeffs` alone is not
enough to reconstruct a galaxy and every consumer has to re-derive M from the SFH.

Here the mass is a first-class part of the representation. `reconstruct(C, M)` takes
both. The honest storage cost is therefore **N_pc + 1 numbers per galaxy**, and
`bytes_per_galaxy()` reports it that way.

    L(λ) = M · μ_eff(λ)  +  Σ_k C_k B_k(λ)          C_k = Σ_i m_i c_ik ,  M = Σ_i m_i

Both terms are linear in the m_i, so composition is exact by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from galspectra.pca.weighting import build_weights, check_weights

__all__ = ["SpectralBasis", "fit_basis"]


@dataclass
class SpectralBasis:
    """A mass-explicit, flux-linear spectral basis.

    Attributes
    ----------
    wave : (N_wave,) — Å
    mu_eff : (N_wave,) — the mean term in flux units, per unit formed mass
    components : (N_pc, N_wave) — basis vectors in flux units
    weights : (N_wave,) — the diagonal weighting used during the fit, retained so
        `transform` works in the same space the fit was performed in, and so the
        scheme is auditable from the stored product
    explained_variance_ratio : (N_pc,) — reported, but see the note in
        `documents/error_vs_bytes.md`: component count is chosen from the
        error-vs-bytes curve, never from a variance threshold
    meta : dict — provenance
    """

    wave: np.ndarray
    mu_eff: np.ndarray
    components: np.ndarray
    weights: np.ndarray
    explained_variance_ratio: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def fit(cls, wave, X, n_components, weight_scheme="inverse_std",
            weight_kwargs=None, svd_solver="full", random_state=0):
        """Fit a basis to an SSP library.

        Parameters
        ----------
        wave : (N_wave,) — Å
        X : (N_ssp, N_wave) — SSP spectra **per unit formed stellar mass**.
            Nothing here checks that; if the rows carry different mass
            normalisations the basis is still self-consistent but the mass term
            loses its meaning.
        n_components : int
        weight_scheme : str — see `galspectra.pca.weighting`
        weight_kwargs : dict — extra arguments to `build_weights`

        Returns
        -------
        SpectralBasis
        """
        from sklearn.decomposition import PCA

        wave = np.asarray(wave, dtype=float)
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != wave.size:
            raise ValueError(f"X {X.shape} incompatible with {wave.size} wavelengths")

        n_max = min(X.shape[0], X.shape[1])
        if not 1 <= n_components <= n_max:
            raise ValueError(
                f"n_components={n_components} outside 1..{n_max} "
                f"({X.shape[0]} spectra, {X.shape[1]} wavelengths)"
            )

        w, wmeta = build_weights(weight_scheme, X, wave, **(weight_kwargs or {}))

        # ── the fit happens in weighted space ──────────────────────────────
        Xw = X * w                                  # broadcast over wavelength only
        pca = PCA(n_components=n_components, svd_solver=svd_solver,
                  random_state=random_state)
        pca.fit(Xw)

        # ── and the weights come straight back out ─────────────────────────
        # B = B̃ · diag(1/w) and μ_eff = m̄ · diag(1/w) put the basis in flux units.
        mu_eff = pca.mean_ / w
        components = pca.components_ / w[None, :]

        return cls(
            wave=wave,
            mu_eff=mu_eff,
            components=components,
            weights=w,
            explained_variance_ratio=pca.explained_variance_ratio_.copy(),
            meta={
                "n_components": int(n_components),
                "n_wave": int(wave.size),
                "n_ssp": int(X.shape[0]),
                "wave_min": float(wave.min()),
                "wave_max": float(wave.max()),
                "weighting": wmeta,
            },
        )

    # ── use ─────────────────────────────────────────────────────────────────

    @property
    def n_components(self):
        return self.components.shape[0]

    @property
    def n_wave(self):
        return self.wave.size

    def transform(self, X):
        """Project spectra onto the basis. Rows of X are per unit formed mass.

        Returns (N, n_pc) coefficients c such that `reconstruct(c, 1.0)` is the
        rank-n_pc approximation of the input.

        The projection is done in weighted space, where the components are
        orthonormal. Projecting in flux space would need the metric diag(w²) and is
        an easy place to introduce a subtle inconsistency, so it is not offered.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != self.n_wave:
            raise ValueError(f"X has {X.shape[1]} wavelengths, basis has {self.n_wave}")
        Xw_centred = X * self.weights - self.mu_eff * self.weights
        Btilde = self.components * self.weights[None, :]
        return Xw_centred @ Btilde.T

    def reconstruct(self, coeffs, mass):
        """Reconstruct flux from coefficients and total formed stellar mass.

        Parameters
        ----------
        coeffs : (n_pc,) or (N, n_pc) — C_k = Σ_i m_i c_ik, i.e. already
            mass-weighted sums over the star-formation history.
        mass : scalar or (N,) — M = Σ_i m_i, the total formed stellar mass in the
            same units the coefficients were accumulated in.

        Returns
        -------
        (N_wave,) or (N, N_wave) flux

        This is the whole reconstruction. There is no per-spectrum normalisation
        step and no place to put one; `tests/test_pca_basis_linearity.py` asserts
        that composition is exact to machine precision.
        """
        coeffs = np.asarray(coeffs, dtype=float)
        mass = np.asarray(mass, dtype=float)
        single = coeffs.ndim == 1
        C = np.atleast_2d(coeffs)
        M = np.atleast_1d(mass)

        if C.shape[1] != self.n_components:
            raise ValueError(
                f"coeffs have {C.shape[1]} components, basis has {self.n_components}"
            )
        if M.size == 1:
            M = np.repeat(M, C.shape[0])
        elif M.size != C.shape[0]:
            raise ValueError(f"mass has {M.size} entries, coeffs have {C.shape[0]} rows")

        out = M[:, None] * self.mu_eff[None, :] + C @ self.components
        return out[0] if single else out

    def reconstruct_without_mass(self, coeffs):
        """Reconstruct with the mean term *unscaled* — i.e. the bug this design prevents.

        Exists only so `tests/test_pca_basis_linearity.py` can demonstrate that the
        mass term is load-bearing rather than decorative. Never use it for science.
        """
        C = np.atleast_2d(np.asarray(coeffs, dtype=float))
        out = self.mu_eff[None, :] + C @ self.components
        return out[0] if np.asarray(coeffs).ndim == 1 else out

    def compose(self, coeffs_per_ssp, masses):
        """Accumulate a composite population: returns (C, M).

        `coeffs_per_ssp` is (N_bins, n_pc) unit-mass coefficients, `masses` is
        (N_bins,). This is the only place the SFH sum is written down, so the
        convention cannot drift between call sites.
        """
        c = np.asarray(coeffs_per_ssp, dtype=float)
        m = np.asarray(masses, dtype=float)
        if c.shape[0] != m.size:
            raise ValueError(f"{c.shape[0]} coefficient rows vs {m.size} masses")
        return m @ c, float(m.sum())

    # ── accounting ──────────────────────────────────────────────────────────

    def bytes_per_galaxy(self, dtype=np.float32, store_mass=True):
        """Storage cost per galaxy, counting the mass term.

        The project has historically quoted "50 floats per galaxy". With the mass
        term it is 51, because `pca_coeffs` alone cannot reconstruct a galaxy.
        """
        itemsize = np.dtype(dtype).itemsize
        return int((self.n_components + (1 if store_mass else 0)) * itemsize)

    def summary(self):
        evr = self.explained_variance_ratio
        return {
            "n_components": self.n_components,
            "n_wave": self.n_wave,
            "wave_min": float(self.wave.min()),
            "wave_max": float(self.wave.max()),
            "weighting": self.meta.get("weighting", {}).get("scheme"),
            "bytes_per_galaxy_f32": self.bytes_per_galaxy(np.float32),
            "cumulative_variance": None if evr is None else float(evr.sum()),
        }

    # ── persistence ─────────────────────────────────────────────────────────

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            wave=self.wave,
            mu_eff=self.mu_eff,
            components=self.components,
            weights=self.weights,
            explained_variance_ratio=(
                np.array([]) if self.explained_variance_ratio is None
                else self.explained_variance_ratio
            ),
            meta=np.array(self.meta, dtype=object),
            # A version marker, because the committed binary grid has none and a
            # stale file of the right shape currently loads silently.
            format_version=np.array("galspectra.SpectralBasis/1"),
        )
        return path

    @classmethod
    def load(cls, path):
        d = np.load(Path(path), allow_pickle=True)
        version = str(d["format_version"]) if "format_version" in d else "(none)"
        if version != "galspectra.SpectralBasis/1":
            raise ValueError(f"{path}: unrecognised basis format '{version}'")
        evr = d["explained_variance_ratio"]
        return cls(
            wave=d["wave"],
            mu_eff=d["mu_eff"],
            components=d["components"],
            weights=check_weights(d["weights"], d["wave"].size),
            explained_variance_ratio=None if evr.size == 0 else evr,
            meta=d["meta"].item(),
        )

    def truncate(self, n_components):
        """A view of the same fit with fewer components.

        PCA components are nested, so truncating is exact — there is no need to
        refit for every point on an error-vs-bytes curve.
        """
        if not 1 <= n_components <= self.n_components:
            raise ValueError(f"n_components must be in 1..{self.n_components}")
        evr = (None if self.explained_variance_ratio is None
               else self.explained_variance_ratio[:n_components])
        meta = dict(self.meta)
        meta["n_components"] = int(n_components)
        meta["truncated_from"] = int(self.n_components)
        return SpectralBasis(
            wave=self.wave,
            mu_eff=self.mu_eff,
            components=self.components[:n_components],
            weights=self.weights,
            explained_variance_ratio=evr,
            meta=meta,
        )


def fit_basis(wave, X, n_components, weight_scheme="inverse_std", **kw):
    """Convenience wrapper around SpectralBasis.fit."""
    return SpectralBasis.fit(wave, X, n_components, weight_scheme=weight_scheme, **kw)
