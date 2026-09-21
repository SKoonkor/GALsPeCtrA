"""
Mass-explicit, flux-linear spectral basis.

Weighting is applied before the fit and divided back out afterwards, so
`components`/`mu_eff` are in flux units. The mean term must scale with the
total formed mass M, not just be added once — `reconstruct(C, M)` takes both:

    L(λ) = M · μ_eff(λ) + Σ_k C_k B_k(λ)     C_k = Σ_i m_i c_ik,  M = Σ_i m_i

Both terms are linear in the m_i, so composition is exact by construction.
Storage cost is N_pc + 1 numbers per galaxy (see `bytes_per_galaxy`).
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
    wave : (N_wave,) Å
    mu_eff : (N_wave,) mean term in flux units, per unit formed mass
    components : (N_pc, N_wave) basis vectors, flux units
    weights : (N_wave,) diagonal weighting used during the fit
    explained_variance_ratio : (N_pc,) — component count is chosen from an
        error-vs-bytes curve, not this; see `documents/error_vs_bytes.md`
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

        X : (N_ssp, N_wave) SSP spectra per unit formed stellar mass
            (unchecked — inconsistent rows silently make the mass term meaningless).
        weight_scheme : see `galspectra.pca.weighting`.
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

        # fit in weighted space, then divide the weights back out: B = B̃·diag(1/w)
        Xw = X * w
        pca = PCA(n_components=n_components, svd_solver=svd_solver,
                  random_state=random_state)
        pca.fit(Xw)

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
        """Project spectra (per unit formed mass) onto the basis.

        Returns (N, n_pc) coefficients; `reconstruct(c, 1.0)` is the rank-n_pc
        approximation of the input. Done in weighted space, where the
        components are orthonormal.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != self.n_wave:
            raise ValueError(f"X has {X.shape[1]} wavelengths, basis has {self.n_wave}")
        Xw_centred = X * self.weights - self.mu_eff * self.weights
        Btilde = self.components * self.weights[None, :]
        return Xw_centred @ Btilde.T

    def reconstruct(self, coeffs, mass):
        """Reconstruct flux from coefficients and total formed stellar mass.

        coeffs : (n_pc,) or (N, n_pc) — C_k = Σ_i m_i c_ik (mass-weighted sums).
        mass : scalar or (N,) — M = Σ_i m_i, same units as coeffs.
        Returns (N_wave,) or (N, N_wave) flux.
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
        """Reconstruct with the mean term unscaled — the bug this design prevents.

        Test-only; never use for science.
        """
        C = np.atleast_2d(np.asarray(coeffs, dtype=float))
        out = self.mu_eff[None, :] + C @ self.components
        return out[0] if np.asarray(coeffs).ndim == 1 else out

    def compose(self, coeffs_per_ssp, masses):
        """Accumulate a composite population: returns (C, M).

        coeffs_per_ssp : (N_bins, n_pc) unit-mass coefficients. masses : (N_bins,).
        """
        c = np.asarray(coeffs_per_ssp, dtype=float)
        m = np.asarray(masses, dtype=float)
        if c.shape[0] != m.size:
            raise ValueError(f"{c.shape[0]} coefficient rows vs {m.size} masses")
        return m @ c, float(m.sum())

    # ── accounting ──────────────────────────────────────────────────────────

    def bytes_per_galaxy(self, dtype=np.float32, store_mass=True):
        """Storage cost per galaxy, counting the mass term (N_pc + 1 by default)."""
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
            format_version=np.array("galspectra.SpectralBasis/1"),  # guards against a stale/wrong-shape file
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
        """A view of the same fit with fewer components (PCA components are
        nested, so truncating is exact — no need to refit)."""
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
