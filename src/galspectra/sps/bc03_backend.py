"""
BC03 SSP backend.

Reads Bruzual & Charlot 2003 FullSED files from the L-GALAXIES SpecPhotTables directory.

File format (4 columns per row):
    age_yr   metallicity_Z   wavelength_Ang   flux

221 age steps × 1221 wavelength points per metallicity file.
6 metallicity files: Z = 0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05

Exposes the same (wave, flux) interface as fsps_backend.py so the two backends
can be swapped transparently in generate_seds.py.
"""

from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# BC03 metallicity grid (Z, not logZ)
BC03_METALLICITIES = np.array([0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05])

# Corresponding filenames
_BC03_FILENAMES = {
    0.0001: "BC03_Chabrier_FullSED_m0.0001.dat",
    0.0004: "BC03_Chabrier_FullSED_m0.0004.dat",
    0.004:  "BC03_Chabrier_FullSED_m0.0040.dat",
    0.008:  "BC03_Chabrier_FullSED_m0.0080.dat",
    0.02:   "BC03_Chabrier_FullSED_m0.0200.dat",
    0.05:   "BC03_Chabrier_FullSED_m0.0500.dat",
}


class BC03Library:
    """
    Lazy-loading wrapper around the BC03 FullSED files.

    Parameters
    ----------
    bc03_dir : str or Path
        Directory containing BC03_Chabrier_FullSED_m*.dat files.
    wave_min, wave_max : float, optional
        Wavelength range to load (Å). Reduces memory if set.
    """

    def __init__(self, bc03_dir, wave_min=None, wave_max=None):
        self.bc03_dir = Path(bc03_dir)
        self.wave_min = wave_min
        self.wave_max = wave_max
        self._cache = {}  # Z -> (ages_gyr, wave, flux_grid)
        self._wave = None  # shared wavelength grid

    def _load_metallicity(self, Z):
        """
        Load one BC03 FullSED file into a (N_age, N_wave) flux array.

        The file is strictly regular: N_ages × N_waves_per_age rows, where
        all rows for one age are contiguous.  We exploit this to avoid any
        Python-level loops over 270 K rows.
        """
        fname = self.bc03_dir / _BC03_FILENAMES[Z]
        if not fname.exists():
            raise FileNotFoundError(f"BC03 file not found: {fname}")

        # Fast read: numpy's C parser on the whole file
        data = np.loadtxt(fname)            # (N_total, 4)

        ages_yr_col  = data[:, 0]
        waves_ang_col = data[:, 2]
        flux_col      = data[:, 3]

        # Determine grid dimensions from first column changes (ages repeat in blocks)
        # The file has N_waves rows per age block — find the first repeated age value
        n_waves_per_age = int(np.searchsorted(ages_yr_col[1:], ages_yr_col[0], side='right')) + 1

        n_total = len(data)
        n_ages  = n_total // n_waves_per_age

        # Extract grids from the first block (ages from column 0 of each block start)
        ages_yr   = ages_yr_col[::n_waves_per_age]          # (N_age,)
        wave_full = waves_ang_col[:n_waves_per_age]          # (N_wave,)

        # Reshape flux to (N_age, N_wave) — pure numpy, no Python loop
        flux_grid = flux_col[:n_ages * n_waves_per_age].reshape(n_ages, n_waves_per_age)

        # Apply wavelength mask
        wave_mask = np.ones(n_waves_per_age, dtype=bool)
        if self.wave_min is not None:
            wave_mask &= (wave_full >= self.wave_min)
        if self.wave_max is not None:
            wave_mask &= (wave_full <= self.wave_max)

        wave_unique = wave_full[wave_mask]
        flux_grid   = flux_grid[:, wave_mask]

        ages_gyr = ages_yr / 1e9   # yr -> Gyr
        return ages_gyr, wave_unique, flux_grid

    def get(self, Z):
        """Return (ages_gyr, wave_Ang, flux_grid) for a given metallicity."""
        if Z not in self._cache:
            self._cache[Z] = self._load_metallicity(Z)
        return self._cache[Z]

    @property
    def wave(self):
        """Shared wavelength array (loads Z=0.02 file as reference)."""
        if self._wave is None:
            _, w, _ = self.get(0.02)
            self._wave = w
        return self._wave


def _nearest_Z(Z_query, Z_grid=BC03_METALLICITIES):
    """Return the nearest BC03 metallicity grid point."""
    return Z_grid[np.argmin(np.abs(Z_grid - Z_query))]


def get_bc03_spectrum(tage_gyr, Z, bc03_dir, lib=None):
    """
    Interpolate BC03 SSP spectrum at a given age and metallicity.

    Parameters
    ----------
    tage_gyr : float
        Stellar population age in Gyr.
    Z : float
        Metallicity (linear, e.g. 0.02 = solar). Clamped to BC03 grid range.
    bc03_dir : str or Path
        Path to directory with BC03_Chabrier_FullSED_*.dat files.
    lib : BC03Library, optional
        Pre-loaded library (avoids re-reading files on repeated calls).

    Returns
    -------
    wave : (N_wave,) array — wavelength in Å
    flux : (N_wave,) array — spectral flux (BC03 internal units; consistent within backend)
    """
    if lib is None:
        lib = BC03Library(bc03_dir)

    # Clamp metallicity to grid
    Z_lo = BC03_METALLICITIES[np.searchsorted(BC03_METALLICITIES, Z, side='right') - 1]
    idx_hi = min(np.searchsorted(BC03_METALLICITIES, Z, side='right'),
                 len(BC03_METALLICITIES) - 1)
    Z_hi = BC03_METALLICITIES[idx_hi]

    ages_lo, wave, flux_lo = lib.get(Z_lo)
    _, _, flux_hi          = lib.get(Z_hi)

    # Interpolate in age for each metallicity bracket
    tage_clamped = np.clip(tage_gyr, ages_lo[0], ages_lo[-1])

    flux_at_Zlo = np.array([np.interp(tage_clamped, ages_lo, flux_lo[:, j])
                             for j in range(len(wave))])
    flux_at_Zhi = np.array([np.interp(tage_clamped, ages_lo, flux_hi[:, j])
                             for j in range(len(wave))])

    # Interpolate in metallicity (linear in log Z if both brackets differ)
    if Z_lo == Z_hi or Z_lo <= 0 or Z_hi <= 0:
        flux = flux_at_Zlo
    else:
        w = (np.log10(Z) - np.log10(Z_lo)) / (np.log10(Z_hi) - np.log10(Z_lo))
        w = np.clip(w, 0.0, 1.0)
        flux = (1.0 - w) * flux_at_Zlo + w * flux_at_Zhi

    return wave, flux


def generate_bc03_seds(param_dict, bc03_dir, wave_min=None, wave_max=None, verbose=True, lib=None):
    """
    Generate BC03 SSP SEDs from a parameter grid.

    Drop-in replacement for `generate_seds()` in `sed/generator.py` but uses
    the BC03 FullSED files instead of FSPS.

    Parameters
    ----------
    param_dict : dict
        Output from sampling/paramgrid.py with keys 'samples' and 'param_names'.
        Expected parameter names: 'tage' (linear Gyr, after ensure_linear_age conversion)
        and 'logzsol' (log10 Z/Zsun).
    bc03_dir : str or Path
    wave_min, wave_max : float, optional — wavelength mask in Å.
    verbose : bool

    Returns
    -------
    dict with keys: 'wave', 'seds', 'params', 'param_names'
    """
    samples     = param_dict["samples"]
    param_names = param_dict["param_names"]
    n_samples   = samples.shape[0]

    tage_idx   = list(param_names).index("tage")
    logzsol_idx = list(param_names).index("logzsol")

    if lib is None:
        lib = BC03Library(bc03_dir, wave_min=wave_min, wave_max=wave_max)

    # Determine output wavelength grid from first sample
    row0 = samples[0]
    tage0   = row0[tage_idx]                 # already linear Gyr (ensure_linear_age in paramgrid)
    Z0      = 0.02 * (10 ** row0[logzsol_idx])
    wave, _ = get_bc03_spectrum(tage0, Z0, bc03_dir, lib)

    seds = np.zeros((n_samples, len(wave)))

    for i, row in enumerate(samples):
        tage_gyr = row[tage_idx]             # linear Gyr (ensure_linear_age already applied)
        Z        = 0.02 * (10 ** row[logzsol_idx])  # logzsol -> Z

        _, flux = get_bc03_spectrum(tage_gyr, Z, bc03_dir, lib)
        seds[i] = flux

        if verbose and i % max(1, n_samples // 100) == 0:
            print(f"\rProgress: {int(i / n_samples * 100)}%", end="")

    if verbose:
        print("\rProgress: 100%")

    return {
        "wave":        wave,
        "seds":        seds,
        "params":      samples,
        "param_names": param_names,
    }
