"""
L-GALAXIES 2020 data reader.

Loads the pre-processed galaxy sample (.npy structured array) and the
SFH timing table (Database_SFH_table.fits) that provides lookback times
and bin widths for each snapshot's SFH bins.
"""

from pathlib import Path
import numpy as np

try:
    from astropy.io import fits as _fits
    _ASTROPY_AVAILABLE = True
except ImportError:
    _ASTROPY_AVAILABLE = False


def load_sample(npy_path):
    """Load a pre-processed L-GALAXIES galaxy sample (.npy structured array).

    Created by main_lgals.py after resolution cuts (Mvir >= 20*m_DM,
    StellarMass >= StellarMassRes). Masses in Msun, lengths in Mpc.
    """
    G = np.load(npy_path, allow_pickle=True)
    return G


def load_sfh_table(fits_path):
    """Load the SFH timing table (Database_SFH_table.fits).

    Maps (SNAPNUM, BIN) -> lookback time and bin width in years. Columns:
    SNAPNUM, BIN (1-based), LOOKBACKTIME, DT, NBINS, T_FROM, T_TO (yr).
    """
    if not _ASTROPY_AVAILABLE:
        raise ImportError(
            "astropy is required to read the SFH FITS table. "
            "Install it with: pip install astropy"
        )
    with _fits.open(fits_path) as hdul:
        data = hdul[1].data.copy()
    return data


def get_snap_sfh_times(sfh_table, snap):
    """Lookback times and bin widths (Gyr) for all SFH bins at one snapshot.

    Returns (lookbacktime_gyr, dt_gyr), ordered by BIN (1=oldest, N=newest).
    """
    mask = sfh_table["SNAPNUM"] == snap
    if not np.any(mask):
        raise ValueError(f"Snapshot {snap} not found in SFH table.")

    order = np.argsort(sfh_table["BIN"][mask])  # ascending BIN (1 = oldest)
    lt = sfh_table["LOOKBACKTIME"][mask][order] / 1e9  # yr -> Gyr
    dt = sfh_table["DT"][mask][order] / 1e9

    return lt, dt
