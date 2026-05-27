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
    """
    Load a pre-processed L-GALAXIES galaxy sample.

    The .npy file is created by main_lgals.py after applying resolution cuts
    (Mvir >= 20 * m_DM and StellarMass >= StellarMassRes).  All masses are
    already in Msun and lengths in Mpc.

    Parameters
    ----------
    npy_path : str or Path

    Returns
    -------
    G : numpy structured array, shape (N_galaxies,)
    """
    G = np.load(npy_path, allow_pickle=True)
    return G


def load_sfh_table(fits_path):
    """
    Load the SFH timing lookup table from Database_SFH_table.fits.

    The table maps (SNAPNUM, BIN) -> lookback time and bin width in years.

    Parameters
    ----------
    fits_path : str or Path

    Returns
    -------
    data : astropy FITS record array with columns:
        SNAPNUM        — snapshot number
        BIN            — bin index (1-based)
        LOOKBACKTIME   — lookback time to bin centre (yr)
        DT             — bin width (yr)
        NBINS          — number of original timesteps merged
        T_FROM, T_TO   — bin edges (yr)
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
    """
    Return lookback times and bin widths for all SFH bins at a given snapshot.

    Parameters
    ----------
    sfh_table : FITS record array from load_sfh_table()
    snap : int — snapshot number (e.g. 58 for z=0 in Mil-I Planck)

    Returns
    -------
    lookbacktime_gyr : (N_bins,) array — lookback time to bin centre in Gyr
                       Ordered by BIN index (bin 1 = oldest, bin N = most recent)
    dt_gyr           : (N_bins,) array — bin width in Gyr
    """
    mask = sfh_table["SNAPNUM"] == snap
    if not np.any(mask):
        raise ValueError(f"Snapshot {snap} not found in SFH table.")

    # Sort by BIN index (ascending: 1 = oldest half of lookback time)
    order = np.argsort(sfh_table["BIN"][mask])
    lt = sfh_table["LOOKBACKTIME"][mask][order] / 1e9  # yr -> Gyr
    dt = sfh_table["DT"][mask][order] / 1e9

    return lt, dt
