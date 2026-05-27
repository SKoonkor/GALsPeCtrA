"""
Filter transmission curve loader.

Reads filter files from SpecPhotTables/Filters/ in the L-GALAXIES distribution.
Each filter file has:
  Line 1:  integer — number of data points
  Lines 2+: wavelength_Ang  transmission (0-1)

Also provides a registry of common filters with their paths.
"""

from pathlib import Path
import numpy as np

# SDSS filter short names (matches L-GALAXIES Mag/MagDust band order)
SDSS_FILTER_NAMES = ["u", "g", "r", "i", "z"]

# Filter filenames in SpecPhotTables/Filters/ for SDSS ugriz
SDSS_FILTER_FILES = {
    "u": "u_band_trans.dat",
    "g": "g_band_trans.dat",
    "r": "r_band_trans.dat",
    "i": "i_band_trans.dat",
    "z": "z_band_trans.dat",
}

# Additional available filters
EXTRA_FILTER_FILES = {
    "GALEX_FUV":  "GALEX_FUV.txt",
    "GALEX_NUV":  "GALEX_NUV.txt",
    "HST_F435W":  "ACS_WFC_F435W.txt",
    "HST_F606W":  "ACS_WFC_F606W.txt",
    "HST_F814W":  "ACS_WFC_F814W.txt",
    "WFC3_F160W": "WFC3_IR_F160W.txt",
}


def load_filter(filter_path):
    """
    Load a filter transmission curve.

    Parameters
    ----------
    filter_path : str or Path
        Full path to the filter file.

    Returns
    -------
    wave : (N,) array — wavelength in Å
    trans : (N,) array — transmission (0–1)
    """
    filter_path = Path(filter_path)
    if not filter_path.exists():
        raise FileNotFoundError(f"Filter file not found: {filter_path}")

    with open(filter_path) as f:
        first_line = f.readline().strip()
        try:
            n_points = int(first_line)
            # First line is a count; data follows
            data = np.loadtxt(f)
        except ValueError:
            # First line is already data (no count header)
            row0 = np.fromstring(first_line, sep=' ')
            rest = np.loadtxt(f)
            data = np.vstack([row0, rest])

    wave  = data[:, 0].astype(float)
    trans = data[:, 1].astype(float)

    # Normalise to [0,1] if not already
    if trans.max() > 1.0:
        trans = trans / trans.max()

    return wave, trans


def load_sdss_filters(filter_dir):
    """
    Load all 5 SDSS ugriz filter curves from a given directory.

    Parameters
    ----------
    filter_dir : str or Path

    Returns
    -------
    filters : dict {band_name: (wave_Ang, transmission)}
    """
    filter_dir = Path(filter_dir)
    filters = {}
    for name, fname in SDSS_FILTER_FILES.items():
        wave, trans = load_filter(filter_dir / fname)
        filters[name] = (wave, trans)
    return filters


def load_filters(filter_dir, names):
    """
    Load an arbitrary set of filters by name.

    Parameters
    ----------
    filter_dir : str or Path
    names : list of str — keys from SDSS_FILTER_FILES or EXTRA_FILTER_FILES

    Returns
    -------
    filters : dict {name: (wave_Ang, transmission)}
    """
    all_files = {**SDSS_FILTER_FILES, **EXTRA_FILTER_FILES}
    filter_dir = Path(filter_dir)
    filters = {}
    for name in names:
        if name not in all_files:
            raise ValueError(f"Unknown filter '{name}'. "
                             f"Available: {list(all_files.keys())}")
        wave, trans = load_filter(filter_dir / all_files[name])
        filters[name] = (wave, trans)
    return filters
