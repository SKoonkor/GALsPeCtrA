"""
L-GALAXIES SFH extractor.

Converts raw sfh_* fields into a clean per-galaxy SFH dict for the CSP builder.

sfh_DiskMass/BulgeMass/MetalsDiskMass are already in Msun in the .npy sample
(the 1e10/h conversion is applied upstream). Metallicity per bin:

    Z_j = sum(sfh_MetalsDiskMass[j, :]) / sfh_DiskMass[j]   (3 channels: SNII, SNIa, AGB)
"""

import numpy as np

# Solar metallicity (Asplund et al. 2009)
Z_SUN = 0.0142

# Minimum Z and logzsol values to avoid log(0) and out-of-grid interpolation
Z_MIN    = 1e-4
Z_MAX    = 0.05
LOGZSOL_MIN = np.log10(Z_MIN / 0.02)   # ~ -2.30
LOGZSOL_MAX = np.log10(Z_MAX / 0.02)   # ~ 0.40


def extract_sfh(galaxy, sfh_table, hubble_h=0.673):
    """Extract a structured SFH dict for one galaxy.

    galaxy : one row from load_sample(). sfh_table : from load_sfh_table().

    Returns dict: n_bins, age_Gyr, dt_Gyr, disk_mass, bulge_mass (Msun),
    Z_disk, Z_bulge (linear), logzsol_disk, logzsol_bulge
    (log10(Z/0.02), clamped to [LOGZSOL_MIN, LOGZSOL_MAX]).
    """
    snap    = int(galaxy["SnapNum"])
    n_bins  = int(galaxy["sfh_ibin"])  # index of the highest active bin; bins are 0..sfh_ibin
    n_active = n_bins + 1

    from .reader import get_snap_sfh_times
    lt_gyr, dt_gyr = get_snap_sfh_times(sfh_table, snap)

    n_table = len(lt_gyr)
    if n_active > n_table:
        n_active = n_table  # shouldn't happen, but guard

    lt_gyr  = lt_gyr[:n_active]
    dt_gyr  = dt_gyr[:n_active]

    disk_mass  = np.array(galaxy["sfh_DiskMass"][:n_active],  dtype=float)
    bulge_mass = np.array(galaxy["sfh_BulgeMass"][:n_active], dtype=float)

    # metal mass per bin, shape (n_active, 3) -- 3 enrichment channels (SNII, SNIa, AGB)
    metal_disk  = np.array(galaxy["sfh_MetalsDiskMass"][:n_active,  :], dtype=float)
    metal_bulge = np.array(galaxy["sfh_MetalsBulgeMass"][:n_active, :], dtype=float)

    total_metals_disk  = metal_disk.sum(axis=1)
    total_metals_bulge = metal_bulge.sum(axis=1)

    # Z = metal_mass / stellar_mass. sfh_DiskMass is INITIAL formed mass (not
    # surviving), correct for SSP convolution; sfh_total/StellarMass ~ 1.7 is
    # the expected stellar-mass return (~40% over a Hubble time, Chabrier IMF).
    safe_disk  = np.where(disk_mass  > 0, disk_mass,  1.0)
    safe_bulge = np.where(bulge_mass > 0, bulge_mass, 1.0)
    Z_disk  = np.where(disk_mass  > 0, total_metals_disk  / safe_disk,  Z_SUN * 0.1)
    Z_bulge = np.where(bulge_mass > 0, total_metals_bulge / safe_bulge, Z_SUN * 0.1)

    Z_disk  = np.clip(Z_disk,  Z_MIN, Z_MAX)
    Z_bulge = np.clip(Z_bulge, Z_MIN, Z_MAX)

    # logzsol = log10(Z/Zsun), Zsun = 0.02 in BC03/FSPS convention
    logzsol_disk  = np.clip(np.log10(Z_disk  / 0.02), LOGZSOL_MIN, LOGZSOL_MAX)
    logzsol_bulge = np.clip(np.log10(Z_bulge / 0.02), LOGZSOL_MIN, LOGZSOL_MAX)

    return {
        "n_bins":         n_active,
        "age_Gyr":        lt_gyr,
        "dt_Gyr":         dt_gyr,
        "disk_mass":      disk_mass,
        "bulge_mass":     bulge_mass,
        "Z_disk":         Z_disk,
        "Z_bulge":        Z_bulge,
        "logzsol_disk":   logzsol_disk,
        "logzsol_bulge":  logzsol_bulge,
    }


def total_sfh_mass(sfh):
    """Sum disk + bulge mass across all SFH bins (~= StellarMass + ICM). For validation."""
    return sfh["disk_mass"].sum() + sfh["bulge_mass"].sum()
