"""
L-GALAXIES SFH extractor.

Converts the raw sfh_* fields from the galaxy structured array into a
clean per-galaxy SFH dictionary suitable for feeding into the CSP builder.

Unit conversions applied here:
  sfh_DiskMass, sfh_BulgeMass : stored as 10^10 Msun/h internally, but the
      .npy sample file has already been multiplied by 1e10/Hubble_h, so the
      values are in Msun directly.
  sfh_MetalsDiskMass : same conversion as masses (already in Msun in .npy).
  sfh_DiskMass_elements : already in Msun (no h correction in the original code).

The metallicity Z is computed per bin as:
    Z_j = sum(sfh_MetalsDiskMass[j, :]) / sfh_DiskMass[j]
where the sum is over the 3 enrichment channels (SNII, SNIa, AGB).
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
    """
    Extract a structured SFH for one galaxy.

    Parameters
    ----------
    galaxy : numpy structured array row (one galaxy from load_sample())
    sfh_table : FITS record array from load_sfh_table()
    hubble_h : float — Hubble parameter h (used only if mass conversion is needed)

    Returns
    -------
    sfh : dict with keys:
        'n_bins'     — number of active SFH bins
        'age_Gyr'    — (n_bins,) lookback time of each bin centre in Gyr
        'dt_Gyr'     — (n_bins,) bin width in Gyr
        'disk_mass'  — (n_bins,) disk stellar mass per bin in Msun
        'bulge_mass' — (n_bins,) bulge stellar mass per bin in Msun
        'Z_disk'     — (n_bins,) mean metallicity (linear Z) for disk stars per bin
        'Z_bulge'    — (n_bins,) mean metallicity for bulge stars per bin
        'logzsol_disk'  — (n_bins,) log10(Z_disk / 0.02), clamped to [LOGZSOL_MIN, LOGZSOL_MAX]
        'logzsol_bulge' — (n_bins,) same for bulge
    """
    snap    = int(galaxy["SnapNum"])
    n_bins  = int(galaxy["sfh_ibin"])  # number of active bins (0-based max index -> n_bins+1 bins)

    # sfh_ibin is the index of the *highest active bin*; bins are 0..sfh_ibin
    n_active = n_bins + 1

    # Get timing info from SFH table
    from .reader import get_snap_sfh_times
    lt_gyr, dt_gyr = get_snap_sfh_times(sfh_table, snap)

    # The SFH table has n_table_bins entries for this snap; use the first n_active
    n_table = len(lt_gyr)
    if n_active > n_table:
        n_active = n_table  # shouldn't happen, but guard

    lt_gyr  = lt_gyr[:n_active]
    dt_gyr  = dt_gyr[:n_active]

    # Stellar mass per bin (already in Msun in the .npy sample)
    disk_mass  = np.array(galaxy["sfh_DiskMass"][:n_active],  dtype=float)
    bulge_mass = np.array(galaxy["sfh_BulgeMass"][:n_active], dtype=float)

    # Metal mass per bin — shape (n_active, 3) [3 enrichment channels]
    metal_disk  = np.array(galaxy["sfh_MetalsDiskMass"][:n_active,  :], dtype=float)
    metal_bulge = np.array(galaxy["sfh_MetalsBulgeMass"][:n_active, :], dtype=float)

    total_metals_disk  = metal_disk.sum(axis=1)   # sum over 3 channels
    total_metals_bulge = metal_bulge.sum(axis=1)

    # Metallicity Z = metal_mass / stellar_mass per bin.
    # sfh_DiskMass stores INITIAL mass formed (not surviving mass), which is correct for
    # SSP convolution since SSP templates are also normalized to initial stellar mass.
    # The ratio sfh_total / current StellarMass ~ 1.7 is expected from stellar mass return
    # (~40% over a Hubble time for a Chabrier IMF).
    safe_disk  = np.where(disk_mass  > 0, disk_mass,  1.0)
    safe_bulge = np.where(bulge_mass > 0, bulge_mass, 1.0)
    Z_disk  = np.where(disk_mass  > 0, total_metals_disk  / safe_disk,  Z_SUN * 0.1)
    Z_bulge = np.where(bulge_mass > 0, total_metals_bulge / safe_bulge, Z_SUN * 0.1)

    # Clamp to physically sensible range
    Z_disk  = np.clip(Z_disk,  Z_MIN, Z_MAX)
    Z_bulge = np.clip(Z_bulge, Z_MIN, Z_MAX)

    # Convert to logzsol (log10(Z/Zsun), where Zsun = 0.02 in BC03/FSPS convention)
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
    """
    Sum disk + bulge mass across all SFH bins.  Should equal StellarMass + ICM
    within floating point tolerance.

    Useful for validation.
    """
    return sfh["disk_mass"].sum() + sfh["bulge_mass"].sum()
