"""
generate_seds_bc03.py

Generate a BC03 SSP SED grid using the FullSED files from the L-GALAXIES
SpecPhotTables directory.

Grid: 200 ages × 6 metallicities = 1200 SSPs
(matches BC03's native 6 metallicity bins)

Output: data/sed_grid_bc03.npz

Usage:
  cd /path/to/GALsPeCtrA
  python scripts/generate_seds_bc03.py [--bc03-dir /path/to/FullSEDs]
"""

import argparse
from pathlib import Path
import time
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Default BC03 directory (adjust if your L-GALAXIES installation is elsewhere)
DEFAULT_BC03_DIR = (
    PROJECT_ROOT.parent
    / "L-GALAXIES"
    / "LGalaxies2020_PublicRepository-master"
    / "SpecPhotTables/FullSEDs"
)


def parse_args():
    p = argparse.ArgumentParser(description="Generate BC03 SSP SED grid")
    p.add_argument("--bc03-dir", default=str(DEFAULT_BC03_DIR),
                   help="Path to directory with BC03_Chabrier_FullSED_m*.dat files")
    p.add_argument("--n-ages",   type=int, default=200,
                   help="Number of age points (log-spaced, default 200)")
    p.add_argument("--wave-min", type=float, default=912.0,
                   help="Minimum wavelength in Å (default 912 = Lyman limit)")
    p.add_argument("--wave-max", type=float, default=25000.0,
                   help="Maximum wavelength in Å (default 25000)")
    p.add_argument("--output",   default=str(DATA_DIR / "sed_grid_bc03.npz"),
                   help="Output file path")
    return p.parse_args()


def main():
    args = parse_args()
    bc03_dir = Path(args.bc03_dir)
    output_file = Path(args.output)

    if not bc03_dir.exists():
        raise FileNotFoundError(
            f"BC03 directory not found: {bc03_dir}\n"
            "Set --bc03-dir to the path of the SpecPhotTables/FullSEDs directory."
        )

    from galspectra.sps.bc03_backend import (
        BC03Library, generate_bc03_seds, BC03_METALLICITIES
    )
    from galspectra.sed.io import save_sed_grid

    import numpy as np
    from itertools import product as cartesian_product

    # BC03's 6 metallicity bins in logzsol — use exact native values, not linear interpolation
    Z_grid       = BC03_METALLICITIES                     # [0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05]
    logzsol_grid = np.log10(Z_grid / 0.02)                # [-2.301, -1.699, -0.699, -0.398, 0.000, 0.398]

    # Age axis: log-spaced from 0.0001 to 13.7 Gyr, stored as LINEAR Gyr
    # generate_bc03_seds reads these directly (no 10^ conversion)
    age_axis    = np.logspace(np.log10(1e-4), np.log10(13.7), args.n_ages)

    # Build Cartesian product manually to avoid paramgrid's linear metallicity interpolation
    samples = np.array(list(cartesian_product(age_axis, logzsol_grid)))
    param_dict = {"samples": samples, "param_names": ["tage", "logzsol"]}

    n_samples = samples.shape[0]
    print("Building parameter grid")
    print(f"  {n_samples} SSPs ({args.n_ages} ages × {len(Z_grid)} metallicities)")
    print(f"  Age range: {age_axis[0]:.4f} – {age_axis[-1]:.2f} Gyr")
    print(f"  logzsol:   {logzsol_grid}")

    print(f"\nLoading BC03 library from {bc03_dir}")
    print("  (first load of each metallicity file takes ~30 s; 6 files total)")
    lib = BC03Library(bc03_dir, wave_min=args.wave_min, wave_max=args.wave_max)

    print(f"\nGenerating {n_samples} SEDs ...")
    t0 = time.time()
    sed_data = generate_bc03_seds(
        param_dict, bc03_dir, args.wave_min, args.wave_max,
        verbose=True, lib=lib,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f} s  ({elapsed/n_samples:.2f} s/SED)")
    print(f"  SED shape: {sed_data['seds'].shape}")
    print(f"  Wavelength range: {sed_data['wave'][0]:.0f} – {sed_data['wave'][-1]:.0f} Å")

    print(f"\nSaving to {output_file}")
    save_sed_grid(output_file, sed_data)
    print("Done.")


if __name__ == "__main__":
    main()
