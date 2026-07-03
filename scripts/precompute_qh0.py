"""
precompute_qh0.py

Build the Q(H0)(age, Z) lookup table by integrating BC03 SSP flux over the
Lyman continuum (91–912 Å).

Must be run from the project root:
  cd /path/to/GALsPeCtrA
  python scripts/precompute_qh0.py

Output: data/qh0_grid_bc03.npz
  Keys: qh0 (N_age, N_Z), age_grid_gyr (N_age,), Z_grid (N_Z,)
"""

import argparse
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

DEFAULT_BC03_DIR = (
    PROJECT_ROOT.parent
    / "L-GALAXIES"
    / "LGalaxies2020_PublicRepository-master"
    / "SpecPhotTables/FullSEDs"
)


def parse_args():
    p = argparse.ArgumentParser(description="Pre-compute Q(H0) grid from BC03 SSPs")
    p.add_argument("--bc03-dir", default=str(DEFAULT_BC03_DIR))
    p.add_argument("--sed-grid", default=str(DATA_DIR / "sed_grid_bc03.npz"),
                   help="Existing SSP SED grid (to reuse its age/Z grid)")
    p.add_argument("--output",   default=str(DATA_DIR / "qh0_grid_bc03.npz"))
    return p.parse_args()


def main():
    args     = parse_args()
    bc03_dir = Path(args.bc03_dir)
    sed_file = Path(args.sed_grid)
    out_file = Path(args.output)

    if not bc03_dir.exists():
        raise FileNotFoundError(f"BC03 directory not found: {bc03_dir}")

    # ── Reuse the age/Z grid from the existing SSP SED grid ───────────────
    print(f"Loading SSP grid from {sed_file} to extract age/Z axes...")
    sed = np.load(sed_file)
    params      = sed["params"]        # (1200, 2): [tage_gyr, logzsol]
    param_names = list(sed["param_names"])
    tage_idx    = param_names.index("tage")
    logz_idx    = param_names.index("logzsol")

    # Unique sorted axes
    age_grid_gyr = np.unique(params[:, tage_idx])   # (200,) Gyr
    # Use exact BC03 metallicity values (avoids floating-point key mismatches)
    from galspectra.sps.bc03_backend import BC03_METALLICITIES
    Z_grid = BC03_METALLICITIES                       # (6,) exact Python floats

    print(f"  {len(age_grid_gyr)} ages × {len(Z_grid)} metallicities")
    print(f"  Age range: {age_grid_gyr[0]:.4f}–{age_grid_gyr[-1]:.2f} Gyr")
    print(f"  Z grid:    {Z_grid}")

    # ── Compute Q(H0) grid ────────────────────────────────────────────────
    from galspectra.nebular.ionizing import compute_qh0_grid

    print("\nComputing Q(H0) grid (integrates BC03 Lyman continuum, 91–912 Å)...")
    qh0, _, _ = compute_qh0_grid(bc03_dir, age_grid_gyr, Z_grid)

    print(f"\nQ(H0) grid shape: {qh0.shape}")
    print(f"Q(H0) range: {qh0.min():.3e} – {qh0.max():.3e} photons/s/Msun")

    # Sanity check: Q(H0) at 10 Myr, solar Z should be ~10^46 photons/s/Msun
    idx_10myr   = np.argmin(np.abs(age_grid_gyr - 0.010))
    idx_solar   = np.argmin(np.abs(Z_grid - 0.02))
    q_10myr_sol = qh0[idx_10myr, idx_solar]
    print(f"\nSanity check: Q(H0) at 10 Myr, Z_sun = {q_10myr_sol:.3e} photons/s/Msun")
    print(f"  Expected ~1e44 at 10 Myr (Q(H0) peaks at ~4e46 at 1 Myr, steeply declining).")

    # ── Save ─────────────────────────────────────────────────────────────
    np.savez(out_file, qh0=qh0, age_grid_gyr=age_grid_gyr, Z_grid=Z_grid)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
