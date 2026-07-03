"""
Export PCA coefficient grid to a C-readable binary file for L-GALAXIES's in-code,
output-time PCA-coefficient computation.

L-GALAXIES loads this grid and, at snapshot-output time, convolves each galaxy's
stored SFH with it to fill the 50-element `pca_coeffs` output field (see
`code/pca_sed.c`). This is a compact spectral representation, not a live SED
state variable — see documents/onthefly_factcheck.md.

Binary format written:
  [int32]  N_AGE
  [int32]  N_Z
  [int32]  N_PC
  [float64 × N_AGE]             age_gyr[i]        (linear Gyr, ascending)
  [float64 × N_Z]               logzsol[j]        (log10(Z/0.02), ascending)
  [float64 × N_AGE × N_Z × N_PC] coeff_grid[i][j][k]  (row-major C order)

Usage:
    python scripts/export_pca_for_lgalaxies.py [--out PATH]
"""

import argparse
import struct
from pathlib import Path

import numpy as np
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from galspectra.utils.grid import build_param_grid

DEFAULT_PCA_FILE = PROJECT_ROOT / "data" / "pca_results_bc03.npz"
DEFAULT_OUT_DIR  = Path(__file__).resolve().parents[2] / \
    "L-GALAXIES" / "LGalaxies2020_PublicRepository-master" / "SpecPhotTables" / "PCA"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pca-file", default=str(DEFAULT_PCA_FILE),
                   help="Path to pca_results_bc03.npz (default: %(default)s)")
    p.add_argument("--out", default=None,
                   help="Output .bin path (default: SpecPhotTables/PCA/pca_coeff_grid_bc03.bin)")
    return p.parse_args()


def main():
    args = parse_args()

    pca_file = Path(args.pca_file)
    if not pca_file.exists():
        raise FileNotFoundError(f"PCA file not found: {pca_file}")

    print(f"Loading PCA from {pca_file}")
    d = np.load(pca_file, allow_pickle=True)

    params      = d["params"]           # (N_samples, 2): [tage_Gyr, logzsol]
    coeffs      = d["coeffs"]           # (N_samples, N_pc)
    param_names = list(d["param_names"])

    ages_unique, logzsol_unique, coeff_grid = build_param_grid(
        params, coeffs, param_names
    )
    # coeff_grid shape: (N_AGE, N_Z, N_PC)

    N_AGE, N_Z, N_PC = coeff_grid.shape
    print(f"  Grid: {N_AGE} ages × {N_Z} metallicities × {N_PC} PCs")
    print(f"  Age range: {ages_unique[0]:.4f} – {ages_unique[-1]:.4f} Gyr")
    print(f"  logzsol range: {logzsol_unique[0]:.4f} – {logzsol_unique[-1]:.4f}")

    # Determine output path
    if args.out is not None:
        out_path = Path(args.out)
    else:
        out_path = DEFAULT_OUT_DIR / "pca_coeff_grid_bc03.bin"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write binary file
    with open(out_path, "wb") as f:
        # Header: three int32s
        f.write(struct.pack("iii", N_AGE, N_Z, N_PC))

        # Age array (float64)
        ages_unique.astype(np.float64).tofile(f)

        # logzsol array (float64)
        logzsol_unique.astype(np.float64).tofile(f)

        # Coefficient grid in row-major order (float64)
        coeff_grid.astype(np.float64, order="C").tofile(f)

    # Verify by reading back
    with open(out_path, "rb") as f:
        n_age_r, n_z_r, n_pc_r = struct.unpack("iii", f.read(12))
        age_r   = np.frombuffer(f.read(n_age_r * 8), dtype=np.float64)
        logz_r  = np.frombuffer(f.read(n_z_r   * 8), dtype=np.float64)
        grid_r  = np.frombuffer(f.read(n_age_r * n_z_r * n_pc_r * 8),
                                dtype=np.float64).reshape(n_age_r, n_z_r, n_pc_r)

    assert n_age_r == N_AGE and n_z_r == N_Z and n_pc_r == N_PC
    assert np.allclose(age_r,  ages_unique)
    assert np.allclose(logz_r, logzsol_unique)
    assert np.allclose(grid_r, coeff_grid)

    size_kb = out_path.stat().st_size / 1024
    print(f"\nWritten: {out_path}")
    print(f"  File size: {size_kb:.1f} KB")
    print(f"  Verification: OK")


if __name__ == "__main__":
    main()
