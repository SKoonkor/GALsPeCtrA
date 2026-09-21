"""
run_pca_evalgrid.py

Fit the production PCA basis on the resampled, science-motivated evaluation
grid (R=300, 1048-23552 A -- configs/pca_experiments/default.yaml) instead of
the native BC03 831-bin grid that scripts/run_pca.py fits on.

This is the basis refit motivated by Sec 5.2 of the paper: fitting on the
evaluation grid removes the components the native-grid basis spends on
wavelengths (805-1048 A, 23552-29950 A) no photometric band in the paper
reaches, and was measured to improve PAUS narrow-band accuracy by ~1.4x at
the same storage cost. This script makes that basis the one exported to
L-GALAXIES, rather than leaving it as a resampled-grid-only comparison.

Same steps as run_pca.py (mask -> normalize("std") == inverse_std weighting
-> compute_pca), except the "mask" step is a flux-conserving resample rather
than a simple wavelength cut, using the exact resampling_matrix() function
pca_basis_experiment.py already uses for the evaluation-grid comparison.

Output schema matches run_pca.py exactly (coeffs, components, variance, mean,
wave, norm, params, param_names, config), so export_pca_for_lgalaxies.py and
every downstream consumer (process_lgalaxies.py, the notebooks) work
unchanged.

Usage:
    python scripts/run_pca_evalgrid.py
    python scripts/run_pca_evalgrid.py --input data/sed_grid_bc03.npz \
        --output data/pca_results_bc03.npz --n-components 50 \
        --wave-min 1048 --wave-max 23552 --resolution 300
"""
import argparse
from pathlib import Path
import numpy as np

from galspectra.sed.io import load_sed_grid, save_pca_results
from galspectra.pca.resample import log_wavelength_grid, resampling_matrix
from galspectra.pca.preprocess import normalize_seds
from galspectra.pca.compute import compute_pca

PROJECT_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="Fit the production PCA basis on the evaluation grid")
parser.add_argument("--input", default=str(PROJECT_ROOT / "data/sed_grid_bc03.npz"))
parser.add_argument("--output", default=str(PROJECT_ROOT / "data/pca_results_bc03.npz"))
parser.add_argument("--n-components", type=int, default=50)
parser.add_argument("--wave-min", type=float, default=1048.0)
parser.add_argument("--wave-max", type=float, default=23552.0)
parser.add_argument("--resolution", type=float, default=300.0)
args = parser.parse_args()

print("\nLoading native-grid SEDs")
data = load_sed_grid(Path(args.input))
wave_native = data["wave"]
seds_native = data["seds"]
print(f"Loaded SEDs: {seds_native.shape}, native grid {wave_native.min():.0f}-{wave_native.max():.0f} A "
      f"({len(wave_native)} bins)")

print("\nResampling onto the evaluation grid (flux-conserving)")
centres, edges = log_wavelength_grid(args.wave_min, args.wave_max, args.resolution)
M = resampling_matrix(wave_native, edges)
seds_eval = seds_native @ M.T
print(f"Evaluation grid: R={args.resolution:g}, {len(centres)} bins over "
      f"{args.wave_min:.0f}-{args.wave_max:.0f} A")

print("\nNormalizing SEDs (inverse_std weighting)")
seds_norm, norm_meta = normalize_seds(seds_eval, method="std", wave=centres)

print("\nRunning PCA on normalized SEDs")
pca_dict = compute_pca(seds_norm, n_components=args.n_components)

pca_dict["norm"] = norm_meta
pca_dict["wave"] = centres
pca_dict["params"] = data["params"]
pca_dict["param_names"] = data["param_names"]
pca_dict["config"] = dict(data["config"]) if hasattr(data["config"], "items") else data["config"]

print("\nSaving PCA results")
save_pca_results(Path(args.output), pca_dict)
print(f"PCA results saved to: {args.output}")

variance = pca_dict["variance"]
print("\nSanity check")
print("-" * 12)
print("Variance of first 10 PCs:", variance[:10])
print("Cumulative variance (50 PCs):", variance.cumsum()[-1])

coeffs = pca_dict["coeffs"]
components = pca_dict["components"]
mean = pca_dict["mean"]
recon = mean + coeffs[0] @ components
error = np.mean((recon - seds_norm[0]) ** 2)
print("Reconstruction MSE (SSP 0, normalized space):", error)
