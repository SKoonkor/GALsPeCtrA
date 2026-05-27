import argparse
from pathlib import Path
import numpy as np

from galspectra.sed.io import load_sed_grid
from galspectra.pca.preprocess import mask_wavelength, normalize_seds
from galspectra.pca.compute import compute_pca
from galspectra.sed.io import save_pca_results

PROJECT_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="Run PCA on an SSP SED grid")
parser.add_argument("--input",       default=str(PROJECT_ROOT / "data/sed_grid.npz"),
                    help="Input SED grid .npz file")
parser.add_argument("--output",      default=str(PROJECT_ROOT / "data/pca_results.npz"),
                    help="Output PCA results .npz file")
parser.add_argument("--n-components", type=int, default=50,
                    help="Number of PCA components to keep (default 50)")
parser.add_argument("--wave-min",    type=float, default=3000.0,
                    help="Minimum wavelength to include in PCA (Å, default 3000)")
parser.add_argument("--wave-max",    type=float, default=10000.0,
                    help="Maximum wavelength to include in PCA (Å, default 10000)")
args = parser.parse_args()

INPUT_FILE  = Path(args.input)
OUTPUT_FILE = Path(args.output)

# Load SEDs
print ("\nLoading SEDs")
data = load_sed_grid(INPUT_FILE)

print (data.keys())

wave = data["wave"]
seds = data["seds"]
print (f"Loaded SEDs: {seds.shape}")

# Mask wavelength
print ("\nMasking wavelength")
wave, seds = mask_wavelength(wave, seds, args.wave_min, args.wave_max)
print (f"After masking: {seds.shape}")

# Normalize
print ("\nNormalizing SEDs")
seds_norm, norm_meta = normalize_seds(seds, method="std", wave=wave)

# PCA
print ("\nRunning PCA on normalized SEDs")
pca_dict = compute_pca(seds_norm, n_components=args.n_components)

# Add metadata
pca_dict["norm"] = norm_meta
pca_dict["wave"] = wave
pca_dict["params"] = data["params"]
pca_dict["param_names"] = data["param_names"]
pca_dict["config"] = data["config"]


# Save
print ("\nSaving PCA results")
save_pca_results(OUTPUT_FILE, pca_dict)

print (f"PCA results save to: {OUTPUT_FILE}")

















# Test zone
print ("\nSanity Check")
print ("-"*12)
print ("\nVariance of first 10 PCs")
variance = pca_dict["variance"]
print(variance[:10])
print("Cumulative:", variance.cumsum())

print ("\nReconstruction Error")
coeffs = pca_dict["coeffs"]
components = pca_dict["components"]
mean = pca_dict["mean"]

# reconstruct first SED
recon = mean + coeffs[0] @ components

# compare
original = seds_norm[0]

error = np.mean((recon - original)**2)
print("Reconstruction MSE:", error)



