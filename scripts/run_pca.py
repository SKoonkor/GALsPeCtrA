from pathlib import Path
import numpy as np

from galspectra.sed.io import load_sed_grid
from galspectra.pca.preprocess import mask_wavelength, normalize_seds
from galspectra.pca.compute import compute_pca
from galspectra.sed.io import save_pca_results

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data/sed_grid.npz"
OUTPUT_FILE = PROJECT_ROOT / "data/pca_results.npz"

# Load SEDs
print ("\nLoading SEDs")
data = load_sed_grid(INPUT_FILE)

print (data.keys())

wave = data["wave"]
seds = data["seds"]
print (f"Loaded SEDs: {seds.shape}")

# Mask wavelength (optional)
print ("\nMasking wavelength")
wave, seds = mask_wavelength(wave, seds, 3000, 10000)
print (f"After masking: {seds.shape}")

# Normalize
print ("\nNormalizing SEDs")
seds_norm, norm_meta = normalize_seds(seds, method="std", wave=wave)

# PCA
print ("\nRunning PCA on normalized SEDs")
pca_dict = compute_pca(seds_norm, n_components=20)

# Add metadata
pca_dict["norm"] = norm_meta
pca_dict["wave"] = wave

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



