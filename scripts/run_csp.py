from pathlib import Path
import numpy as np

from galspectra.sed.io import load_sed_grid
from galspectra.pca.compute import compute_pca
from galspectra.csp.sfh import build_sfh
from galspectra.csp.integration import compute_mass_bins
from galspectra.csp.builder import build_csp_coefficients
from galspectra.csp.interpolator import PCACoefficientInterpolator
from galspectra.utils.grid import build_param_grid


PROJECT_ROOT = Path(__file__).resolve().parents[1]

print (PROJECT_ROOT)

pca = np.load(PROJECT_ROOT / "data/pca_results.npz", allow_pickle=True)  # flattened grid

coeffs = pca["coeffs"]
params = pca["params"]
param_names = list(pca["param_names"])
print ("Loaded coeffs shape:", coeffs.shape)

ages, Z, coeff_grid = build_param_grid(
        params,
        coeffs,
        param_names,)

print ("Grid shape:", coeff_grid.shape)

interpolator = PCACoefficientInterpolator(
        ages,
        Z,
        coeff_grid,)

# TODO: use the GALFORM output once the code is stable
t = np.linspace(1e-4, 10, 200)

sfh = build_sfh(t, [
    {"type": "tau", "T0": 0.5, "tau": 2.0}
    ])

age_edges = np.linspace(0, 10, 30)

mass = compute_mass_bins(t, sfh, age_edges)

age_centers = 0.5 * (age_edges[:-1] + age_edges[1:])

Z_csp = 0.0  # constant metallicity, logzsol = solar

csp_coeffs = build_csp_coefficients(
        interpolator,
        age_centers,
        Z_csp,
        mass,)

print ("CSP coeffs shape:", csp_coeffs.shape)

