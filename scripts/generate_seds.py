from pathlib import Path
import time

from galspectra.sampling.paramgrid import generate_lhs_grid
from galspectra.sps.fsps_backend import create_stellar_population
from galspectra.sed.generator import generate_seds
from galspectra.sed.io import save_sed_grid

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT/"data"
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = DATA_DIR/"sed_grid.npz"



params = [
        {"name": "tage", "min": -4, "max": 1.137, "spacing": "log"},
        {"name": "logzsol", "min": -1, "max": 0.1, "spacing": "linear"}
        ]

print ("\nGenerating parameter grid")
param_dict = generate_lhs_grid(params, n_samples = 1000)

print ("\nInitiating stellar population synthesis")
sp = create_stellar_population(logzsol = 0.0)

print ("\nGenerating SEDs")
sed_data = generate_seds(param_dict, sp)

print ("\nSaving SEDs")
save_sed_grid(OUTPUT_FILE, sed_data)
print (f"\nFile saved at {OUTPUT_FILE}")



