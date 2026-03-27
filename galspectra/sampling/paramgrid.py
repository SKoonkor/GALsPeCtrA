import numpy as np
from scipy.stats import qmc
from itertools import product

# Validation
def _validate_params(params):
    for p in params:
        if "name" not in p:
            raise ValueError("Each parameter must have a 'name'")
        if p["min"] >= p["max"]:
            raise ValueError(f"{p['name']}: min must be < max")
        if p["spacing"] not in ["linear", "log"]:
            raise ValueError(f"{p['name']}: spacing must be 'linear' or 'log'")

# Grid Generators
def generate_lhs_grid(params, n_samples=200, seed=None):
    """
    Generate parameter samples using Latin Hypercube Sampling.
    """
    _validate_params(params)

    dim = len(params)
    sampler = qmc.LatinHypercube(d = dim, seed = seed)

    unit_sample = sampler.random(n = n_samples)

    mins = np.array([p["min"] for p in params])
    maxs = np.array([p{"max"] for p in params])

    scaled = qmc.scale(unit_sample, mins, maxs)

    # Apply scaling
    for i, p in enumerate(params):
        if p["spacing"] == "log":
            scaled[:, i] = 10**scaled[:, i]

    return scaled







# Test ZONE

params = [
        {"name": "tage", "min": 1e-4, "max": 13.8, "spacing": "linear"},]

scaled = generate_lhs_grid(params=params, n_samples=500)

print (scaled)

scaled = []
