import numpy as np
from scipy.stats import qmc
from itertools import product

# Validation
def _validate_params(params):
    for p in params:
        if "name" not in p: raise ValueError("Each parameter must have a 'name'")
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
    maxs = np.array([p["max"] for p in params])

    scaled = qmc.scale(unit_sample, mins, maxs)

    # Apply scaling
    for i, p in enumerate(params):
        if p["spacing"] == "log":
            scaled[:, i] = 10**scaled[:, i]

    # return scaled
    return {
            "samples": scaled,
            "param_names": [p["name"] for p in params]
            }

def generate_cartesian_grid(params, grid_sizes):
    """
    Generate full Cartesian grid of parameters.

    Parameters
    ----------
    params: list of dict
    grid_sizes : list of int
        Number of grid points per parameter
    """
    _validate_params(params)

    if len(params) != len(grid_sizes):
        raise ValueError("params and grid_sizes must have same length")

    value_axes = []

    for p, n in zip(params, grid_sizes):
        if p["spacing"] == "log":
            axis = np.logspace(p["min"], p["max"], n)
        else:
            axis = np.linspace(p["min"], p["max"], n)

        value_axes.append(axis)

    # Cartesian product (N-dim)
    grid = np.array(list(product(*value_axes)))

    # return grid
    return {
            "samples": grid,
            "param_names": [p["name"] for p in params]
            }








# Test ZONE

params = [
        {"name": "tage", "min": 1e-4, "max": 13.8, "spacing": "linear"},
        {"name": "Z", "min": 0.012, "max": 0.03, "spacing": "linear"},
        ]

scaled_lhs = generate_lhs_grid(params=params, n_samples=10)
scaled_cartesian = generate_cartesian_grid(params=params, grid_sizes = [5, 2])

print ("\nLHS")
print (scaled_lhs)
print ("\nCartesian")
print (scaled_cartesian)




scaled_lhs = []
scaled_cartesian = []
