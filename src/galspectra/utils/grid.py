import numpy as np

from galspectra.utils.age import ensure_linear_age

def build_param_grid(params, coeffs, param_names):
    """
    Convert flatten SSP samples into structured grid.
    
    Parameters
    ----------
    params: (N_samples, N_params)
    coeffs: (N_samples, N_pc)
    param_names: list

    Returns
    -------
    age_unique: (N_age,) [linear scale]
    Z_unique: (N_Z,)     [logzsol]
    coeff_grid: (N_age, N_Z, N_pc)
    """

    # check parameter indices
    try:
        age_idx = param_names.index("tage")
        z_idx = param_names.index("logzsol")
    except ValueError:
        raise ValueError("param_names must contain 'tage' and 'logzsol'")

    # get parameters
    print (params.shape)
    print (params[0:10])
    ages = params[:, age_idx]
    Zs = params[:, z_idx]

    # Ensure linear age
    ages, _ = ensure_linear_age(ages)

    ages_unique = np.sort(np.unique(ages))
    Z_unique = np.sort(np.unique(Zs))

    N_age = len(ages_unique)
    N_Z = len(Z_unique)
    N_pc = coeffs.shape[1]

    coeff_grid = np.zeros((N_age, N_Z, N_pc))

    # Build index map
    age_index = {val: i for i, val in enumerate(ages_unique)}
    Z_index = {val: j for j, val in enumerate(Z_unique)}

    # fill grid
    filled = np.zeros((N_age, N_Z), dtype=bool)

    for n in range(len(params)):
        age = ages[n]
        Z = Zs[n]

        i = age_index[age]
        j = Z_index[Z]

        if filled[i, j]:
            raise ValueError(f"Duplicate grid point at age={age}, Z={Z}")

        coeff_grid[i, j] = coeffs[n]
        filled[i, j] = True

    # Check completeness
    if not np.all(filled):
        missing = np.where(~filled)
        raise ValueError(f"Incomplete grid: missing entries at indices {missing}")

    return ages_unique, Z_unique, coeff_grid
