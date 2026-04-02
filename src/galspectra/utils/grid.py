import numpy as np

def build_param_grid(params, coeffs, param_names):
    """
    Convert flatten SSP samples into structured grid.
    
    Parameters
    ----------
    params: (N_samples, N_params)
    coeffs: (N_samples, N_pc)

    Returns
    -------
    age_unique: (N_age,)
    Z_unique: (N_Z,)
    coeff_grid: (N_age, N_Z, N_pc)
    """


    age_idx = param_names.index("tage")
    z_idx = param_names.index("logzsol")

    ages = params[:, age_idx]
    Zs = params[:, z_idx]

    # Ensure linear age
    from utils.age import ensure_linear_age
    
    ages, _ = ensure_linear_age(ages)

    ages_unique = np.sort(np.unique(ages))
    Z_unique = np.sort(np.unique(Zs))

    N_age = len(ages_unique)
    N_Z = len(Z_unique)
    N_pc = coeffs.shape[1]

    coeff_grid = np.zeros((N_age, N_Z, N_pc))

    for i, age in enumerate(ages_unique):
        for j, z in enumerate(Z_unique):
            mask = (ages == age) & (Zs == z)

            if not np.any(mask):
                raise ValueError(f"Missing grid point at age={age}, Z={z}")

            coeff_grid[i, j] = coeffs[mask][0]

    return ages_unique, Z_unique, coeff_grid

