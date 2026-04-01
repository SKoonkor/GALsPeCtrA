import numpy as np

def build_csp_coefficients(
        ssp_age,
        ssp_coeffs,
        target_ages,
        mass_weights
        ):
    """
    Build CSP in PCA space.

    Parameters
    ----------
    ssp_age: (N_ssp, )
    ssp_coeffs : (N_ssp, N_pc)
    target_ages : (N_bin, )
        Ages corresponding to each mass bin
    mass_weights : (N_bin, )

    Returns
    -------
    csp_coeffs : (N_pc, )
    """

    n_pc = ssp_coeffs.shape[1]
    csp = np.zeros(n_pc)

    for age, mass in zip(target_ages, mass_weights):
        # Find nearest SSP
        ## Will need to be updated
        ##   - to interpolation in age
        ##   - to interpolation in age and metallicity 
        idx = np.argmin(np.abs(ssp_age - age))   
        

        csp += mass * ssp_coeffs[idx]

    return csp
