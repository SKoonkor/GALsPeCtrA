import numpy as np

def build_csp_coefficients(
        interpolator,
        target_ages,
        target_Z,
        mass_weights):
    """
    Build CSP in PCA space.

    Parameters
    ----------
    interpolator: function
        The interpolation function to interpolate the age and metallicity coeff grid
    target_ages: (N_bins, )
        Ages corresponding to each mass bin
    target_Z : (N_bin, ) or scalar
    mass_weights : (N_bin, )

    Returns
    -------
    csp_coeffs : (N_pc, )
    """

    N_pc = interpolator.N_pc
    csp_coeffs = np.zeros(N_pc)

    for i in range(len(target_ages)):
        age = target_ages[i]
        Z = target_Z[i] if np.ndim(target_Z) > 0 else target_z

        coeff = interpolator.get_coeffs(age, Z)

        csp_coeffs += mass_weights[i] * coeff

    return csp_coeffs
