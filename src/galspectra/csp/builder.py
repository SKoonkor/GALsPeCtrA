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
    interpolator: PCACoefficientInterpolator
        Ojbect dealing with interpolation in (age, Z)
    target_ages: (N_bins, )
        Ages corresponding to each mass bin (linear scale)
    target_Z : (N_bin, ) or scalar
    mass_weights : (N_bin, )
        Stellar mass formed in each bin

    Returns
    -------
    csp_coeffs : (N_pc, )
    """

    target_ages = np.asarray(target_ages)
    mass_weights = np.asarray(mass_weights)

    # Validate input
    if len(target_ages) != len(mass_weights):
        raise ValueError("target_ages and mass_weights must have same length")

    if np.ndim(target_Z) > 0 and len(target_Z) != len(target_ages):
        raise ValueError("target_Z must match target_ages or be scalar")

    # PCA
    N_pc = interpolator.N_pc
    csp_coeffs = np.zeros(N_pc)

    for i in range(len(target_ages)):
        age = target_ages[i]
        Z = target_Z[i] if np.ndim(target_Z) > 0 else target_Z

        coeff = interpolator.get_coeffs(age, Z)

        if np.any(np.isnan(coeff)):
            raise ValueError(f"Interpolation failed at age={age},  Z={Z}")

        csp_coeffs += mass_weights[i] * coeff

    return csp_coeffs
