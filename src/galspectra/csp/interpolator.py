import numpy as np
from scipy.interpolate import RegularGridInterpolator

class PCACoefficientInterpolator:

    def __init__(self, ages, metallicities, coeff_grid):
        """
        Params
        ------
        ages: (N_ages,) linear
        metallicities: (N_Z,) logZ or Z (need to check with the input)
        coeff_grid: (N_age, N_Z, N_pc)
        """

        self.ages = ages
        self.Z = metallicities
        self.coeff_grid = coeff_grid

        self.N_pc = coeff_grid.shape[2]

        # Build one interpolator per PC
        self.interps =[]
        for k in  range(self.N_pc):
            interp = RegularGridInterpolator(
                    (self.ages, self.Z),
                    coeff_grid[:, :, k],
                    bounds_error=False,
                    fill_value=None # allow extrapolation (need to be cautious about this though)
                    )
            self.interps.append(interp)

    def get_coeffs(self, age, Z):
        """

        Params
        ------
        age: scalar
        Z: scalar

        Returns
        -------
        coeffs: (N_pc,)

        """

        point = np.array([[age, Z]])

        coeffs = np.array([
            interp(point)[0] for interp in self.interps])

        return coeffs


