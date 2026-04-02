import numpy as np
 

def logage_to_age(logage):
    """
    Convert log10(age/Gyr) to age (Gyr)
    """
    return 10**logage

def age_to_logage(age):
    """
    Convert age (Gyr) to log10(age/Gyr)
    """

    if np.any(age <= 0):
        raise ValueError("Age must be > 0 for log conversion")

    return np.log10(age)


def validate_age_array(age, name = "age"):
    """
    Validation for age arrays if they are in a log or linear scale
    """

    age = np.asarray(age)

    if np.any(~np.isfinite(age)):
        raise ValueError(f"{name} contains Nan or inf")

    if np.any(age < 0):
        raise ValueError(f"{name} contains negative values")

    if np.all(age == 0):
        raise ValueError(f"{name} is all zeros")

    return age


def ensure_linear_age(age_array):
    """
    Detect if input is log-age and convert to linear.

    Assumptioins:
    - If there are some negative values, assume log(age)
    - Else, assume linear
    """

    age_array = np.asarray(age_array)

    if np.any(age_array < 0):
        # Likely log(age)
        return 10**age_array, "log"
    else:
        return age_array, "linear"


