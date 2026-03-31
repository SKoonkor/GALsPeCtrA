import numpy as np

def mask_wavelength(wave, seds, wave_min=None, wave_max=None):
    """
    Apply wavelength mask to SEDs.
    
    Returns masked wave and SEDs.
    """

    mask = np.ones_like(wave, dtype=bool)

    if wave_min is not None:
        mask &= wave >= wave_min
    if wave_max is not None:
        mask &= wave <= wave_max

    return wave[mask], seds[:, mask]


def normalize_seds(seds, method="std", wave=None, eps=1e-14):
    """
    Normalize SEDs.

    Methods:
    - 'std': per-wavelength standardization
    - 'l2': per-spectrum L2 norm
    - '5500A': normalize at 5500A
    """

    if method == "std":
        mean = np.mean(seds, axis=0)
        std = np.std(seds, axis=0)
        seds_norm = (seds - mean) / (std + eps)
        return seds_norm, {"mean": mean, "std": std}

    elif method == "l2":
        norms = np.linalg.norm(seds, axis=1, keepdims=True)
        seds_norm = seds / (norms + eps)
        return seds_norm, {"norm": norms}

    elif method == "5500A":
        if wave is None:
            raise ValueError("Wave required for 5500A normalization")
        
        idx = np.argmin(np.abs(wave - 5500))
        factors = seds[:, idx][:, None]
        seds_norm = seds / (factors + eps)
        return seds_norm, {"factors": factors}

    else:
        raise ValueError("Unknown normalization method")


