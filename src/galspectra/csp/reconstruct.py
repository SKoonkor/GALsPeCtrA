import numpy as np

def reconstruction_sed_from_pca(
        coeffs,
        components,
        mean,
        norm_meta=None,
        ):
    """
    Reconstrcut SED from PCA coefficients.

    Parameters
    ----------
    coeffs : (N_pc,)
    components : (N_pc, N_wave)
    mean : (N_wave,)
    norm_meta : dict (optional)
        Needed to denormalise 

    Returns
    -------
    sed : (N_wave,)
    """

    sed_norm = mean + coeffs @ components
    
    # Denormalise if needed
    if norm_meta is not None:
        if "std" in norm_meta:
            sed = sed_norm * norm_meta["std"] + norm_meta["mean"]

        else:
            sed = sed_norm
    else:
        sed = sed_norm

    return sed



