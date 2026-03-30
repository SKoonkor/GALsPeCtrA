import numpy as np

def generate_seds(param_dict, sp, verbose=True):
    """
    Generate SEDs from a parameter grid.

    Parameters
    ----------
    param_dict : dict
        Output from sampling/paramgrid.py:
        {
            "samples": (N, P) array,
            "param_names": list of str
        }

    sp : fsps.StellarPopulation
        Pre-initialized FSPS object

    verbose : bool
        Print progress

    Returns
    -------
    dict 
        {
            "wave", (N_wave,),
            "seds": (N_samples, N_wave),
            "params": (N_samples, N_params),
            "param_names": list
        }
    """

    samples = param_dict["samples"]
    param_names = param_dict["param_names"]

    n_samples = samples.shape[0]

    seds = []

    for i in range(n_samples):
        params = dict(zip(params_names, samples[i]))

        # Update FSPS parameters
        for key, val in params.items():
            if key != "tage":
                sp.params[key] = val

        # Generate spectrum
        wave, sed = sp.get_spectrum(
                tage = params.get("tage", 1.0),
                peraa = True,
                )

        seds.append(sed)

        if verbose and i%max(1, n_samples // 50) == 0:
            print (f"\rProgress: {int(i/n_samples*100)}%", end="")

    if verbose:
        print ("\rProgress: 100%")

    seds = np.array(seds)

    return {
            "wave": wave,
            "seds": seds,
            "params": samples,
            "param_names": param_names,
            }

