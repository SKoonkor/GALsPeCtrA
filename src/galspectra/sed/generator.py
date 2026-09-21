import numpy as np

def generate_seds(param_dict, sp, verbose=True):
    """Generate SEDs from a parameter grid.

    param_dict : from sampling/paramgrid.py, {"samples": (N,P), "param_names": [...]}.
    sp : pre-initialized fsps.StellarPopulation.

    Returns dict: wave (N_wave,), seds (N,N_wave), params (N,P), param_names.
    """

    samples = param_dict["samples"]
    param_names = param_dict["param_names"]

    n_samples = samples.shape[0]

    seds = []

    for i in range(n_samples):
        params = dict(zip(param_names, samples[i]))

        for key, val in params.items():
            if key != "tage":
                sp.params[key] = val

        wave, sed = sp.get_spectrum(
                tage = params.get("tage", 1.0),
                peraa = True,
                )

        seds.append(sed)

        if verbose and i%max(1, n_samples // 100) == 0:
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

