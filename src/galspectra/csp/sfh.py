import numpy as np

def sfh_tau(t, T0, tau):
    return np.where(t > T0, np.exp(-(t - T0)/tau), 0.0)

def sfh_delayed_tau(t, T0, tau):
    return (t - T0) * sfh_tau(t, T0, tau)

def sfh_lognormal(t, T0, tau):
    return np.exp(-(np.log(t) - T0)**2 / (2*tau**2)) / t

def sfh_dpl(t, tau, alpha, beta):
    return ((t/tau)**alpha + (t/tau)**(-beta))**-1

def sfh_burst(t, tburst, sigma = 0.05):
    return np.exp(-0.5 * ((t - tburst)/sigma)**2)

def build_sfh(t, components, mass_norm=True):
    """
    Build composite SFH and normalize to total mass = 1.
    """
    sfr = np.zeros_like(t)

    for comp in components:
        typ = comp["type"]

        if typ == "tau":
            val = sfh_tau(t, comp["T0"], comp["tau"])
        elif typ == "delayed_tau":
            val = sfh_delayed_tau(t, comp["T0"], comp["tau"])
        elif typ == "lognormal":
            val = sfh_lognormal(t, comp["T0"], comp["tau"])
        elif typ == "burst":
            val = sfh_burst(t, comp["tburst"], comp.get("sigma", 0.05))
        else:
            raise ValueError(f"Unkown SFH type: {typ}")

        sfr += comp.get("strength", 1.0) * val

    if mass_norm:
        # Normalize mass formed
        dt = np.gradient(t)
        mass = np.sum(sfr * dt)
        return sfr / mass
    else: 
        return sfr

