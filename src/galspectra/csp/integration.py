import numpy as np

def compute_mass_bins(t_sfh, sfr, age_edges):
    """
    Integrate SFH into SSP age bins.

    Parameters
    ----------
    t_sfh : (Nt, )
    sfr   : (Nt, )
    age_edges : (Na+1, ) SSP age bin edges

    Returns
    -------
    mass : (Na, )
    """

    t_sfh = np.asarray(t_sfh)
    sft = np.asarray(sfr)

    mass = np.zeros(len(age_edges) - 1 )

    for i in range(len(mass)):
        t1, t2 = age_edges[i], age_edges[i+1]

        mask = (t_sfh >= t1) & (t_sfh <= t2)

        if not np.any(mask):
            continue

        t_local = t_sfh[mask]
        sfr_local = sfr[mask]

        mass[i] = np.trapz(sfr_local, t_local)

    return mass




