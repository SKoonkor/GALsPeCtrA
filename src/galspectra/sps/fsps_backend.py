import os

def get_fsps():
    try:
        import fsps
    except ImportError:
        raise ImportError(
                "python-fsps is not installed. Run `pip install fsps'."
                )
    if "SPS_HOME" not in os.environ:
        raise RuntimeError(
                "SPS_HOME environment variable is not set."
                )
    return fsps

def create_stellar_population(**kwargs):
    fsps = get_fsps()

    defaults = dict(
            compute_vega_mags=False,
            zcontinuous=1,
            sfh=0, #SSP
            logzsol=0.0,
            add_dust_emission=False,
            add_neb_emission=False,
            )
    defaults.update(kwargs)

    return fsps.StellarPopulation(**defaults)


# # Test zone
# fsps = get_fsps()
# 
# 
# sp = create_stellar_population(compute_vega_mags = True)
# print (sp)
# 
# wave, spec = sp.get_spectrum(tage=1.0)
