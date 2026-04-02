import matplotlib.pyplot as plt

def plot_sed(wave, sed, label=None):
    plt.plot(wave,sed, label=label)

def compare_seds(wave, sed_true, sed_pca):
    plt.figure()

    plt.plot(wave, sed_true, label = "True", lw = 2)
    plt.plot(wave, sed_pca, "--", label = "PCA", lw = 2)

    plt.xlabel("Wavelength (A)")
    plt.ylabel("Flux")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.show()


