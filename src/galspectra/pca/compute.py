import numpy as np
from sklearn.decomposition import PCA


def compute_pca(seds, n_components=20):
    """
    Perform PCA on SEDs.

    Returns:
    - coeffs (N_samples, n_components)
    - components (n_components, N_wave)
    - explained variance
    """

    pca = PCA(n_components = n_components)

    coeffs = pca.fit_transform(seds)
    components = pca.components_
    variance = pca.explained_variance_ratio_

    return {
            "coeffs": coeffs,
            "components": components,
            "variance": variance,
            "mean": pca.mean_,
            }
