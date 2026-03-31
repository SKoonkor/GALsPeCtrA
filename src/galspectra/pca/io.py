def save_pca_results(filename, pca_dict):
    np.savez(
            filename,
            coeffs=pca_dict["coeffs"],
            components=pca_dict["components"],
            variance=pca_dict["variance"],
            mean=pca_dict["mean"],
            )


