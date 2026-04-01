from pathlib import Path
import numpy as np

def save_sed_grid(filename, sed_dict, config=None):
    filename = Path(filename)

    filename.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
            filename,
            wave=sed_dict["wave"],
            seds=sed_dict["seds"],
            params=sed_dict["params"],
            param_names=sed_dict["param_names"],
            config=config
            )

def load_sed_grid(filename):
    data = np.load(filename, allow_pickle=True)

    return {
            "wave": data["wave"],
            "seds": data["seds"],
            "params": data["params"],
            "param_names": list(data["param_names"]),
            "config": data.get("config", None), 
    }


def save_pca_results(filename, pca_dict):
    np.savez(
            filename,
            coeffs=pca_dict["coeffs"],
            components=pca_dict["components"],
            variance=pca_dict["variance"],
            mean=pca_dict["mean"],
            wave=pca_dict["wave"],
            norm=pca_dict["norm"],
            params=pca_dict["params"],
            param_names=pca_dict["param_names"],
            )
