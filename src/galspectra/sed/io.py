from pathlib import Path
import numpy as np

def save_sed_grid(filename, sed_dict):
    filename = Path(filename)

    filename.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
            filename,
            wave=sed_dict["wave"],
            seds=sed_dict["seds"],
            params=sed_dict["params"],
            param_names=sed_dict["param_names"],
            )

def load_sed_grid(filename):
    data = np.load(filename, allow_pickle=True)

    return {
            "wave": data["wave"],
            "seds": data["seds"],
            "params": data["params"],
            "param_names": list(data["param_names"]),
    }
