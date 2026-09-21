#!/usr/bin/env python
"""
Build a self-contained, joined per-galaxy property + photometry + PCA-basis
table for a given L-GALAXIES sample.

The notebooks and validation scripts read the large external L-GALAXIES data
tree (galaxy catalog, BC03 FullSEDs, SFH FITS table) directly, but a few
tools -- ``validate_by_colour.py``, ``pca_basis_experiment.py`` -- want one
small, portable file joining a galaxy's stored properties, its native
L-GALAXIES photometry, its synthetic photometry, its PCA coefficients, and
the PCA basis itself, without re-reading the external tree each time. This
script is run **once** on a machine where that external data is present and
writes that joined table to ``data/galaxy_table_<TREE>.npz`` (``TREE`` is
``MR`` or ``MRII``, matching ``galspectra.trees.canonical_label``).

Everything after this step reads only ``data/``. If any external path is
missing the script fails loudly with a clear message.

Usage
-----
    python scripts/build_galaxy_table.py
    python scripts/build_galaxy_table.py --tree MRII --sample /path/to/sample.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Default external L-GALAXIES location (identical to the notebooks)
LGAL_ROOT_DEFAULT = PROJECT_ROOT.parent / "L-GALAXIES" / "LGalaxies2020_PublicRepository-master"

BANDS = ["u", "g", "r", "i", "z"]


def _p(msg: str) -> None:
    print(msg, flush=True)


def build_property_table(sample_file: Path, coeffs_file: Path) -> dict:
    """Per-galaxy properties + photometry for every galaxy in the sample.

    Row i of every output array corresponds to sample galaxy i (the post-
    processing coeff file stores galaxies in the same sequential order, which
    is asserted below).
    """
    from galspectra.lgalaxies import load_sample

    _p(f"Loading galaxy sample: {sample_file}")
    G = load_sample(sample_file)
    N = len(G)
    _p(f"  {N:,} galaxies")

    _p(f"Loading PCA coefficients + synthetic mags: {coeffs_file}")
    coeffs_npz = np.load(coeffs_file, allow_pickle=True)
    pca_coeffs = coeffs_npz["pca_coeffs"].astype(np.float32)  # (N, 50)
    gidx = coeffs_npz["galaxy_index"]
    if not (len(pca_coeffs) == N and np.array_equal(gidx, np.arange(N))):
        raise ValueError(
            "Coefficient file is not aligned 1:1 with the sample "
            f"(coeffs N={len(pca_coeffs)}, sample N={N}). "
            "This builder assumes sequential galaxy_index; regenerate the "
            "coeffs file with process_lgalaxies.py on this sample."
        )

    # Total mass formed across active SFH bins (Msun; sample sfh_* already Msun).
    n_sfh = G["sfh_DiskMass"].shape[1]
    bin_range = np.arange(n_sfh)[np.newaxis, :]
    sfh_mask = bin_range <= G["sfh_ibin"].astype(int)[:, np.newaxis]
    m_total = ((G["sfh_DiskMass"] * sfh_mask).sum(1) +
               (G["sfh_BulgeMass"] * sfh_mask).sum(1)).astype(np.float64)

    stellar = G["StellarMass"].astype(np.float64)
    table = {
        "StellarMass": stellar.astype(np.float32),
        "BulgeMass": G["BulgeMass"].astype(np.float32),
        "BT": (G["BulgeMass"] / (stellar + 1e-10)).astype(np.float32),
        "Sfr": G["Sfr"].astype(np.float32),
        "sSFR": (G["Sfr"] / (stellar + 1e-30)).astype(np.float32),
        "ColdGas": G["ColdGas"].astype(np.float32),
        "H2fraction": G["H2fraction"].astype(np.float32),
        "Type": G["Type"].astype(np.int32),
        "MassWeightAge": G["MassWeightAge"].astype(np.float32),
        "M_total": m_total.astype(np.float64),
        "pca_coeffs": pca_coeffs,
    }
    # L-GALAXIES stored photometry (5 bands each) + our synthetic photometry.
    for j, b in enumerate(BANDS):
        table[f"Mag_{b}"] = G["Mag"][:, j].astype(np.float32)
        table[f"MagDust_{b}"] = G["MagDust"][:, j].astype(np.float32)
        table[f"synth_mag_{b}"] = coeffs_npz[f"synth_mag_{b}"].astype(np.float32)
        table[f"synth_magdust_{b}"] = coeffs_npz[f"synth_magdust_{b}"].astype(np.float32)

    table["n_galaxies"] = np.int64(N)
    table["source_sample"] = str(sample_file.name)

    # The synth_* columns above are *copied* from the coefficients file, so this
    # table carries only one filter-convolution convention. Carry the product's
    # own record forward, and refuse to build if the record is missing.
    if "convention" not in coeffs_npz:
        raise ValueError(
            f"{coeffs_file.name} predates the convention record, so its magnitudes "
            "cannot be attributed to a convention. Regenerate it with "
            "process_lgalaxies.py, which stores one. See "
            "documents/filter_convention.md.")
    table["convention"] = str(coeffs_npz["convention"])
    return table


def embed_pca_model(table: dict, pca_file: Path) -> None:
    """Fold the PCA basis into the table so a consumer needs no other data file."""
    pca = np.load(pca_file, allow_pickle=True)
    norm = pca["norm"].item() if pca["norm"].ndim == 0 else dict(pca["norm"])
    table["pca_components"] = pca["components"].astype(np.float64)
    table["pca_mean"] = pca["mean"].astype(np.float64)
    table["pca_wave"] = pca["wave"].astype(np.float64)
    table["pca_norm_std"] = np.asarray(norm["std"], dtype=np.float64)
    table["pca_norm_mean"] = np.asarray(norm["mean"], dtype=np.float64)
    table["pca_variance"] = pca["variance"].astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", choices=["MR", "MRII"], default="MR",
                    help="Which simulation tree this table is for (default: MR)")
    ap.add_argument("--lgal-root", type=Path, default=LGAL_ROOT_DEFAULT)
    ap.add_argument("--sample", type=Path, default=None,
                    help="L-GALAXIES sample .npy (default: Mil-I z=0 under --lgal-root)")
    ap.add_argument("--coeffs", type=Path,
                    default=PROJECT_ROOT / "data" / "lgalaxies_sed_coeffs_bc03.npz")
    ap.add_argument("--pca", type=Path,
                    default=PROJECT_ROOT / "data" / "pca_results_bc03.npz")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data")
    args = ap.parse_args()

    sample = args.sample or (
        args.lgal_root / "output" / "samples" /
        "Planck_Mil-I_snapshots_default_test3_z0.00-0.00_All.npy")

    for label, path in [("sample", sample), ("coeffs", args.coeffs), ("pca", args.pca)]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required {label} path is missing: {path}\n"
                "This builder must run where the external L-GALAXIES data lives.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    _p("=== Building per-galaxy property table ===")
    table = build_property_table(sample, args.coeffs)
    embed_pca_model(table, args.pca)
    table_out = args.out_dir / f"galaxy_table_{args.tree}.npz"
    np.savez_compressed(table_out, **table)
    _p(f"Wrote {table_out}  ({table_out.stat().st_size/1e6:.2f} MB, "
       f"{table['n_galaxies']:,} galaxies)")


if __name__ == "__main__":
    main()
