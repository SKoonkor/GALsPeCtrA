"""
Verify L-GALAXIES's in-code (output-time) PCA coefficients against post-processing.

Compares o->pca_coeffs[] written by L-GALAXIES (COMP_PCA_COEFFICIENTS) against
the PCA coefficients computed by process_lgalaxies.py for the same galaxies.

Note on terminology: L-GALAXIES computes o->pca_coeffs[] *in-code at snapshot-
output time* by convolving each galaxy's stored SFH with the PCA(age, Z) grid
(same code path as its post-processed magnitudes) — a compact spectral
representation, not a live SED state variable. See documents/onthefly_factcheck.md.

Usage:
    python scripts/verify_pca_onthefly.py

Both datasets must cover the same snapshot/redshift/file number.
"""

import sys
from pathlib import Path

import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2] / \
    "L-GALAXIES" / "LGalaxies2020_PublicRepository-master"
LGAL_OUTPUT = REPO_ROOT / "output" / "SA_DM_test3_z0.00_5"
LGAL_PYTHON  = REPO_ROOT / "AuxCode" / "Python"

GALSPECTRA_ROOT = Path(__file__).resolve().parents[1]
POSTPROC_FILE   = GALSPECTRA_ROOT / "data" / "lgalaxies_sed_coeffs_bc03.npz"
SAMPLE_FILE     = REPO_ROOT / "output" / "samples" / \
                  "Planck_Mil-I_snapshots_default_test3_z0.00-0.00_All.npy"


def load_lgal_binary(path, lgal_python_dir):
    """Read L-GALAXIES binary snapshot, skipping the variable-length header.

    Header layout: [int32: Ntrees] [int32: TotGals] [int32×Ntrees: per-tree counts]
    """
    sys.path.insert(0, str(lgal_python_dir))
    from LGalaxy_snapshots import LGalaxiesStruct

    with open(str(path), "rb") as f:
        n_trees = np.fromfile(f, np.int32, 1)[0]
        n_gals  = np.fromfile(f, np.int32, 1)[0]
        _       = np.fromfile(f, np.int32, n_trees)     # per-tree counts (discard)
        data    = np.fromfile(f, dtype=LGalaxiesStruct, count=n_gals)

    print(f"Binary output: {len(data):,} galaxies  "
          f"(Ntrees={n_trees}, dtype={LGalaxiesStruct.itemsize} B/gal)")
    return data


def main():
    # ── load in-code (output-time) coefficients from binary ───────────────
    if not LGAL_OUTPUT.exists():
        raise FileNotFoundError(f"L-GALAXIES output not found: {LGAL_OUTPUT}\n"
                                "Run with COMP_PCA_COEFFICIENTS enabled first.")
    gals = load_lgal_binary(LGAL_OUTPUT, LGAL_PYTHON)

    # Check pca_coeffs field exists and has non-zero data
    otf = gals["pca_coeffs"].astype(np.float64)   # (N, 50)
    n_nonzero = np.sum(np.any(otf != 0, axis=1))
    print(f"  Galaxies with non-zero pca_coeffs: {n_nonzero:,} / {len(gals):,}")

    # ── check post-processing file exists (loaded later alongside sample) ─
    if not POSTPROC_FILE.exists():
        raise FileNotFoundError(f"Post-processing output not found: {POSTPROC_FILE}\n"
                                "Run: python scripts/process_lgalaxies.py --backend bc03")

    # ── match galaxies by applying the same selection as the sample ──────
    # Both binary and sample come from the same tree file (same model run,
    # deterministic), so galaxies appear in the same order.  The sample is
    # the subset passing:
    #   StellarMass (Msun) >= 1e9  AND  Mvir_raw * 1e10 >= DM_part * part_res
    # where DM_part = 0.0961104e10 Msun/h (Mil-I particle mass), part_res = 20.
    #
    # We filter the binary with the same cuts; the resulting ordered list
    # should be identical to the sample.  We then match row-by-row using
    # StellarMass as a float32 exact-match guard.

    if not SAMPLE_FILE.exists():
        raise FileNotFoundError(f"Sample file not found: {SAMPLE_FILE}")
    sample = np.load(SAMPLE_FILE)
    pp = np.load(POSTPROC_FILE)
    pp_coeffs = pp["pca_coeffs"]
    print(f"Sample: {len(sample):,} galaxies  |  Post-processing: {len(pp_coeffs):,}")

    hub_h     = 0.673
    DM_part   = 0.0961104 * 1e10   # Msun/h  (Mil-I DM particle mass)
    part_res  = 20.0
    SM_res    = 1e9                  # Msun

    mvir_ok = gals["Mvir"].astype(np.float64) * 1e10 >= DM_part * part_res
    sm_ok   = gals["StellarMass"].astype(np.float64) * 1e10 / hub_h >= SM_res
    gals_sel = gals[mvir_ok & sm_ok]
    print(f"Binary after selection: {len(gals_sel):,}")

    # Ordered scan with one-step lookahead.
    # Both lists are in the same galaxy order; the only difference is ≤ a few
    # extra galaxies in one or the other at Mvir-cut boundaries.  When there's
    # a mismatch we check one step ahead in each list to decide which side has
    # the extra entry, then advance that pointer to re-sync.
    matched_otf = []
    matched_pp  = []
    otf_sel     = otf[mvir_ok & sm_ok]
    SM_TOL      = 1e-5   # relative tolerance on StellarMass float32 round-trip

    def sm_match(i_b, j_s):
        sm_b = float(gals_sel[i_b]["StellarMass"])
        sm_s = float(sample[j_s]["StellarMass"]) * hub_h / 1e10
        return abs(sm_b - sm_s) / (sm_b + 1e-30) < SM_TOL

    i = 0; j = 0
    while i < len(gals_sel) and j < len(sample):
        if sm_match(i, j):
            matched_otf.append(otf_sel[i])
            matched_pp.append(pp_coeffs[j])
            i += 1; j += 1
        elif j + 1 < len(sample) and sm_match(i, j + 1):
            j += 1          # extra galaxy in sample; skip it
        elif i + 1 < len(gals_sel) and sm_match(i + 1, j):
            i += 1          # extra galaxy in binary; skip it
        else:
            i += 1; j += 1  # genuine gap; skip both

    print(f"  Matched: {len(matched_otf):,}  |  unmatched in sample: {len(sample) - len(matched_otf)}")
    if len(matched_otf) < 100:
        print("  WARNING: fewer than 100 matches — results may be unreliable.")
        return

    otf_m = np.array(matched_otf)   # (N_match, 50)
    pp_m  = np.array(matched_pp)    # (N_match, 50)

    # ── comparison: Pearson r per PC component ────────────────────────────
    n_pc = otf_m.shape[1]
    pearson_r = np.array([
        np.corrcoef(otf_m[:, k], pp_m[:, k])[0, 1]
        for k in range(n_pc)
    ])

    print(f"\nPearson r per PC component (first 10):")
    for k in range(min(10, n_pc)):
        print(f"  PC{k+1:02d}: r = {pearson_r[k]:.6f}")
    print(f"  ...  median r = {np.median(pearson_r):.6f}  min r = {np.min(pearson_r):.6f}")

    # ── normalise by total mass (removes overall scale factor) ────────────
    # In-code (output-time) coefficients are mass-weighted sums; post-proc are the same.
    # Normalise each galaxy's coefficients by its L1-norm across PCs.
    def safe_normalise(arr):
        norm = np.linalg.norm(arr, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1.0, norm)
        return arr / norm

    otf_n = safe_normalise(otf_m)
    pp_n  = safe_normalise(pp_m)

    pearson_r_norm = np.array([
        np.corrcoef(otf_n[:, k], pp_n[:, k])[0, 1]
        for k in range(n_pc)
    ])

    print(f"\nAfter L2-normalisation:")
    for k in range(min(10, n_pc)):
        print(f"  PC{k+1:02d}: r = {pearson_r_norm[k]:.6f}")
    print(f"  ...  median r = {np.median(pearson_r_norm):.6f}  min r = {np.min(pearson_r_norm):.6f}")

    # ── fractional residuals on first PC ──────────────────────────────────
    mask = np.abs(pp_m[:, 0]) > 1e30  # non-zero
    if mask.sum() > 10:
        residual = (otf_m[mask, 0] - pp_m[mask, 0]) / (np.abs(pp_m[mask, 0]) + 1.0)
        print(f"\nFractional residual PC1 (non-negligible): "
              f"median={np.median(np.abs(residual)):.4e}  "
              f"90th pct={np.percentile(np.abs(residual), 90):.4e}")

    # ── summary ──────────────────────────────────────────────────────────
    if np.median(pearson_r) > 0.999:
        print(f"\n✓ PASS  median Pearson r = {np.median(pearson_r):.6f} (threshold 0.999)")
    else:
        print(f"\n✗ FAIL  median Pearson r = {np.median(pearson_r):.6f} (threshold 0.999)")


if __name__ == "__main__":
    main()
