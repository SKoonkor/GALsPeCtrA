"""
process_lgalaxies.py

Batch SED processing for the L-GALAXIES 2020 galaxy catalog.

For each galaxy in the pre-processed .npy sample:
  1. Extract the SFH (disk + bulge mass and metallicity per time bin)
  2. Build a composite stellar population (CSP) in PCA space by summing
     mass-weighted PCA coefficient vectors for each SFH bin
  3. Reconstruct the SED from PCA coefficients and scale to absolute flux at 10 pc
  4. Compute synthetic photometry in SDSS ugriz (intrinsic + dust-attenuated)

Outputs:
  data/lgalaxies_sed_coeffs_<backend>.npz

  Keys: pca_coeffs, galaxy_index, backend, n_components, failed_ids,
        synth_mag_{u,g,r,i,z}      — intrinsic (dust-free) absolute AB magnitudes
        synth_magdust_{u,g,r,i,z}  — dust-attenuated absolute AB magnitudes

Usage:
  cd /path/to/GALsPeCtrA
  python scripts/process_lgalaxies.py [--backend fsps|bc03] [--n-gals N] [--no-dust]
  python scripts/process_lgalaxies.py --add-nebular      # include nebular emission

Requirements:
  - PCA results file must exist (run run_pca.py first)
  - L-GALAXIES sample .npy file and Database_SFH_table.fits must be accessible
"""

import argparse
from pathlib import Path
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Path configuration
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LGAL_ROOT    = PROJECT_ROOT.parent / "L-GALAXIES" / "LGalaxies2020_PublicRepository-master"

SAMPLE_FILE = LGAL_ROOT / "output/samples/Planck_Mil-II_snapshots_default_test1_z0.00-0.00_All.npy"
SFH_FITS    = LGAL_ROOT / "AuxCode/Python/Database_SFH_table.fits"
FILTER_DIR  = LGAL_ROOT / "SpecPhotTables/Filters"

# Reference distance for absolute magnitudes (10 pc in cm)
_D_10PC_CM       = 3.0857e19
_FOUR_PI_10PC_SQ = 4.0 * np.pi * _D_10PC_CM ** 2   # 1.197e40 cm²

# Birth-cloud / young-star age threshold (matches L-GALAXIES)
_YOUNG_THRESHOLD_GYR = 0.010   # 10 Myr

# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Batch SED processing for L-GALAXIES output")
    p.add_argument("--backend", choices=["fsps", "bc03"], default="fsps")
    p.add_argument("--pca-file", default=None)
    p.add_argument("--n-gals",  type=int, default=None,
                   help="Process only the first N galaxies (for testing)")
    p.add_argument("--output",  default=None)
    p.add_argument("--filters", nargs="+", default=["u", "g", "r", "i", "z"])
    p.add_argument("--no-photometry", action="store_true")
    p.add_argument("--no-dust",       action="store_true",
                   help="Skip dust attenuation (compute intrinsic magnitudes only)")
    p.add_argument("--add-nebular",   action="store_true",
                   help="Add nebular emission lines + continuum (requires data/qh0_grid_bc03.npz)")
    p.add_argument("--f-ion",         type=float, default=1.0,
                   help="Ionizing photon absorption fraction, 0–1 (default 1.0 = no escape)")
    return p.parse_args()


def _reconstruct_absolute(coeffs, mass, components, mean, norm_meta):
    """Reconstruct SED and scale to flux at 10 pc.  Returns zeros for mass<=0."""
    if mass <= 0:
        return np.zeros_like(mean)
    from galspectra.csp.reconstruct import reconstruction_sed_from_pca
    sed_per_msun = reconstruction_sed_from_pca(coeffs / mass, components, mean, norm_meta)
    return sed_per_msun * mass / _FOUR_PI_10PC_SQ


def main():
    args = parse_args()

    # ── Resolve paths ────────────────────────────────────────────────────────
    if args.pca_file:
        pca_file = Path(args.pca_file)
    else:
        pca_file = PROJECT_ROOT / "data" / (
            "pca_results_fsps_v2.npz" if args.backend == "fsps"
            else "pca_results_bc03.npz"
        )
    _neb_suffix = "_nebular" if args.add_nebular else ""
    output_file = Path(args.output) if args.output else (
        PROJECT_ROOT / "data" / f"lgalaxies_sed_coeffs_{args.backend}{_neb_suffix}.npz"
    )

    # ── Load PCA ─────────────────────────────────────────────────────────────
    print(f"Loading PCA from {pca_file}")
    if not pca_file.exists():
        raise FileNotFoundError(
            f"PCA file not found: {pca_file}\n"
            "Run scripts/run_pca.py first."
        )
    pca = np.load(pca_file, allow_pickle=True)
    pca_coeffs      = pca["coeffs"]
    pca_components  = pca["components"]
    pca_mean        = pca["mean"]
    pca_wave        = pca["wave"]
    pca_params      = pca["params"]
    pca_param_names = list(pca["param_names"])
    norm_meta = pca.get("norm", None)
    if norm_meta is not None:
        norm_meta = norm_meta.item() if norm_meta.ndim == 0 else dict(norm_meta)
    print(f"  PCA shape: {pca_coeffs.shape}  ({len(pca_wave)} wavelength bins)")

    # ── Build interpolator ───────────────────────────────────────────────────
    from galspectra.utils.grid import build_param_grid
    from galspectra.csp.interpolator import PCACoefficientInterpolator
    ages_grid, Z_grid, coeff_grid = build_param_grid(pca_params, pca_coeffs, pca_param_names)
    interpolator = PCACoefficientInterpolator(ages_grid, Z_grid, coeff_grid)
    print(f"  Interpolator: {len(ages_grid)} ages × {len(Z_grid)} metallicities")

    # ── Load sample ──────────────────────────────────────────────────────────
    print(f"\nLoading galaxy sample from {SAMPLE_FILE}")
    from galspectra.lgalaxies import load_sample, load_sfh_table, extract_sfh
    G         = load_sample(SAMPLE_FILE)
    sfh_table = load_sfh_table(SFH_FITS)
    N_total   = len(G)
    N_proc    = min(args.n_gals or N_total, N_total)
    print(f"  {N_total:,} galaxies loaded; processing {N_proc:,}")

    # ── Load filters ─────────────────────────────────────────────────────────
    filters = {}
    if not args.no_photometry:
        from galspectra.photometry.filters import load_filters, SDSS_FILTER_FILES, EXTRA_FILTER_FILES
        all_names   = {**SDSS_FILTER_FILES, **EXTRA_FILTER_FILES}
        valid_names = [f for f in args.filters if f in all_names]
        if valid_names:
            filters = load_filters(FILTER_DIR, valid_names)
            print(f"  Loaded {len(filters)} filters: {valid_names}")
        else:
            print("  No valid filter names; skipping photometry")

    # ── Load dust module ─────────────────────────────────────────────────────
    apply_dust = False
    if filters and not args.no_dust:
        from galspectra.photometry.dust import apply_dust_to_seds
        apply_dust = True
        print(f"  Dust attenuation: enabled (L-GALAXIES Mathis 1983 model)")
    elif filters and args.no_dust:
        print(f"  Dust attenuation: disabled (--no-dust)")

    # ── Load nebular interpolator ────────────────────────────────────────────
    qh0_interp = None
    if args.add_nebular:
        qh0_file = PROJECT_ROOT / "data" / "qh0_grid_bc03.npz"
        if not qh0_file.exists():
            raise FileNotFoundError(
                f"Q(H0) grid not found: {qh0_file}\n"
                "Run: python scripts/precompute_qh0.py"
            )
        from galspectra.nebular.ionizing import build_qh0_interpolator
        from galspectra.nebular.emission import csp_nebular_sed as _csp_nebular_sed
        qh0_interp = build_qh0_interpolator(qh0_file)
        print(f"  Nebular emission: enabled  (f_ion = {args.f_ion:.2f})")

    # ── Imports ──────────────────────────────────────────────────────────────
    from galspectra.csp.builder import build_csp_coefficients
    from galspectra.csp.reconstruct import reconstruction_sed_from_pca
    if filters:
        from galspectra.photometry.synthetic import compute_ab_magnitudes

    # ── Allocate output arrays ────────────────────────────────────────────────
    N_pc = pca_coeffs.shape[1]
    all_pca_coeffs  = np.zeros((N_proc, N_pc), dtype=np.float32)
    all_synth_mags  = {name: np.full(N_proc, np.nan, dtype=np.float32) for name in filters}
    all_dust_mags   = {name: np.full(N_proc, np.nan, dtype=np.float32) for name in filters} if apply_dust else {}
    failed_ids      = []
    rng = np.random.default_rng(42)   # reproducible μ draws for birth cloud

    print(f"\nProcessing {N_proc:,} galaxies...")
    for i in range(N_proc):
        if i % max(1, N_proc // 20) == 0:
            print(f"  {i:6,} / {N_proc:,}  ({100*i/N_proc:.0f}%)", flush=True)

        # ── SFH extraction ────────────────────────────────────────────────────
        try:
            sfh = extract_sfh(G[i], sfh_table)
        except Exception:
            failed_ids.append(i)
            continue

        ages    = sfh["age_Gyr"]
        is_young = ages < _YOUNG_THRESHOLD_GYR
        is_old   = ~is_young

        # ── CSP coefficients ──────────────────────────────────────────────────
        try:
            disk_coeffs  = build_csp_coefficients(interpolator, ages,
                                                   sfh["logzsol_disk"], sfh["disk_mass"])
            bulge_coeffs = build_csp_coefficients(interpolator, ages,
                                                   sfh["logzsol_bulge"], sfh["bulge_mass"])
            total_coeffs = disk_coeffs + bulge_coeffs
        except Exception:
            failed_ids.append(i)
            continue

        all_pca_coeffs[i] = total_coeffs

        if not filters:
            continue

        # ── Photometry ────────────────────────────────────────────────────────
        try:
            total_mass = sfh["disk_mass"].sum() + sfh["bulge_mass"].sum()
            if total_mass <= 0:
                continue

            # Intrinsic (dust-free) total SED at 10 pc
            norm_coeffs  = total_coeffs / total_mass
            sed_per_msun = reconstruction_sed_from_pca(
                norm_coeffs, pca_components, pca_mean, norm_meta)
            sed_intrinsic = sed_per_msun * total_mass / _FOUR_PI_10PC_SQ

            # ── Nebular emission ──────────────────────────────────────────────
            sed_neb_disk   = np.zeros(len(pca_wave))
            sed_neb_bulge  = np.zeros(len(pca_wave))
            if qh0_interp is not None:
                _neb_d, _ = _csp_nebular_sed(
                    pca_wave, ages, sfh["logzsol_disk"],  sfh["disk_mass"],
                    qh0_interp, f_ion=args.f_ion)
                _neb_b, _ = _csp_nebular_sed(
                    pca_wave, ages, sfh["logzsol_bulge"], sfh["bulge_mass"],
                    qh0_interp, f_ion=args.f_ion)
                sed_neb_disk  = _neb_d / _FOUR_PI_10PC_SQ
                sed_neb_bulge = _neb_b / _FOUR_PI_10PC_SQ
                sed_intrinsic = sed_intrinsic + sed_neb_disk + sed_neb_bulge

            mags = compute_ab_magnitudes(pca_wave, sed_intrinsic, filters)
            for name, m in mags.items():
                all_synth_mags[name][i] = m

            # Dust-attenuated SED
            if apply_dust:
                # Build 4 component SEDs (young/old × disk/bulge)
                def _csp_coeffs_subset(mask):
                    d_mass = sfh["disk_mass"][mask];  b_mass = sfh["bulge_mass"][mask]
                    if d_mass.sum() <= 0 and b_mass.sum() <= 0:
                        return np.zeros(N_pc), np.zeros(N_pc)
                    dc = build_csp_coefficients(interpolator, ages[mask],
                                                sfh["logzsol_disk"][mask], d_mass) if d_mass.sum() > 0 else np.zeros(N_pc)
                    bc = build_csp_coefficients(interpolator, ages[mask],
                                                sfh["logzsol_bulge"][mask], b_mass) if b_mass.sum() > 0 else np.zeros(N_pc)
                    return dc, bc

                dc_young, bc_young = _csp_coeffs_subset(is_young)
                dc_old,   bc_old   = _csp_coeffs_subset(is_old)

                disk_young_mass  = sfh["disk_mass"][is_young].sum()
                disk_old_mass    = sfh["disk_mass"][is_old].sum()
                bulge_young_mass = sfh["bulge_mass"][is_young].sum()
                bulge_old_mass   = sfh["bulge_mass"][is_old].sum()

                sed_disk_young  = _reconstruct_absolute(dc_young, disk_young_mass,
                                                         pca_components, pca_mean, norm_meta)
                sed_disk_old    = _reconstruct_absolute(dc_old,   disk_old_mass,
                                                         pca_components, pca_mean, norm_meta)
                sed_bulge_young = _reconstruct_absolute(bc_young, bulge_young_mass,
                                                         pca_components, pca_mean, norm_meta)
                sed_bulge_old   = _reconstruct_absolute(bc_old,   bulge_old_mass,
                                                         pca_components, pca_mean, norm_meta)

                # Nebular emission is physically co-located with young stellar
                # regions, so it receives the same dust treatment: birth-cloud
                # + ISM for disk, fixed 0.5 factor for bulge.
                sed_dust = apply_dust_to_seds(
                    pca_wave,
                    sed_disk_old,
                    sed_disk_young  + sed_neb_disk,
                    sed_bulge_old,
                    sed_bulge_young + sed_neb_bulge,
                    float(G[i]["ColdGas"]),
                    float(G[i]["ColdGasRadius"]),
                    G[i]["MetalsColdGas"],
                    float(G[i]["CosInclination"]),
                    redshift=0.0,
                    hubble_h=0.673,
                    rng=rng,
                )

                dust_mags = compute_ab_magnitudes(pca_wave, sed_dust, filters)
                for name, m in dust_mags.items():
                    all_dust_mags[name][i] = m

        except Exception:
            pass   # photometry failure is non-fatal

    print(f"  Done.  {len(failed_ids)} galaxies skipped (SFH extraction failures).")

    # ── Save ─────────────────────────────────────────────────────────────────
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "pca_coeffs":   all_pca_coeffs,
        "galaxy_index": np.arange(N_proc, dtype=np.int32),
        "backend":      np.array(args.backend),
        "pca_file":     np.array(str(pca_file)),
        "n_components": np.array(N_pc),
        "failed_ids":   np.array(failed_ids, dtype=np.int32),
        "add_nebular":  np.array(args.add_nebular),
        "f_ion":        np.array(args.f_ion),
    }
    for name, arr in all_synth_mags.items():
        save_dict[f"synth_mag_{name}"] = arr
    for name, arr in all_dust_mags.items():
        save_dict[f"synth_magdust_{name}"] = arr

    np.savez(output_file, **save_dict)
    print(f"\nResults saved to {output_file}")
    print(f"  PCA coefficients: {all_pca_coeffs.shape}")
    if all_synth_mags:
        sample_r = all_synth_mags.get("r", list(all_synth_mags.values())[0])
        print(f"  Intrinsic r-band: median = {np.nanmedian(sample_r):.2f}  "
              f"(valid: {np.sum(~np.isnan(sample_r)):,})")
    if all_dust_mags:
        sample_rd = all_dust_mags.get("r", list(all_dust_mags.values())[0])
        print(f"  Dust-att. r-band: median = {np.nanmedian(sample_rd):.2f}  "
              f"(valid: {np.sum(~np.isnan(sample_rd)):,})")


if __name__ == "__main__":
    main()
