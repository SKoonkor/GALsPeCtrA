# GALsPeCtrA

**GAL**axy **S**p**eCtrA** — a PCA-based pipeline for reconstructing synthetic galaxy spectra and computing synthetic photometry from star-formation histories.

The pipeline compresses a grid of simple stellar population (SSP) spectra into a compact PCA basis, then integrates any SFH into a composite stellar population (CSP) by summing mass-weighted PCA coefficient vectors. It is validated against the L-GALAXIES 2020 semi-analytic model catalog, reproducing L-GALAXIES `Mag` and `MagDust` fields to within **~0.001–0.017 mag** in the intrinsic *ugriz* bands.

---

## Overview

The workflow has four stages:

```
1. Generate SSP grid  →  2. Run PCA  →  3. Process galaxy catalog  →  4. Validate
```

| Stage | Script | Output |
|-------|--------|--------|
| Generate BC03 SSP grid | `scripts/generate_seds_bc03.py` | `data/sed_grid_bc03.npz` |
| Run PCA | `scripts/run_pca_evalgrid.py` | `data/pca_results_bc03.npz` |
| Process L-GALAXIES catalog | `scripts/process_lgalaxies.py` | `data/lgalaxies_sed_coeffs_bc03.npz` |
| Validate | `notebooks/05_validate_against_lgalaxies.ipynb` | plots in `data/` |

An FSPS backend is also available (`scripts/generate_seds.py`) for generating SSPs without the BC03 files.

### In-code L-GALAXIES integration

The PCA basis can also be handed to L-GALAXIES itself: `scripts/export_pca_for_lgalaxies.py` writes the PCA(age, Z) coefficient grid as a C-readable binary, and L-GALAXIES (built with `COMP_PCA_COEFFICIENTS`) fills a 50-element `pca_coeffs` output field **in-code at snapshot-output time** by convolving each galaxy's stored SFH with that grid — the same code path as its post-processed magnitudes. `scripts/verify_pca_onthefly.py` confirms these match the Python post-processing (Pearson r = 1.000000).

This is a compact spectral **representation** computed at output time, **not** a live/wavelength-resolved SED carried as a state variable during the run. See [`documents/onthefly_factcheck.md`](documents/onthefly_factcheck.md) for the full analysis and the standard terminology.

---

## Requirements

```
numpy
scipy
scikit-learn
pyyaml
astropy
matplotlib
```

For the FSPS backend, [python-fsps](https://dfm.io/python-fsps/current/) is also required (see [FSPS Installation](#fsps-installation) below). `scripts/colour_cut_calibration.py` additionally needs `h5py` and `pandas` (to read the GALFORM lightcone HDF5 and a comparison table); install with `pip install -e ".[colour-cut]"`.

Install the package and its dependencies:

```bash
git clone https://github.com/SKoonkor/GALsPeCtrA.git
cd GALsPeCtrA
pip install -e .
```

The test suite (156 tests) needs `pytest`:

```bash
pip install -e ".[dev]"
pytest
```

---

## Data requirements

### BC03 backend

The BC03 backend reads Bruzual & Charlot (2003) FullSED files from the L-GALAXIES 2020 public repository:

```
LGalaxies2020_PublicRepository-master/SpecPhotTables/FullSEDs/
    BC03_Chabrier_FullSED_m0.0001.dat
    BC03_Chabrier_FullSED_m0.0004.dat
    BC03_Chabrier_FullSED_m0.0040.dat
    BC03_Chabrier_FullSED_m0.0080.dat
    BC03_Chabrier_FullSED_m0.0200.dat
    BC03_Chabrier_FullSED_m0.0500.dat
```

Update the path in `configs/sed_generator_bc03.yaml`:

```yaml
bc03:
  dir: /path/to/LGalaxies2020_PublicRepository-master/SpecPhotTables/FullSEDs
```

### L-GALAXIES catalog (for `process_lgalaxies.py`)

The batch processor also requires the pre-processed `.npy` sample and the SFH timing table:

```
LGalaxies2020_PublicRepository-master/output/samples/
    Planck_Mil-I_snapshots_default_test3_z0.00-0.00_All.npy

LGalaxies2020_PublicRepository-master/AuxCode/Python/
    Database_SFH_table.fits
```

`LGAL_ROOT` at the top of `scripts/process_lgalaxies.py` locates the SFH table and the
filter directory. The **sample** and **output** paths are not defaults — they must be
passed explicitly with `--sample` and `--output` (see Usage below).

---

## Usage

All scripts should be run from the project root:

```bash
cd /path/to/GALsPeCtrA
```

### Step 1 — Generate the SSP SED grid (BC03)

```bash
python scripts/generate_seds_bc03.py
```

This reads the six BC03 FullSED files and builds a 200 × 6 Cartesian grid (200 log-spaced ages from 0.1 Myr to 13.7 Gyr × 6 native BC03 metallicities). Output: `data/sed_grid_bc03.npz`.

Optional flags:
```
--bc03-dir   Path to FullSEDs directory (overrides config)
--n-ages     Number of age points (default 200)
--wave-min   Minimum wavelength in Å (default 912)
--wave-max   Maximum wavelength in Å (default 25000)
--output     Output file path
```

### Step 2 — Run PCA

```bash
python scripts/run_pca_evalgrid.py \
  --input  data/sed_grid_bc03.npz \
  --output data/pca_results_bc03.npz \
  --n-components 50 \
  --wave-min 1048 --wave-max 23552 --resolution 300
```

This is the script that produced the committed `data/pca_results_bc03.npz`, and the
one whose coefficient grid `scripts/export_pca_for_lgalaxies.py` exports to
L-GALAXIES. It first **resamples** the native BC03 grid, flux-conservingly, onto a
log-λ evaluation grid at resolving power R = 300 spanning **1048–23552 Å — the union
of rest-frame wavelength ranges every photometric band used in this pipeline needs
over its full redshift range** — then fits a 50-component PCA on that resampled grid
under `inverse_std` weighting (dividing each wavelength bin by its standard
deviation across the library, divided back out after reconstruction). Measured from
the committed file: it spans **934 bins**, and the first component captures
**69.22%** of variance, 10 components reach 99.9955%, and 50 components reach
99.99999847%.

**The component count and weighting scheme are chosen from an error-against-storage
curve, not from this variance figure.** Cumulative explained variance saturates to
within 10⁻⁴ of unity by ~15 components, where the 99th-percentile narrow-band
photometric error is still two orders of magnitude above the target used here; the
curve is built in `scripts/pca_basis_experiment.py` and documented in
`documents/error_vs_bytes.md`. `inverse_std` weighting must be applied **per
wavelength bin**, not per spectrum — a per-spectrum rescaling would depend on which
SSP is being reconstructed and so could not be undone by a single operation after a
mass-weighted composite population is summed from many SSPs (see
`src/galspectra/pca/weighting.py`).

**Reconstruction needs 51 numbers, not 50.** The mean spectrum term must be scaled by
the same total formed stellar mass as every coefficient, so that mass has to be
carried alongside the 50 `pca_coeffs` — it is not derivable from them alone.

Reconstruction accuracy is strongly wavelength-dependent: inside the 1048–23552 Å
evaluation range it is good to well under 1% median in real photometric bands (see
Validation results below); shortward of roughly 2800 Å, old and metal-rich
populations reconstruct poorly and can go negative, because per-wavelength
standardisation gives every bin equal weight regardless of how little flux an old
population has there. Harmless for the z=0 *ugriz* validation in this README; would
need addressing before using this basis at redshifts high enough to bring the
far-ultraviolet into an observed band.

`scripts/run_pca.py` (native BC03 grid, no resampling, 3000–10000 Å default window) is
also available as a general-purpose fitter, but it is **not** what the committed basis
or the L-GALAXIES export are built from.

### Step 3 — Process L-GALAXIES galaxies

```bash
python scripts/process_lgalaxies.py --backend bc03 \
  --sample "$LGAL/output/samples/Planck_Mil-I_snapshots_default_test3_z0.00-0.00_All.npy" \
  --output data/lgalaxies_sed_coeffs_bc03.npz
```

`--sample` and `--output` are required and have no defaults. The script refuses to
overwrite an existing output without `--overwrite`, and refuses to run if the two
filenames refer to different simulations.

The dust column density carries `n_H ∝ (1+z)⁻¹`, as `code/post_process_spec_mags.c:417` does, so the
script reads the sample's `SnapNum` and uses that snapshot's **Planck** redshift. Override
with `--redshift` only for a deliberate experiment; the value used is recorded in the
product as `dust_redshift`.

The L-GALAXIES run writes one sample per output redshift.
`scripts/run_multiz_coeffs.sh` loops this script over all of them, naming each product
`lgalaxies_sed_coeffs_bc03_z<tag>.npz` — except z = 0, which keeps the bare name above.
`notebooks/11_multiz_sed_explorer.ipynb` consumes the set.

For each of the 11,328 z=0 galaxies in the sample:
1. Extracts the disk + bulge SFH (mass and metallicity per time bin)
2. Builds the CSP by summing mass-weighted PCA coefficient vectors
3. Reconstructs the SED and scales to absolute flux at 10 pc
4. Computes synthetic SDSS ugriz photometry (intrinsic and dust-attenuated)

Output: `data/lgalaxies_sed_coeffs_bc03.npz`

Optional flags:
```
--n-gals N      Process only the first N galaxies (for testing)
--no-dust       Skip dust attenuation (intrinsic magnitudes only)
--no-photometry Skip photometry entirely (PCA coefficients only)
--filters u g r Restrict to specific SDSS bands
--output FILE   Custom output path
```

### Step 4 — Validate

Open `notebooks/05_validate_against_lgalaxies.ipynb` in Jupyter. The notebook compares synthetic photometry against L-GALAXIES stored `Mag` and `MagDust` fields across all five SDSS bands. See `notebooks/README.md` for the full notebook ladder.

---

## Dust model

The dust attenuation follows the same two-component model used in L-GALAXIES (`model_dust.c`):

**ISM component** (all disk stars):
- Mathis, Mezger & Panagia (1983) extinction curve
- Metallicity-dependent opacity scaled by `(Z_gas / Z_sun)^s` where `s = 1.35` for λ < 0.2 μm and `s = 1.6` for λ ≥ 0.2 μm
- Slab geometry attenuation: `a_λ = (1 − exp(−τ_λ)) / τ_λ`

**Birth-cloud component** (disk stars younger than 10 Myr):
- Power-law optical depth: `τ_BC ∝ λ^(−0.7)`
- Amplitude set by `τ_V_BC = τ_V_ISM × (1/μ − 1)`, where μ is drawn from Gaussian(0.3, 0.2)

Young bulge stars receive a fixed birth-cloud attenuation factor of 0.5 (constant over wavelength). Old bulge stars are dust-free.

---

## Validation results

Tested on 10,191 valid z=0 galaxies from the L-GALAXIES Millennium-I Planck run. "Scatter" is the standard deviation of `synth − native` (not RMS, which is larger for a non-zero-median distribution):

| Comparison | Band | Median offset | Scatter (std) | Pearson r |
|---|---|---|---|---|
| Intrinsic (synth vs `Mag`) | r | +0.004 mag | 0.002 mag | 1.0000 |
| Dust-attenuated (synth vs `MagDust`) | r | +0.030 mag | **0.114 mag** | 0.9720 |
| Intrinsic colour (synth vs `Mag`) | g−r | +0.013 mag | 0.006 mag | 1.0000 |
| Dust colour (synth vs `MagDust`) | g−r | +0.025 mag | 0.030 mag | 0.9823 |

Per band, the intrinsic offsets run from **+0.0008 mag in *z* to +0.017 mag in *g***.

**The dust scatter figure is quoted over gas-bearing galaxies.** Over the whole sample it
is 0.294 mag, but that number is dominated by **209 galaxies (2.1 %) whose native `MagDust`
is numerically invalid** — L-GALAXIES assigns them up to 4.30 mag of attenuation despite
`ColdGas ≤ 0`, while GALsPeCtrA correctly assigns none. Excluding them, the scatter is
0.114 mag, and it really is the stochastic birth-cloud parameter μ: it correlates with
optical depth at Spearman ρ = +0.839 against a flat intrinsic control (ρ = −0.008).
See `documents/validation_by_colour.md` §4.

*Numbers regenerated 7 August 2026 under the photon-counting filter convention
(`documents/filter_convention.md`). Every band improved by 1.7–4.4×; the previous
energy-weighted values were +0.007 / +0.033 / +0.024 / +0.041. Re-verified 21
September 2026 against the evaluation-grid basis of Step 2 above — unchanged at the
precision shown here.*

---

## Project structure

```
configs/                     YAML configs for SSP grid generation and PCA basis experiments
  sed_generator_bc03.yaml      BC03 grid parameters and file paths
  sed_generator_config.yaml    FSPS grid parameters
  pca_experiments/              per-experiment configs for pca_basis_experiment.py

scripts/
  generate_seds_bc03.py       Build BC03 SSP SED grid
  generate_seds.py            Build FSPS SSP SED grid
  run_pca_evalgrid.py         Fit the production PCA basis (evaluation grid — Step 2 above)
  run_pca.py                  Fit PCA on the native BC03 grid (general-purpose, not production)
  process_lgalaxies.py        Batch SED + photometry processing for an L-GALAXIES catalog
  run_multiz_coeffs.sh        Loop process_lgalaxies.py over every output redshift
  export_pca_for_lgalaxies.py Write the PCA(age, Z) grid as a C-readable binary for L-GALAXIES
  verify_pca_onthefly.py      Check L-GALAXIES' in-code coefficients against the Python path
  validate_by_colour.py       Colour-split validation against native L-GALAXIES photometry
  validate_redshifting.py     Check observer-frame magnitudes against L-GALAXIES' ObsMag
  colour_cut_calibration.py   Locate the red/blue colour-bimodality valley (needs `.[colour-cut]`)
  pca_basis_experiment.py     Error-against-storage-cost curves for component count/weighting
  build_galaxy_table.py       Join sample + coefficients + basis into one small table for the above
  precompute_qh0.py           Ionizing-photon rate grid for nebular emission
  run_csp.py                  Interactive CSP construction (development/testing)

src/galspectra/
  sps/          SSP backends (BC03, FSPS)
  sampling/     Parameter grid generators (Cartesian, LHS)
  sed/          SED I/O and FSPS generation
  pca/          PCA fitting, preprocessing, resampling, weighting, basis, metrics
  csp/          CSP builder, interpolator, SFH utilities, mass-weight matrix
  lgalaxies/    L-GALAXIES catalog reader and SFH extractor
  photometry/   Filter loading, synthetic photometry, dust attenuation, redshifting
  nebular/      Ionizing-photon rate and nebular emission
  plotting/     Figure builders for the basis-experiment and validation scripts
  trees.py      Per-simulation registry (MR / MRII paths, box size, particle mass, ...)
  cosmology.py  L-GALAXIES' own cosmological parameters and distances
  utils/        Age conversion, grid utilities

notebooks/       See notebooks/README.md for the full ladder and execution model of each.
```

---

## FSPS Installation

The FSPS backend requires [FSPS](https://github.com/cconroy20/fsps) and its Python bindings.

1. Clone and compile FSPS:
```bash
git clone https://github.com/cconroy20/fsps
cd fsps
make
```

2. Set the environment variable:
```bash
export SPS_HOME=/path/to/fsps
```

3. Install Python bindings:
```bash
pip install fsps
```

4. Verify:
```python
import fsps
sp = fsps.StellarPopulation()
wave, spec = sp.get_spectrum(tage=1.0)
```
