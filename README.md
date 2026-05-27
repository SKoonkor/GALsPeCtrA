# GALsPeCtrA

**GAL**axy **S**p**eCtrA** — a PCA-based pipeline for reconstructing synthetic galaxy spectra and computing synthetic photometry from star-formation histories.

The pipeline compresses a grid of simple stellar population (SSP) spectra into a compact PCA basis, then integrates any SFH into a composite stellar population (CSP) by summing mass-weighted PCA coefficient vectors. It is validated against the L-GALAXIES 2020 semi-analytic model catalog, reproducing L-GALAXIES `Mag` and `MagDust` fields to within ~0.01–0.03 mag.

---

## Overview

The workflow has four stages:

```
1. Generate SSP grid  →  2. Run PCA  →  3. Process galaxy catalog  →  4. Validate
```

| Stage | Script | Output |
|-------|--------|--------|
| Generate BC03 SSP grid | `scripts/generate_seds_bc03.py` | `data/sed_grid_bc03.npz` |
| Run PCA | `scripts/run_pca.py` | `data/pca_results_bc03.npz` |
| Process L-GALAXIES catalog | `scripts/process_lgalaxies.py` | `data/lgalaxies_sed_coeffs_bc03.npz` |
| Validate | `notebooks/validate_lgalaxies.ipynb` | plots in `data/` |

An FSPS backend is also available (`scripts/generate_seds.py`) for generating SSPs without the BC03 files.

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

For the FSPS backend, [python-fsps](https://dfm.io/python-fsps/current/) is also required (see [FSPS Installation](#fsps-installation) below).

Install the package and its dependencies:

```bash
git clone https://github.com/<your-username>/GALsPeCtrA.git
cd GALsPeCtrA
pip install -e .
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

The paths are set at the top of `scripts/process_lgalaxies.py` via `LGAL_ROOT`.

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
python scripts/run_pca.py \
  --input  data/sed_grid_bc03.npz \
  --output data/pca_results_bc03.npz \
  --n-components 50 \
  --wave-min 3000 --wave-max 10000
```

Fits a 50-component PCA on the optical (3000–10000 Å) portion of the SED grid. The first component captures 88.7% of variance; 50 components capture >99.999%.

### Step 3 — Process L-GALAXIES galaxies

```bash
python scripts/process_lgalaxies.py --backend bc03
```

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

Open `notebooks/validate_lgalaxies.ipynb` in Jupyter. The notebook compares synthetic photometry against L-GALAXIES stored `Mag` and `MagDust` fields across all five SDSS bands.

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

Tested on 10,191 valid z=0 galaxies from the L-GALAXIES Millennium-I Planck run:

| Comparison | Band | Offset | Scatter | Pearson r |
|---|---|---|---|---|
| Intrinsic (synth vs `Mag`) | r | +0.007 mag | 0.004 mag | 1.0000 |
| Dust-attenuated (synth vs `MagDust`) | r | +0.033 mag | 0.294 mag | 0.9720 |
| Intrinsic colour (synth vs `Mag`) | g−r | +0.024 mag | 0.011 mag | 1.0000 |
| Dust colour (synth vs `MagDust`) | g−r | +0.041 mag | 0.031 mag | 0.9826 |

The ~0.3 mag scatter in dust-attenuated magnitudes comes from stochastic per-galaxy sampling of the birth-cloud parameter μ.

---

## Project structure

```
configs/                  YAML configs for SSP grid generation
  sed_generator_bc03.yaml   BC03 grid parameters and file paths
  sed_generator_config.yaml FSPS grid parameters

scripts/
  generate_seds_bc03.py   Build BC03 SSP SED grid
  generate_seds.py        Build FSPS SSP SED grid
  run_pca.py              Fit PCA on SED grid
  process_lgalaxies.py    Batch SED processing for L-GALAXIES catalog
  run_csp.py              Interactive CSP construction (development/testing)

src/galspectra/
  sps/        SSP backends (BC03, FSPS)
  sampling/   Parameter grid generators (Cartesian, LHS)
  sed/        SED I/O and FSPS generation
  pca/        PCA fitting and preprocessing
  csp/        CSP builder, interpolator, SFH utilities
  lgalaxies/  L-GALAXIES catalog reader and SFH extractor
  photometry/ Filter loading, synthetic photometry, dust attenuation
  utils/      Age conversion, grid utilities

notebooks/
  validate_lgalaxies.ipynb  Validation against L-GALAXIES Mag/MagDust
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
