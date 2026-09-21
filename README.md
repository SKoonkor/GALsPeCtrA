# GALsPeCtrA

**GAL**axy s**P**e**C**tr**A**: a PCA-based pipeline for reconstructing synthetic
galaxy spectra and computing synthetic photometry from star-formation
histories, validated against the L-GALAXIES 2020 semi-analytic model catalog.

This is **NOT** yet complete, it will be also aimed be implemented in the GALFORM galaxy simulation.

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

```bash
git clone https://github.com/SKoonkor/GALsPeCtrA.git
cd GALsPeCtrA
pip install -e .
```

Optional extras: `pip install -e ".[dev]"` (pytest, for the 156-test suite);
`pip install -e ".[colour-cut]"` (h5py + pandas, for
`scripts/colour_cut_calibration.py` only). The FSPS backend needs
`pip install fsps` with `SPS_HOME` set — see the
[python-fsps docs](https://dfm.io/python-fsps/current/).

---

## Data requirements

### BC03 backend

Reads Bruzual & Charlot (2003) FullSED files from the L-GALAXIES 2020 public
repository:

```
LGalaxies2020_PublicRepository-master/SpecPhotTables/FullSEDs/
    BC03_Chabrier_FullSED_m0.0001.dat
    BC03_Chabrier_FullSED_m0.0004.dat
    BC03_Chabrier_FullSED_m0.0040.dat
    BC03_Chabrier_FullSED_m0.0080.dat
    BC03_Chabrier_FullSED_m0.0200.dat
    BC03_Chabrier_FullSED_m0.0500.dat
```

Path is set in `configs/sed_generator_bc03.yaml`:

```yaml
bc03:
  dir: /path/to/LGalaxies2020_PublicRepository-master/SpecPhotTables/FullSEDs
```

### L-GALAXIES catalog (for `process_lgalaxies.py`)

```
LGalaxies2020_PublicRepository-master/output/samples/
    Planck_Mil-I_snapshots_default_test3_z0.00-0.00_All.npy

LGalaxies2020_PublicRepository-master/AuxCode/Python/
    Database_SFH_table.fits
```

`LGAL_ROOT` at the top of `scripts/process_lgalaxies.py` locates the SFH table
and filter directory. `--sample` and `--output` are required, with no defaults.

---

## Usage

Run all scripts from the project root.

### 1 Generate the SSP SED grid

```bash
python scripts/generate_seds_bc03.py
```

Output: `data/sed_grid_bc03.npz`. Flags: `--bc03-dir`, `--n-ages` (default 200),
`--wave-min`/`--wave-max` (default 912–25000 Å), `--output`.

An FSPS backend is also available: `python scripts/generate_seds.py`.

### 2 Run PCA

```bash
python scripts/run_pca_evalgrid.py \
  --input  data/sed_grid_bc03.npz \
  --output data/pca_results_bc03.npz \
  --n-components 50 \
  --wave-min 1048 --wave-max 23552 --resolution 300
```

Produces the basis used everywhere downstream, including the L-GALAXIES export.
Use this script, not `scripts/run_pca.py` (a native-grid general-purpose fitter
that the committed basis is not built from).

### 3 Process an L-GALAXIES catalog

```bash
python scripts/process_lgalaxies.py --backend bc03 \
  --sample "$LGAL/output/samples/Planck_Mil-I_snapshots_default_test3_z0.00-0.00_All.npy" \
  --output data/lgalaxies_sed_coeffs_bc03.npz
```

Refuses to overwrite an existing output without `--overwrite`, and refuses to
run if `--sample`/`--output` name different simulations. Flags: `--n-gals N`,
`--no-dust`, `--no-photometry`, `--filters u g r`, `--redshift` (override the
dust-model redshift).

For every output redshift in a run: `scripts/run_multiz_coeffs.sh` loops this
script over all of them.

### 4 Validate

```bash
jupyter notebook notebooks/05_validate_against_lgalaxies.ipynb
```

See `notebooks/README.md` for the full notebook ladder.

### L-GALAXIES integration

`scripts/export_pca_for_lgalaxies.py` writes the PCA(age, Z) grid as a binary
L-GALAXIES reads directly, computing PCA coefficients in-code at output time.
`scripts/verify_pca_onthefly.py` checks those against the Python path. See
`documents/onthefly_factcheck.md` for terminology notes.

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
