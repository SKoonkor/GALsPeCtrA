# GALsPeCtrA notebooks

A teaching ladder: single operations first, building up to a full pipeline run. The audience
is both users of GALsPeCtrA and the developer checking their own understanding.

**Five rungs exist today. Five are planned and not yet written** — some demonstrate stages that
do not exist yet. **Four of the five are unblocked** (see *Blockers*): the filter convolution
convention that gated them was settled on 7 August 2026 and every stored output regenerated.

Anything superseded lives in [`archive/`](archive/). Nothing is deleted.

---

## The merger tree

Every notebook that touches simulation data opens with the same two lines:

```python
from galspectra.trees import get_tree
TREE = get_tree("MR")      # Millennium-I;  "MRII" for Millennium-II
```

`src/galspectra/trees.py` is the single registry. It carries the sample file, the
coefficient product, box size, total file count, file range, DM particle mass, stellar-mass
cut and the L-GALAXIES output path pattern. The notebooks hold none of those values, and
`tests/test_notebook_tree_hygiene.py` enforces that.

**Labels are `"MR"` and `"MRII"` — exactly, with no aliases.** `"MRI"`, `"mr"` and
`"Millennium-I"` all raise. The spelling matches `process_lgalaxies.py --simulation`, so one
tree has one name across the project. Human-readable text comes from `TREE.pretty`.

Three things the registry deliberately does not own: `hubble_h`, which is identical for both
trees and is cosmology rather than tree identity; `PART_RES`, likewise; and simulation
*detection from a filename*, which stays in `process_lgalaxies.py`. The two compose —
`get_tree(detect_simulation(path))`.

`n_files_used` is derived from `file_range` rather than stored beside it, and
`effective_volume_mpc_h3` documents that it assumes the tree files partition the box into
equal volumes. **Notebook 08's luminosity-function amplitude scales directly with that
number**, so the assumption is stated where it can be seen.

**Selecting a tree whose data is absent fails immediately, naming the missing path.** There
is no fallback to the other tree, by design: a silent fallback would mean every number and
figure below quietly described a different simulation from the one in the heading.

Millennium-II merger trees live on an external drive and there is no `galaxy_table_MRII.npz`,
so `TREE = "MRII"` will often be unavailable. That is expected. On this machine today:

| | `TREE = "MRI"` | `TREE = "MRII"` |
|---|---|---|
| galaxy sample `.npy` | available | available |
| PCA coefficient product | available | available |
| L-GALAXIES binary output | available | **absent** — notebook 09 stops here |

---

## Execution models

| Model | Meaning |
|---|---|
| **run end-to-end** | Cheap. Execute the whole notebook from stored inputs; seconds to a minute. |
| **read stored outputs** | The inputs cost minutes to regenerate, so the notebook loads a product written by a script in `scripts/` and does not recompute it. |
| **guarded** | Needs an artefact that may be absent. Fails at the top with the missing path rather than part-way through. |

---

## The ladder

### Rungs that exist

| # | Notebook | Execution model | What it demonstrates |
|---|---|---|---|
| 01 | [`01_filters_and_photometry.ipynb`](01_filters_and_photometry.ipynb) | run end-to-end | Loading the 61 filter curves; pivot wavelengths and edges, and the PAUS red leak; one AB magnitude by hand then from the library; `BandOperator` as a fixed matrix, asserted bit-identical; both conventions side by side; and why L-GALAXIES' own tables disagree by ~7 mmag |
| 02 | [`02_ssp_library_and_pca_basis.ipynb`](02_ssp_library_and_pca_basis.ipynb) | run end-to-end (~15 s) | The SSP library and what "per unit formed mass" buys; the library's piecewise grid and why R = 300 beats it by 1.4x; the five permitted weighting schemes and the four refused ones, with the silent composition failure reproduced; fitting a basis and asserting composition is exact; **why the far-UV reconstructs badly and why no wavelength-only weight can fix it** (r = -0.989 with UV faintness); why 99.99 % of the variance leaves a 116 mmag tail; **reconstruction quality mapped across age and metallicity**; and which of the repo's two bases everything downstream actually uses |
| 04 | [`04_galaxy_sed_explorer.ipynb`](04_galaxy_sed_explorer.ipynb) | run end-to-end | Reconstructs individual galaxy spectra from stored PCA coefficients; SFH, nebular and dust layers; direct BC03 comparison; interactive explorer |
| 05 | [`05_validate_against_lgalaxies.ipynb`](05_validate_against_lgalaxies.ipynb) | run end-to-end | Synthetic *ugriz* against native `Mag`/`MagDust`; colour–colour; CMD count-difference; PCA coefficient distributions |
| 08 | [`08_luminosity_functions.ipynb`](08_luminosity_functions.ipynb) | run end-to-end (~1 min) | *ugriz* luminosity functions, L-GALAXIES vs reconstruction, intrinsic and dust, with an SDSS reference |
| 09 | [`09_incode_pca_verification.ipynb`](09_incode_pca_verification.ipynb) | guarded | The C path: coefficients written in-code at snapshot-output time vs Python post-processing vs native `Mag` |
| 11 | [`11_multiz_sed_explorer.ipynb`](11_multiz_sed_explorer.ipynb) | run end-to-end | Notebook 04's explorer across every output redshift of the multi-snapshot run: SFH bin structure per snapshot, rest- and observer-frame SEDs, *ugriz* against `Mag`/`ObsMag`. Redshifts at `z_phot` (WMAP7) and attenuates at `z_planck`, for the reason in `galspectra.lgalaxies.snapshots` |
| 12 | [`12_incode_pca_across_redshift.ipynb`](12_incode_pca_across_redshift.ipynb) | guarded | Notebook 09's in-code-vs-`Mag` comparison and colour diagnostics run at every output redshift, with a redshift selector; adds the colour-colour panel 09's heading promises but never draws, and the cross-redshift trend — agreement improves monotonically with z, the *g*-band bias falling from +0.031 to −0.005 mag |

Numbers 03, 06, 07 and 10 are free; the gaps are intentional.

### Rungs still to be written

| # | Notebook | Execution model | What it will demonstrate | What must land first |
|---|---|---|---|---|
| 03 | `03_sfh_to_spectrum.ipynb` | run end-to-end | The composition law `L = M·μ_eff + C·B`; the mass sandwich; a live check that composition is exact to machine precision and that dropping `M_total` breaks it by 98 % | Nothing — writable now |
| 06 | `06_colour_split_validation.ipynb` | read stored outputs | Why an aggregate offset hides a population-dependent one; the five colour definitions; the calibrated cut; and the correction — the offset is `FullSED` vs `PhotTables`, not the basis | **Unblocked 7 Aug 2026.** `data/validation_by_colour_stats.json` has been regenerated under the adopted convention |
| 07 | `07_error_vs_bytes_budget.ipynb` | read stored outputs | Error against bytes per galaxy; the weighting comparison; choosing N from the curve; the four baselines; D4000 via Renard Eqs. 1–5 | **Unblocked 7 Aug 2026.** `data/pca_experiments/bc03_R300_results.json` regenerated; its metrics moved by 0.1 % |
| 10 | `10_full_pipeline_demo.ipynb` | run end-to-end on a small slice | SSP grid → basis → SFH → coefficients → spectra → photometry → colours in one pass | Rungs 01–03, plus a decision on scope: a live run on ~50 galaxies, or a narrated tour of the production run |

### Stages the plan calls for that have no notebook yet

These appear in ResearchPlan_2026 §5 but are not implemented, so no rung is proposed until
the code exists: redshifting and apparent magnitudes, IGM attenuation, PAUS/CFHTLS/Euclid
narrow- and broad-band synthesis, CLOUDY-based nebular emission, and lightcone construction.

---

## Blockers

**The filter convolution convention — cleared 7 August 2026.** It used to gate most of the
ladder: integrating the BC03 `FullSED` files gave a *g−r* that was +0.001 mag redder at 0.1 Gyr
and **+0.050 at 13 Gyr** than the `PhotTables` L-GALAXIES itself reads, and until the choice was
made, any notebook quoting a magnitude would have taught a number that was about to change.

It was measured over 221 ages × 6 metallicities × 40 bands and **photon counting adopted**
(`documents/filter_convention.md`); every product has been regenerated. **Rung 01 is written**,
06 and 07 are unblocked, and 02, 03 always were. Rung 10 still waits on 02–03 plus a scope
decision.

The ~7 mmag that remained after the migration is no longer unexplained either: it is
**L-GALAXIES' own quadrature** (`documents/lgalaxies_quadrature.md`), and rung 01 shows it.
That has one consequence for every rung above: **agreement with the model's own `Mag` is a
sanity check, not a target to minimise**, because the target is itself approximate.

---

## Conventions

- **Terminology.** The coefficients L-GALAXIES writes are *in-code, output-time PCA spectral
  coefficients* — not an "on-the-fly SED". They live in `struct GALAXY_OUTPUT`, are computed
  in the `POST_PROCESS_MAGS` branch of `save.c`, and are never fed back into the physics. See
  `documents/onthefly_factcheck.md`, which uses the older phrase only in order to refute it,
  and ResearchPlan_2026 §2.2.
- **Stored cell outputs.** The four merged notebooks carry no outputs. They had been executed
  against Millennium-II, and leaving those figures under a notebook that now defaults to
  Millennium-I would show one simulation's results under another's heading. Re-executing
  restores them.

## Follow-ups, not done

| Item | Why it is deferred |
|---|---|
| `scripts/verify_pca_onthefly.py` | Hard-codes five Millennium-I values (`:26` output path, `:30` coeffs, `:32` sample, `:90` DM particle mass, and its `SM_res`), and its filename plus two lines still carry the "on-the-fly" phrasing. Would consume `galspectra.trees` |
| `scripts/validate_by_colour.py:70` | One hard-coded Millennium-I sample path |
| `scripts/build_galaxy_table.py` | `--tree` already selects `MR`/`MRII` by name and writes `galaxy_table_<TREE>.npz`, but `--sample` still defaults to the Millennium-I sample path. Would consume `galspectra.trees` for the default |
| `hubble_h = 0.673` in six places | `lgalaxies/sfh.py:31`, `photometry/dust.py:120,196`, `process_lgalaxies.py:379`, `build_galaxy_table.py`, `verify_pca_onthefly.py:89`. Not tree-dependent, so it does not belong in `trees.py`, but it deserves a single home of its own |
- **Numbering.** Two-digit prefix, gaps left. A new rung slots in as `05a` or takes a free
  number rather than forcing a renumber.

---

## Archive

[`archive/`](archive/) holds notebooks no longer part of the ladder. They still run; they are
simply not the path a reader should take.

| Notebook | Why |
|---|---|
| `hi_mass_function.ipynb`, `hi_mass_function_MRII.ipynb` | HI, H₂ and cold-gas mass functions. They import no `galspectra` code — they check the L-GALAXIES run, not GALsPeCtrA. ResearchPlan_2026 targets the *luminosity* function faint end; no paper targets gas mass functions |
| `luminosity_functions_MRII.ipynb` | Merged into 08. Never executed, and carried a Millennium-I box-size comment beside the Millennium-II value — a class of bug the `CFG` registry removes |
| `pca_onthefly_comparison.ipynb` | Merged into 09. Its Millennium-I configuration became the default; the Millennium-II variant had the better code (tree-file selector, absent-post-processing guard) |
| `validate_lgalaxies.ipynb` | Merged into 05. The Millennium-II variant's CMD count-difference panel replaced two separate CMDs |
| `visualise_seds.ipynb` | Merged into 04. Carried 13 empty cells and a stored `SyntaxError` |
