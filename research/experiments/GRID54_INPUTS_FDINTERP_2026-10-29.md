# GRID54 inputs, field_d_interp habitat: 54 km obsroute input from a different Z cube (2026-10-29)

Inputs only; no MAP or refine was run. This mirrors research/experiments/GRID54_INPUTS_2026-10-29.md (revision 3,
the observer-route input `numpyro_input_s64_grid54_obsroute`) step for step. The only change is the habitat cube.
Interpretive nuance for any fit on these inputs: research/CONSIDERATIONS.md (G3 division of labour, M4, M6, M11,
M26; memory "regional trend is not dynamics").

**The cube is a masked EVALUATION build, not a production candidate.** Source:
`.research_runtime/community_z_repair_20261026/cubes/backcast40/field_d_interp_s1`, the `field_d` arm of
`research/exp/z_assim.py --summer interp` (ASSIM_FIELD_2026-10-27.md, "In-era interpolation test"). The field
was fitted on top of the masked `control_s1` DESK. The backcast40 spatial-holdout panel and its buffer were
withheld, and House Finch and the sealed species were excluded. Values at withheld places are predictions.
The summer sources are training points from 1996 on, plus pre-1996 rows outside the holdout and buffer.

## What was built, and where

All outputs are under `$HOUFIN_PROCESSED/grid54` = `/mnt/e/Datasets/houfin/grid54`. Nothing existing was
overwritten.

| Piece | How | Output |
|---|---|---|
| Z cube (64-d summer) | `research/exp/grid54_fdinterp_cube.py`. Takes columns 0-63 (the summer teacher space) of each 27 km 128-d `Z_latent_{year}.npy`, gathers the 27 km land cells and applies the SAVED 54 km averaging operator (`land_mask/averaging_operator.npz`; asserted equal to a fresh rebuild). The winter half (64-127) is dropped. | `encoder/cube_fdinterp_s64_grid54/` (228 MB) |
| cube_meta.json | The source has no cube_meta (z_assim's export writes none). I copied the control_s1 contract and amended it with: the source dir and per-file sha256s, ASSIM_DONE.json (the z_assim.py sha `ceed5f...` equals the current untracked file), `evaluation_build` (backcast40, spatial holdouts masked, House Finch and sealed species excluded), and the note that the arm overwrites control land cells in the same teacher spaces. That amended 27 km contract is saved as `source_cube_meta_27km.json` and embedded. The 54 km contract is `kernel: ruzicka`, `latent_dim: 64`, one `summer` block (backcast40 summer ESK basis), a `seasonal_selection` record, `coarsened_from`, the same mean-embedding note, and the control DESK checkpoint `32015e...`. | same dir |
| Z_disp | `generate_all_path_features.py` from the scratch snapshot `code54`, the same snapshot that built `latent_avian_paths_s64_grid54` (its generate, kernel, path, config and contract files are byte-identical to the live tree). Run on CPU (8.9 min) with the 54 km mask and cell-integrated kernels (S = 31). | `latent_avian_paths_fdinterp_s64_grid54/` (2.6 GB) |
| Numpyro input | `python -m src.data.combine.model_inputs` from the same patched `snap_d0e59ff` snapshot (`grid54/snapshot_d0e59ff_grid54_patch.diff`) that built `numpyro_input_s64_grid54_obsroute`. It used the existing `grid54/bbs/bbs_data_obsroute_for_python.npz` (snap, pinned holdout, 1,814 pseudo-zero sites) and the data overlay `config/overlays/grid54_data.json`. | `model_inputs/numpyro_input_fdinterp_s64_grid54_obsroute/` (1.7 GB) |
| Overlay | `config/overlays/grid54_age_fdinterp_obsroute.json` = `grid54_age_obsroute.json` with only these changed: `input_dir`, `raw_z_dir`, `path_features`, `path_diagnostics_dir`, `run_names.map` / `vis_map` (`age_map_{precision}_fdinterp_s64_grid54_obsroute`), plus a comment. No new data overlay was needed. | |

**Order of selection.** The 27 km clean-v2 input path-integrated all 128 columns and then selected 0-63. Here the
selection comes first. Both steps are linear and act per feature, so the order does not matter. This is
established: averaging then selecting differs from selecting then averaging by 0.0, and an impulse in one feature
reaches no other feature.

The local `model_inputs.py` raises on a 128-d `ruzicka_sum` cube. The snapshot ingest never saw one, because the
cube is 64-d `ruzicka`.

## Checks (`grid54/grid54_fdinterp_verify.json`, `research/exp/grid54_fdinterp_verify.py`)

**(1) Z (established).**
* The stored 54 km cube equals the block mean of the 27 km summer columns: maximum absolute difference 3.0e-8
  over all 124 years.
* The ingested Z_gathered equals the cube exactly (0.0).
* Mean-embedding identity z_bar_B . z_bar_B' = mean child-pair z_i . z_j: maximum relative error 9.1e-4 over
  3,500 sampled pairs. The maximum comes from float32 storage over pairs with near-zero similarity; the block
  mean itself is exact.
* \|z_bar\|^2 / mean \|z_i\|^2: q1 0.71, q5 0.82, q50 0.964, minimum 0.54. For s64 the values were 0.83 / 0.91 /
  0.987 / 0.59.
  * Mechanism (likely): the field's per-cell GMRF level offset adds sub-54 km heterogeneity, so averaging
    shrinks the norm more.

**(2) Z_disp.**
* `path_feature_meta` matches the s64 build on every key except `build_id` and the contract: dispersal spec,
  `kernel_discretization`, kernel mass 1.0000008, realized MDD 331.09 km, mask sha `8f0892...`, 20 steps.
* Z_raw in every npz equals the cube, and the ingested Z_disp equals the npz on land (0.0 on every third year).
* Recomputing 1990 with the operator reproduces the stored values exactly (0.0).
* Same operator as the s64 build: L(Z_fd − Z_s64) equals Z_disp_fd − Z_disp_s64 to 7.5e-5 relative.
* Against a 27 km Z_disp of the SAME cube, computed here because none exists on disk, block-meaned. This used
  the production 27 km point kernels, over 8 years from 1902 to 2025.
  * fdinterp: per-feature correlation minimum 0.9960, median 0.9981. Over all kernel x feature pairs the
    minimum is 0.990. Per-kernel relative RMS is 1.9-7.2%, largest for the 0-155 km bands.
  * s64 reference with the identical code: 0.9969 / 0.9986, minimum 0.991, 1.6-5.8%.
  * First I validated this code path: it reproduces the stored 27 km s64 Z_disp to 9.2e-5.
  * Mechanism for the slightly larger fdinterp RMS (likely): the same fine-scale heterogeneity as in (1), and
    the comparison includes the point-to-integrated kernel switch.
* **Pre-existing operator caveat (established; it affects every input, not only this one).**
  * The operator: `integrate_paths` divides the land-masked convolution by the land weight wherever that weight
    is above 1e-10.
  * Where a (cell, band) sends almost all its mass off-grid or to water, the land share is below 1e-4 at some
    integration step. The stored value there is FFT roundoff divided by roughly 0, so it is not a function of Z.
  * Scale: 181 of 51,816 land cell-kernel pairs at 54 km (168 cells), and 1,031 of 206,508 at 27 km.
  * There, additivity fails by up to 1.41, and 7 cells exceed the convex bound \|Z_disp\| <= max\|Z\| (up to
    1.45x in the s64 input and 1.42x here).
  * The checks above exclude these pairs. They are identical in the s64 inputs, because the same mask and
    kernels are used.
  * Production implication: a land-share floor (for example, set the value to 0 or to the local Z below
    1e-4) would remove the noise. It was not applied, because that would change the operator relative to every
    existing input.

**(3) Everything else is bit-identical to `numpyro_input_s64_grid54_obsroute` (established).**
* 44 metadata keys are identical, and 0 differ. They include:
  * observation order, cells, years, counts and quality;
  * `obs_route_idx`, `obs_obsroute_idx` and `obs_first_year`;
  * `inv_locations` (27,89), (27,88), (26,89), (28,88), `inv_timestep` 38 and `initpop_latent`;
  * the forward kernels and edge corrections;
  * the disease inputs and every prior spec.
* Excluded from the comparison: the ingest id, the Z paths, `path_feature_meta`, and `z_kernel_contract`.
  Only `desk_checkpoint`, `esk_basis_dir`, `feature_blocks` and `ensemble_*` differ inside the contract.
* Pseudo-zeros: 1,814 sites, 68,932 observations. Total observations: 192,123.
* The pinned 216 km holdout mask is identical: 35,050 withheld (18.24%).
* Fit-code load (`research/exp/grid54_logdensity_check.py`, CPU, 3.7 GB RSS, 22 s). Run against the clean
  `d0e59ff` archive (`grid54_fdinterp_logdensity_check.json`) and the current ladder fit code `f4c872d`
  (`..._f4c872d.json`). The two give identical numbers.
  * `load_data` passed, including the dispersal-spec check and the pinned holdout.
  * There are 28 latent sites, so the observer effect is on.
  * Both init points are finite: potential energy 260,977 for init_to_median (obs log-lik −242,146) and 217,140
    for init_to_sample (−189,553).

**(4) GPU memory (established by identity).** The Z and Z_disp tensors have the same shape and byte size as the
obsroute input: 137,070,592 B and 1,644,847,104 B; T 124, N 4,318, K 12, M 64. The measured MAP device peak on
that input is 3.20 GiB of 17.3 GiB (`grid54/runs/step3_obsroute/out/map.log`). It was not re-measured, because
the GPU is busy.

## How different is the habitat input? (54 km land cells, summer 64)

Coordinate correlations depend on the basis. fdinterp and control_s1 use the backcast40 summer teacher basis;
s64 uses `esk_all369_breed`. The isotropic coefficient prior makes the habitat GP depend on Z only through
Z.Z', so the Gram correlation and the cross-fitted linear-map R² are the relevant measures. The linear-map R² is
fitted on one 216 km checkerboard half and tested on the other. Change is the mean over 2010-2025 minus the mean
over 1902-1915, per cell.

| | fdinterp vs s64 | control_s1 vs s64 | fdinterp vs control_s1 |
|---|---|---|---|
| Level: pooled coordinate correlation | −0.05 | −0.04 | 0.92 |
| Level: Gram correlation (2000) | **0.924** | 0.942 | 0.979 |
| Level: linear-map R² (a→b / b→a) | 0.91 / 0.87 | 0.92 / 0.88 | 0.92 / 0.94 |
| Change: pooled coordinate correlation | 0.00 | 0.02 | 0.72 |
| Change: Gram correlation (centred) | **0.766** (0.775) | 0.596 (0.714) | 0.715 (0.800) |
| Change: per-cell \|Δz\| correlation (Spearman) | **0.79** (0.76) | 0.59 (0.60) | 0.77 (0.79) |
| Change: linear-map R² (a→b / b→a) | 0.84 / 0.78 | 0.84 / 0.81 | 0.95 / 0.91 |
| Mean \|Δz\| (a, b) | 0.246, 0.330 | 0.510, 0.330 | 0.246, 0.510 |

**Reading.**
* The spatial habitat geometry is very close to s64's: Gram correlation 0.92. The basis differs, so
  coordinate-wise comparisons are meaningless (established).
* The temporal change differs more: Gram correlation 0.77, and \|Δz\| correlation 0.79. Still, fdinterp's change
  is closer to s64's than its own covariate-only control's is (0.60).
* fdinterp's change is the smallest of the three: 0.25 against 0.33 for s64 and 0.51 for the control, in units
  where the level spread across cells is about 0.65.
* Mechanism (likely): the field recalibrates the control's covariate change (A ridge to I), and the observations
  pull it toward observed summer change. s64 was trained on all data.
* The 1902-1915 end is before any summer observation, so the early window is model extrapolation for every
  cube (M22, M26).
* Per M11 and "regional trend is not dynamics", the field's position x time term is predictive, not causal.

## Caveats

1. **Masked evaluation build.**
   * The control DESK and the field never saw the backcast40 holdout places or their buffers, so those places
     carry interpolated Z.
   * A fit on this input is not comparable to a production cube's fit at those places. It is a habitat
     sensitivity diagnostic.
   * House Finch is excluded from the source, so there is no target leakage. That is established from the
     z_assim design.
2. **Teacher basis.**
   * The basis is the backcast40 summer ESK basis (`teachers/backcast40/summer/spacetime`), not the production
     `esk_all369_breed`.
   * Coefficients from clean-v2 fits do not transfer, and starting a fit from a clean-v2 MAP is not meaningful.
   * The GP prior is rotation-invariant (likely, for this readout: `rank` 64 = full, `feature_columns` null), so
     fits are comparable at the level of fitted habitat.
3. **Paths.** `esk_basis_dir` and the source dir in the contract are local `/mnt/c/Dev/...` paths. Staging
   to TACC needs the input dir, the Z_disp dir (for provenance only; the fit reads `input_dir`) and the overlay.
4. **Overlay base.** The overlay is a mirror of step 3's `grid54_age_obsroute.json`. A replica of a later ladder
   step (step 4 amplitude, step 5 injection) needs that step's overlay with these paths swapped in.
5. **Not done.** No 27 km fdinterp input was built. The 27 km Z_disp used in check (2) was computed in memory
   for 8 years only. GPU memory was not re-measured.

## Files

* New code: `research/exp/grid54_fdinterp_cube.py`, `research/exp/grid54_fdinterp_verify.py`.
* New overlay: `config/overlays/grid54_age_fdinterp_obsroute.json`.
* Logs and run scripts are in the session scratchpad: `fd_pathfeat.log`, `fd_inputs.log`, `fd_logdens*.log`,
  `fd_verify_zdisp2.log`.
