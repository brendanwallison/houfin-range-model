# Habitat-input provenance for LA1 (J200 V&L-faithful, native-MX)

Traced 2026-11-08 (read-only). Status tags: **[E]** established (measured, or quoted with the evidence path),
**[L]** likely (inferred), **[U]** unknown. Paths: `G` = `/mnt/e/Datasets/houfin/grid54` (= `E:\Datasets\houfin\grid54`;
`/work/07980/bwa386/ls6/houfin/processed/grid54` is a symlink to it in WSL). `K` = `G/sandbox/KSENS_local`.
`SP` = the session scratchpad `C:\Users\brend\AppData\Local\Temp\claude\23b2334d1d63b197\1d135948-c54f-4d07-94f0-cc6b964cfcd5\scratchpad`.
Helper scripts and raw outputs: `SP\desk_prov\` (`meta_diff.*`, `repro.*`, `hull_rebuild.*`, `sha.*`).

**Headline.**
- The Z in LA1 is **not** the `cube_s64_grid54` DESK cube named by the overlay's `path_features.input_dir`. It is the
  **FIELD54 production cube `field54prod_ema` (v1 states)**, built as native 54 km DESK (EMA) → z_assim field correction
  → summer dims 0-63. Z_gathered equals that cube exactly (max |diff| 0.0), and its correlation with `cube_s64_grid54`
  is only .17-.19 [E].
- No 27→54 km coarsening step is in this chain [E].
- Every step after the DESK/field cube can be reproduced, or was reproduced here, except DESK training itself. Its code
  (the frozen `coord_code`) exists only on TACC, at no git commit reachable from this repo [U].
- Two steps are not reproducible from the current worktree:
  - the path features: worktree code has no land-share floor and no cell-integrated path kernel;
  - `_vl`: the V&L-faithful kernel exists only in a scratchpad script.

## (a) Provenance chain

| # | Stage | Path | Files / size / sha256 (first 8) | Date | Producing code + version | Evidence | Status |
|---|---|---|---|---|---|---|---|
| 0 | ESK teachers (Ružička → 64-D uncentred Nyström) | TACC `research/field54/runs/field54_prod/teachers/production/{summer,winter_full}/spacetime` | summer: `esk_landmarks.npy` 42,560,128 B sha **da9e1137**, `esk_projmat.npy` sha 8f92a267, `meta.json` sha 21510b4a; winter_full: landmarks 37,504,128 B sha 3e101759 | 2026-10-30 | `z_repair_teacher` recipe (16,000 landmarks, 64 dims, augment half-width 2 yr, seed 20261026) via `bc_teachers.slurm` | `research/experiments/FIELD54_OBS_TEACHERS_2026-10-30.md:118-125`; basis_identity in `G/kernel_sens/J200/latent_avian_paths/path_feature_meta.json` | E (identity); files TACC-only |
| 1 | DESK production (native 54 km) | TACC `field54_prod/models/production/interp_s1` → `cubes/production/desk_interp_s1{,_ema}` | `env_model_semisup.pth` 10,577,051 B sha **81c912a6**; `desk_meta.npz` 44,611 B sha **83e3c229** | 2026-10-30 | frozen `coord_code` (copy of `research/z_repair_20261026/coord_code`), `z_coord_run.py run --arm interp`, then `z_ema_cube.py`; task `011_field54_desk_prod.sh` (A100) | `research/experiments/FIELD54_DESK_2026-10-30.md:84-118`; `research/experiments/field54_obs_teachers_20261030/011_field54_desk_prod.sh`; sha in `cube_meta.json` | E (identity, recipe); code commit U |
| 2 | Field assimilation (z_assim `field_d`, production) | TACC `field54_prod/cubes/production/field_d_prod_ema_s1` (128-D) | sha not available locally | 2026-10-30 (job e4 3543710) | `research/exp/z_assim.py fit --task production --production --arm field_d --summer train --control-cube desk_interp_s1_ema`; same as main repo `research/exp/z_assim.py` (= code_bc) | `FIELD54_FIELD_2026-10-30.md:176-178,217-221`; `assim_arm` block of cube_meta | E |
| 3 | s64 cube (summer dims 0-63) | TACC `processed/field54/stageF/encoder/cube_field54prod_ema_s64`; local copy `K/cube_field54prod_ema_s64` | 124 × `Z_latent_<yr>.npy` (67×112×64 f32); `cube_meta.json` sha 6eba3a01; source digest bc60f67f | 2026-10-30 (Stage F job 3543749); local copy 10-30 21:35 | `research/exp/field54_s64_cube.py` via `f_inputs.slurm` (code_bc) | `FIELD54_FIELD_2026-10-30.md:111-126,217-221`; `K/code_bc/research/field54/f_inputs.slurm` | E |
| — | (NOT in chain) `G/encoder/cube_s64_grid54` | 27 km DESK ensemble (`desk_final_clean_v2` s1+s2, `esk_all369_breed`) coarsened by `coarsen_model_grid.py --factor 2` | — | 2026-10-29 | — | its `cube_meta.json` `coarsened_from`; Z_gathered vs it: max \|d\| .95-1.28, corr .17-.19 (`SP\desk_prov\repro.out` [4]) | E |
| 4 | Path features J200 | `G/kernel_sens/J200/latent_avian_paths` | 124 × `Z_disp_<yr>.npz` (~22.4 MB; keys Z_raw, Z_disp, labels, land_mask, build_id) + `path_feature_meta.json` sha a5bffb88; build_id **fae6a3f3** | 2026-11-05 21:49-22:00 (local CPU) | `K/code_bc/src/processing/generate_all_path_features.py --land-share-floor 1e-4`, overlay `G/kernel_sens/J200/overlay.json` (sha a45a6d4c, made by `kernel_sens.py overlay`); driver `...\7655df36-...\scratchpad\kernel\k_inputs_local.sh` | `K/logs/J200.log`; `SANDBOX_INVASION_SHAPE_2026-11.md:4915-4927,5100-5107` | E |
| 5 | numpyro_input J200 | `G/kernel_sens/J200/numpyro_input` | `metadata.pkl` 16,847,312 B sha **616aa6ea**; `Z_gathered_ba1d….dat` 137,070,592 B sha **b8e70938**; `Z_disp_gathered_ba1d….dat` 1,644,847,104 B sha **07a097e8**; ingest_id ba1d27a9 | 2026-11-05 22:02 | `K/snap_d0e59ff` (`src.data.combine.model_inputs`; GRID54_LADDER snapshot, BASE_COMMIT d9bd47a + `grid54_patch.diff`); check `field54_stage_f.py check` → `CHECK.json` (sha 89745c75) vs J330 production input | `K/logs/J200.log`; `G/kernel_sens/J200/CHECK.json` | E |
| 6 | established1951 | `G/kernel_sens/J200/established1951` | `metadata.pkl` sha **0090967d** (= INPUT_CONTRACT new sha); .dat = symlinks to #5 | 2026-11-05 22:02:58 | `research/exp/invasion_sandbox/kernel_sens.py derive` (main repo; committed only at dee075d 2026-11-08) | `INPUT_CONTRACT.json` there; diff: only `inv_timestep` 38→49 (`meta_diff.out`) | E |
| 7 | `_vl` (V&L-faithful juvenile kernel) | `G/inputs_shape/J200_established1951_vl` | `metadata.pkl` 17,207,746 B sha **eb29a8e3**; `Z_disp_gathered_vl24.937.dat` 1,644,847,104 B sha **818ae49f**; `KERNEL_DISCRETISATION.json` sha e88238f6; Z_gathered + contract = symlinks | 2026-11-07 21:07 | `SP\build_vl.py` (sha 4dfbf344; `build_shape.py` edited by `SP\vl_patch.py`), run by `SP\vl_build_j200_j110.sh` with `TAKEOVER_CODE_ROOT` = worktree (build_kernels + **worktree** `integrate_paths`) | `KERNEL_DISCRETISATION.json`; `SANDBOX_INVASION_SHAPE_2026-11.md:8399-8414,8755-8761` (b = 24.94 / 13.72 km); exact command only in scratchpad | E (content); record partial |
| 8 | `_nativeMX` | `G/inputs_shape/J200_established1951_vl_nativeMX` | `metadata.pkl` 17,193,706 B sha **063b9e57**; all other files symlinks to #7 / #5 / #6; companion `G/sandbox/ROUTEMARG/range_absence_halo400_nativeMX.npz` (8,174 B) | 2026-11-08 08:50 | `research/exp/invasion_sandbox/native_hull_build.py build --cutoff 1970` (main repo, sha 92fe97f3; committed dee075d 15:31, after the build) via `SP\hull_build.sh`; TACC copy via `SP\mx_tacc.sh` | `native_hull_variant` key in metadata; `SANDBOX_INVASION_SHAPE_2026-11.md:9938-9946`; **rebuilt bit-identically** (`SP\desk_prov\hull_rebuild.out`) | E |

## (b) Parameters

| Parameter | Value | Where recorded | Status |
|---|---|---|---|
| Kernel / contract | uncentred Ružička, 64-D Nyström summer block; `Z·Z' ≈ Ružička`; truncation `none` | `metadata.pkl` `z_kernel_contract`; cube_meta | E |
| Species (production teachers) | summer BBS 665 (369 + 296 sealed), winter CBC 586 (333 + 253 sealed); House Finch excluded. Landmark file sizes check out: 16,000 × 665 × 4 B + 128 = 42,560,128 B; 16,000 × 586 × 4 B + 128 = 37,504,128 B | `FIELD54_OBS_TEACHERS_2026-10-30.md:58-66,118-125,147-148` | E |
| Seasons | 128-D cube = summer teacher (0-63) + winter teacher (64-127); **winter half dropped** in s64 | cube_meta `seasonal_selection` | E |
| DESK training | native 54 km "desk_interp" recipe; production points = all 87,524 summer rows 1966-2025; presence head 15,693 early lists / 586 species; 500 fixed epochs, final weights, seed 1, latent 64 + 64 private, conv kernel 3; output EMA half-life 10.90 yr; **states v1** (interim; a v2 coastal-climate rebuild exists on TACC) | `FIELD54_DESK_2026-10-30.md:19-50,84-123` | E |
| Field (z_assim field_d) | D = desk_interp_s1_ema; rank 16, 128 centres, 200 km, 10-yr knots, lam_lap .5, lam_u 1e-4, lam_A/lam_C 1e-2, lam_curv 1, 2000 steps | cube_meta `assim_arm`; `FIELD54_FIELD_2026-10-30.md:101-110` | E |
| Grid / mask | 67 × 112, 54 km, 4,318 land cells, ESRI:102003; mask `grid54/land_mask/ocean_mask_54km.tif` sha 8f089228 | `path_feature_meta.json` | E |
| Dispersal (path features, J200) | juvenile mdd 200 km shape .468; adult mdd 100 km shape .859; splits [0, 155.36, 482.74, 1e9] km; 4 directions × 3 bands = K 12; 20 steps; `land_conditioned_neighborhood`; `cell_integrated` 31 subsamples; realized discrete MDD 202.22 km; mass .99999845 | `path_feature_meta.json`; `K/logs/J200.log` | E |
| FIELD54 land-share floor | 1e-4 on land SHARE; 175 cell-band pairs / 163 cells, by band [0,0,35,0,0,80,0,0,29,0,0,31] | `CHECK.json`, `path_feature_meta.json` | E |
| CHECK vs J330 production input | 41 metadata keys identical; differ: `dispersal_spec` + 2 juvenile stacks; Z_gathered identical; Z_disp differs at 51,812 / 51,816 pairs (expected: different kernel) | `G/kernel_sens/J200/CHECK.json` | E |
| `_vl` kernel | 2-D density `exp(-(r/b)^a)/r`, a .468, **b 24.937** (1-D V&L mean = 200.0 km; V&L's own b 41.056 → 329.3 km); `1/r` regularised below 54/62 km; same raised-cosine wedges and logistic band edges | `KERNEL_DISCRETISATION.json`; `repro.out` [2] | E |
| `_vl` vs parent master | own-cell mass .210 vs .092; 2-D grid mean 202.7 vs 202.2 km; P(r > 155) .368 vs .451; P(r > 483) .109 vs .089 | `repro.out` [2] | E |
| `_nativeMX` edit | hull = union(convex hull of BBS presences ≤ 1970 in the West, 128 cells; convex hull of 1,131 pre-1940 Mexican specimen points); 299 cells seeded at margin .5 (seeded 1,110 → 1,409); 228 pre-1940 Mexican pseudo-zero rows dropped (obs 192,123 → 191,895); halo 400 km; absences 77,790 → 74,306 | `native_hull_variant`; `hull_rebuild.out` | E |
| Invasion | inv_timestep 49 (1951); 4 candidate sites (27,89), (27,88), (26,89), (28,88) | metadata; J200.log | E |

## (c) Reproducibility verdict per stage

| Stage | Verdict | Basis |
|---|---|---|
| 0-1 DESK teachers and training | **Not reproducible from this repo.** Code is the frozen `coord_code` on TACC only. Its source commit is not in this git history (snapshot base `d9bd47a` is not a valid object here either). Identity is pinned by checkpoint / basis sha. A retrain would not be bit-identical (GPU nondeterminism, `FIELD54_DESK_2026-10-30.md:132-134`). | U (code), L (non-determinism) |
| 2-3 Field + s64 | Code in main repo (`z_assim.py`, `field54_s64_cube.py`, byte-identical to code_bc). Inputs and outputs are TACC-only except the local s64 copy. Not re-executed. | L |
| 4 Path features | **Main repo (dee075d) can; worktree (0f73b31) cannot.** Main `src/` is byte-identical (LF) to code_bc except `climate_regroup.py`. Worktree `generate_all_path_features.py` / `build_path_features.py` have no `land_share_floor` and do not pass `kernel_discretization`. They would build point-sampled path kernels without the floor, and worktree `model_inputs.py:289` would then refuse the build. Not re-executed. | E (code diff); L (rebuild) |
| 5 Ingest | Worktree `model_inputs.py`, `dispersal_spec.py`, `build_kernels.py` are byte-identical to snap_d0e59ff, and so are its other direct imports. Needs the raw HOUFIN_DATA inputs. Not re-executed. Z_gathered = path Z_raw = s64 cube exactly; numpyro Z_disp = path npz Z_disp exactly (3 years checked). | E (code, arrays); L (rerun) |
| 6 established1951 | Reproduced by construction: metadata sha matches INPUT_CONTRACT. | E |
| 7 `_vl` | **Reproduced, but only with the scratchpad script.** Kernel stack rebuilt with `build_vl.py`'s functions + worktree `build_kernels`: FFT complex64 **bit-identical**. Z_disp rebuilt with worktree `integrate_paths` (no floor): max \|diff\| 3.0e-8 (years 1902, 1951, 2025). No V&L-faithful kernel in worktree or main code: `radial_kernel_grid` is the 2-D area density `exp(-(r/s)^a)`. | E |
| 8 `_nativeMX` | **Reproduced bit-identically** from committed `native_hull_build.py`. All 50 metadata keys and the absence npz are equal. | E |

## (d) Inconsistencies and open questions

**Inconsistencies (all E unless marked):**
1. **Stale `path_features` in the LA1 overlay.** It points at `cube_s64_grid54` / `latent_avian_paths_s64_grid54` (the old
   27 km-coarsened DESK ensemble). The runtime ignores it: input_dir and raw_z_dir are prebuilt. But
   `src/vis/visualize_advi_model.py:140` and `visualize_hmc_model.py:220` read `path_features.output_dir`, so those
   figures would show the **wrong Z** [L]. The kernel_sens overlay had the right source
   (`.../field54/stageF/encoder/cube_field54prod_ema_s64`).
2. **`_grid54_comment` is stale.** It says "Inputs built by coarsen_model_grid.py --factor 2", but FIELD54 is native
   54 km. `community_bridge.ruzicka_dir` is also stale (`processed/encoder/esk/spacetime`); only
   `visualize_community_similarity.py` reads it.
3. **`kernel_contract._comment` says "first latent_dim (24)".** In fact latent_dim = 64, readout.rank = 64, and the
   metadata says truncation `none`.
4. **`_vl` dropped the FIELD54 land-share floor.** `build_vl.py` called the worktree `integrate_paths`, which has no
   floor. On the `_vl` stack the floor would act on 172 cell-band pairs in 164 cells (by band
   [0,0,33,0,0,80,0,0,29,0,0,30]). At those pairs the stored values differ from the floored rule by up to .24-.31.
   Elsewhere they are identical. The parent J200 Z_disp is floored, so LA1 mixes a floored lineage with an unfloored
   final Z_disp.
5. **`_vl` / `_nativeMX` metadata misdescribe the kernel.** `dispersal_spec` still reads juvenile mdd 200 / shape .468
   (the old area-density kernel), and the embedded `path_feature_meta` still carries the old labels, realized MDD
   202.22 and a `land_share_floor` record. So the guard `require_inputs_match_config` (`src/model/data_loading.py:425`;
   `dispersal_spec.py:184`) **passes while the kernel differs**: the silent-mismatch case its own docstring warns about.
   Only `KERNEL_DISCRETISATION.json` records the truth.
6. **Cosmetic changes in `_vl`.** `juvenile_edge_correction_stack` went float32 → float64, and the labels changed
   `483-1000000000` → `483-99999`.
7. **"DESK Z" is DESK EMA + a field correction.** The field correction is a regional space-time term fitted to all
   1966-2025 summer BBS rows, not raw encoder output. Per CONSIDERATIONS M6/M17, it predicts shared regional change
   without identifying cause.
8. **Build code was committed after use.** `kernel_sens.py` and `native_hull_build.py` were committed after use
   (dee075d), and `build_vl.py` exists only in a Temp scratchpad (the `_vl` step is recorded in prose, not by script).

**Open questions:**
- [U] The git commit / source of the frozen DESK `coord_code`, and whether any local copy exists. None was found under
  `C:\Dev`, `E:\Datasets\houfin` or the WSL `/work` mirror.
- [U] Should LA1 move from the interim v1 states to `field54prod_v2_ema`? That input exists on TACC; the coastal climate
  fix changes 219 cells.
- [U] Should the `_vl` Z_disp be rebuilt with the land-share floor, as the parent lineage is?
- [U] Adult kernel. It is unchanged: the 2-D area density, mdd 100, shape .859. The records do not say whether the same
  1-D → 2-D translation issue found for the juvenile kernel applies to it.
- [L] TACC copies. The TACC `inputs_shape/J200_established1951_vl_nativeMX` (used by the HMC slot `SP\CURRENT_A.sh`) is
  stated md5-identical to local (`SANDBOX_INVASION_SHAPE_2026-11.md:9946`). It was not re-verified here (no TACC
  access).
