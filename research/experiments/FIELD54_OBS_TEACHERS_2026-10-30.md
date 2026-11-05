# FIELD54 Stages B and C: native 54 km observations and teachers (production + masked twin)

Plan: FIELD54_NATIVE_PLAN_2026-10-29.md (Stages B, C). Stage A: FIELD54_COVARIATES_2026-10-29.md.
Considerations cited: G6 (production ships raw z), M3 (observers), M7 (scales), M12 (survey composition).
Full machine reports: `field54_obs_teachers_20261030/STAGE_B_DONE.json`, `STAGE_C_CHECKS.json`; job scripts
copied alongside (`envbc.sh`, `bc_obs.slurm`, `bc_teachers.slurm`).

## What was run (TACC ls6, development queue, CPU)

| job | what | result |
|---|---|---|
| 3542362 (f54bc_obs) | Stage B `research/exp/field54_obs.py` (27 km data env) | Stage B COMPLETED, all asserts passed; the two packaging steps after it failed (env: `HOUFIN_DATA54` not exported, so `${HOUFIN_DATA54}` was not expanded in field54_data.json). Nothing was written by the failed steps. |
| 3542363 | first teacher job | DependencyNeverSatisfied; cancelled by me |
| 3542420 (f54bc_teach) | packaging (`z_repair_data.py` twin + `--production`), 4 ESK teachers, 2 x `z_repair_aux.py`, `z_assim_interp_prep.py`, `z_coord_run.write_points`, `field54_teacher_checks.py` | COMPLETED 18:52 min, exit 0 |

Code snapshot: `/work/07980/bwa386/ls6/houfin/research/field54/code_bc` (= the Stage A tree + the local working tree's
`src/`, `research/exp/`, `research/lib/`; CR stripped; `SNAPSHOT_MANIFEST.txt` holds sha256 of every file).
27 km inputs pushed to `.../research/field54/inputs27` (SHA256SUMS there). Changed/new files (sha256 of LF content, first
16 hex): field54_obs.py 8ba937de3685333f, field54_teacher_checks.py c0c3d63c58ac3d82, z_repair_data.py adef00ba1369116b,
z_repair_aux.py f74eb82ffac400b7, build_cbc_presence.py b5d67335fd09ae0e, bbs_community_points.py 1bbd42f9c41cece5,
coarsen_model_grid.py c0196e60d749e6c2, tests/test_field54_obs.py 3b3a6fa6640aa3c5, tests/test_z_repair_data.py
7d6d0ab3c08bad7f. Not committed (several of these files were already untracked/uncommitted in the working tree).

### Code changes (all default-off or parameterised; existing behaviour unchanged)
- `src/processing/coarsen_model_grid.py`: new `parent_cells()` (the coastal snap rule, pulled out of `coarsen_bbs`, which now
  calls it). Same rule, one copy.
- `bbs_community_points.build_bbs_rows(..., route_cell_map=None)`: optional route-table remap; when set, meta records
  routes / route-years. Default path verified unchanged: with an identity map the 27 km build reproduced the existing
  `bbs_points_all369` exactly (keys, X, weights all identical, 114,172 rows).
- `z_repair_data.py`: `--production` (nothing withheld, sealed allowed, House Finch forbidden, task "production",
  summer_start 1966, winter full only), `--tasks`, `--winters` (defaults = old behaviour), holdout/buffer optional only in
  production mode.
- `z_repair_aux.py`: `--task production`; `--block-cells` (default 6) passed to `build_cbc_presence.winter_operator`, whose
  lambda-validation blocks were hard-coded 6 x 6 cells (162 km at 27 km). Used 3 at 54 km (162 km).
- New: `research/exp/field54_obs.py` (Stage B), `research/exp/field54_teacher_checks.py` (Stage C checks).
- Tests: `tests/test_field54_obs.py` (snap rule, any-child block, mask mapping incl. snap destinations),
  production-mask test in `tests/test_z_repair_data.py`. 33 pass locally, 8 re-run in the TACC job.
- Sealed read logged before use: `research/sealed_reads.csv` 2026-10-30T08:57:01 (FIELD54 production build).

## Outputs

Stage B (`/work/07980/bwa386/ls6/houfin/processed/field54/obs`): `bbs_points_eval54/`, `bbs_points_prod54/` (points in the
standard X_points / point_index / point_weights / points_meta layout), `cbc_eval54.npz`, `cbc_prod54.npz`,
`holdout_cells54.npy`, `buffer_cells54.npy`, `base_eval/`, `base_prod/`, `route_cells_27_to_54.csv`, species CSVs, DONE.json.

Stage C, z_repair layout (what z_repair_control / z_coord_run read), roots under
`/work/07980/bwa386/ls6/houfin/research/field54/runs/`:
- twin `field54_eval/`: `data/backcast40/{summer_train_points, summer_masks.npz, winter_full/...}`,
  `data/backcast40_interp/summer_train_points` (training rows + pre-1996 training-place rows),
  `teachers/backcast40/{summer,winter_full}/spacetime`, `aux/backcast40/full/{cbc_ruzicka.npz, winter_coords.npz,
  cbc_presence_early_w.npz, *_training_coords.npy}`, `aux/backcast40/interp/summer_extra.npz`.
- production `field54_prod/`: `data/production/...`, `teachers/production/{summer,winter_full}/spacetime`,
  `aux/production/full/...`.
- `FIELD54_TEACHER_CHECKS.json`.

## Stage B checks

Species (rule: twin = cache community + dev, never sealed, never House Finch, = the 27 km all-species build;
production = twin + the cache's sealed species):
- BBS: twin 369, production 665 (+296 sealed); crosswalk matched 665/665. House Finch: not in any list, AOU 5190 not in
  the production crosswalk (asserted).
- CBC: twin 333, production 586 (333 + 253 sealed). The CBC exports drop species with taxonomy splits (36 non-sealed,
  43 sealed), as at 27 km. The production CBC matrix is the existing non-sealed export plus the existing sealed export
  (`FINAL_HOLDOUT_20261025/sealed/cbc_circles.npz`), joined column-wise after asserting every circle-year array
  (circles, years, row/col, lat/lon, effort fields) is identical between them.
- Sealed species in production: listed in STAGE_B_DONE.json (`checks.species.sealed_species_in_production`).

BBS routes (route -> its 27 km cell -> 54 km parent, snapped to nearest 54 km land if the parent is water):
- routes 5,383 -> 5,383; route-years 122,955 -> 122,955 (twin and production). Conserved.
- 104 routes (79 27 km cells) snapped; mean 54.5 km, max 108 km (2 cells).
- cell-years 114,172 -> 87,524; cells 3,902 -> 2,555; 1.40 routes per cell-year at 54 km (1.08 at 27 km). No all-zero
  rows dropped in either build.
- Count conservation (twin): for every species and 54 km cell-year, mean x routes equals the sum over its 27 km children;
  max relative difference 4.7e-7 (float32 rounding); total counts 66,337,702.9 vs 66,337,702.7.
- First-year (new observer) downweighted rows: 20.1% at 54 km vs 16.4% at 27 km (mean weight .919 vs .922). Mechanism
  (established by construction): a 54 km cell-year is downweighted if any of its routes had a first-year observer, and it
  pools more routes.

CBC circles (row/col from the 54 km raster, same formula as cbc_prepare):
- 103,188 circle-years / 4,740 circles; raster row/col equals the parent of the 27 km cell for all 103,188.
- Valid lists 102,287 -> 102,287; valid circles 4,703 -> 4,703. Conserved. 3,337 lists (141 circles) snapped, max 76 km.
- 901 lists invalid at 27 km (27 km cell is water) stay out at 54 km, although 248 of them fall on 54 km land
  (deliberate: keeps the observation set identical to 27 km; see deviations).

Twin masks (54 km cell holdout if ANY 27 km child is holdout; buffer likewise; holdout wins where both; plus every 54 km
cell that receives a snapped route or circle from a 27 km holdout / buffer cell):
| | 27 km | 54 km |
|---|---|---|
| holdout cells | 532 | 351 (= 1,404 27 km-cell equivalents) |
| buffer cells | 579 | 379 |
| cells added by snap destinations | | 3 holdout, 3 buffer |
| BBS route-years in holdout / buffer | 17,515 / 19,000 | 17,573 / 20,243 |
| BBS cell-years in holdout / buffer | 16,264 / 17,371 | 12,308 / 13,637 |
| CBC lists in holdout / buffer cells | 6,462 / 6,853 | 14,072 / 14,434 |
- All 284 withheld-panel cells of the 27 km twin have their 54 km destination in holdout; every route-year and list from
  a 27 km holdout cell is in a 54 km holdout cell; every one from a 27 km buffer cell is withheld (asserted).
- Why the winter side grows much more than the summer side (likely): the any-child rule makes the withheld area 2.6x the
  27 km area, mostly by absorbing water-free neighbour children; BBS routes in those children were already mostly in the
  27 km buffer (route-years withheld in total rise only 36,515 -> 37,816), while CBC circles there were often training
  circles, so withheld lists roughly double.

## Stage C checks

Packages (z_repair_data at 54 km):
| | twin (backcast40) | 27 km twin (for reference) | production |
|---|---|---|---|
| summer rows (all) | 87,524 | 114,172 | 87,524 |
| summer training rows | 18,198 (>= 1996, no row at holdout/buffer) | 24,600 | 87,524 (all, 1966-2025) |
| eval halves modern / historical | 11,889 / 8,334 each | 15,668 / 10,399 | none |
| panel known / withheld cells | 1,028 / 219 | 1,262 / 284 | none |
| winter training lists | 72,238 | 87,088 | 100,181 (= all valid lists 1902-2024) |
| winter evaluation lists | 13,795 | 6,353 | 0 |
| training circles | 3,340 | 4,046 | 4,675 |
| winter count cell-years | 44,540 | 70,763 | 58,983 |
| early 1902-1939 training lists | 3,182 | 4,191 | 5,135 |
| interp rows added (pre-1996, training places) | 25,048 (total interp points 43,246) | 31,693 | n/a |

Teachers (recipe of z_repair_teacher unchanged: 64 dims, 16,000 diverse landmarks, window-mean augmentation half-width 2 yr,
seed 20261026; rebuilt from the 54 km points, not projected):
| teacher | rows | species | median z^2 annual / window | uncentered combined rmse |
|---|---|---|---|---|
| twin summer | 18,198 (+14,962 window) | 369 | .581 / .608 | .075 |
| twin winter | 44,540 (+44,361) | 333 | .652 / .689 | .052 |
| prod summer | 87,524 (+86,399) | 665 | .540 / .588 | .085 |
| prod winter | 58,983 (+58,762) | 586 | .606 / .652 | .061 |
| 27 km twin summer (786) | 24,600 (+19,772) | 369 | .580 / .605 | .073 |
| 27 km twin winter (787) | 70,763 | 333 | .656 / .692 | .048 |
The 54 km twin teachers match the 27 km ones closely (established: same numbers to ~.005). Production summer is a little
less representable (.540; likely: 665 species and 4.8x more rows at the same 64 dims / 16k landmarks, so each landmark
covers more of the manifold).

Auxiliaries (z_repair_aux, operator blocks 3 cells = 162 km):
| | twin | 27 km twin (788) | production |
|---|---|---|---|
| cross-season calibration cell-years | 7,572 | 4,159 | 34,564 |
| a / b / mse | .0052 / .777 / .0113 | .0085 / .825 / .0106 | .0080 / .833 / .0094 |
| winter operator lists, lambda | 61,295, 1e-6 | (73,302), 1e-6 | 84,223, 1e-6 |
| operator nll (uniform) | 4.065 (5.808) | 4.0925 heldout | 4.640 (6.373) |
| early presence lists | 10,718 | 13,379 | 15,693 |
More calibration cell-years at 54 km (established mechanism: a 54 km cell-year is more often surveyed by both BBS and
CBC). The production operator nll is higher because it predicts across 586 species rather than 333 (uniform 6.37 vs 5.81);
not a quality comparison.

Leak checks passed (asserted in field54_teacher_checks.py): twin summer teacher rows = training mask, all >= 1996, none
at 54 km holdout/buffer cells; twin winter count points none at holdout/buffer cells; training and evaluation circles
disjoint; interp rows all < 1996 and outside holdout/buffer (and z_coord_run asserted none at a withheld panel cell);
no sealed species or House Finch in any twin list. Production: every valid summer row and winter list trains, nothing
evaluated, all 296 / 253 sealed species present, no House Finch.

## Deviations from the plan, and why
1. **Landmark strata "tile_cells halved": not applied, because it does not exist.** The ESK landmark strata use
   `spatial_bins = 8` equal tiles over the occupied extent (`esk_kernel.spatial_tiles`), not a cell count, so they already
   keep their km size at 54 km (established, by reading the code). `tile_cells` is DESK's masking granularity, a Stage D
   setting (plan Stage D: 6 -> 3).
2. **Winter operator validation blocks 6 -> 3 cells** (keeps 162 km). Plan did not mention it; it is the same
   "settings counted in cells" rule the plan applies in Stage D. Only affects which lambda is chosen (1e-6 at both grids).
3. **CBC observation set identical to 27 km.** Lists whose 27 km cell is water stay out (248 of them fall on 54 km land and
   could have been added). Done so the route/circle counts are conserved and the twin compares like with like. Lists valid
   at 27 km whose 54 km parent is water are snapped (the BBS rule), not dropped.
4. **Holdout wins over buffer** where a 54 km cell has both kinds of children (the plan's "buffer likewise" is otherwise
   ambiguous; z_repair_data would skip a cell that is both, losing evaluation cells).
5. **Snap destinations inherit the mask.** A route/circle in a 27 km holdout cell whose parent is water is snapped to a land
   cell; that cell is marked holdout (3 cells; 3 likewise for buffer). Without this, withheld-place data could train.
6. Only the backcast40 task and the full winter were packaged for the twin (`--tasks backcast40 --winters full`), as used by
   field_d_interp / desk_interp; backcast20 and sparse/no-target winters were not built.
7. Production completeness benchmarks use all 586 CBC species (sealed included); the manifest string
   "nonsealed species only" in `winter.full.benchmark_fit` is a hard-coded label from z_repair_data and is wrong for
   production.

## Notes on interpretation
- A 54 km cell-year averages 1.40 routes vs 1.08 at 27 km. Where different routes of one cell are surveyed in different
  years, the cell-year mean mixes route turnover into apparent change (M12 survey composition; likely small, unmeasured).
- The twin's withheld set is not the 27 km one: 219 withheld panel cells at 54 km vs 284 at 27 km, and twice as many
  winter evaluation lists. Rule 1 of the plan (native vs averaged on the SAME 54 km panel) must use this 54 km panel for both.

## What Stage D (DESK at 54 km) still needs
- `z_repair_control.py` / `z_coord_run.py` accept only `--task backcast40|backcast20`; production needs `production` added
  (same layout). Their frozen snapshot copies `config/overlays/desk_final_clean_v2_s1.json` from the repo; a 54 km overlay
  is needed with the 54 km data config (DATA_CONFIG=field54_data.json, export HOUFIN_DATA54 / HOUFIN_PROCESSED54 -- the
  failure above) and the Stage A states (`processed/field54/states`, overlay field54_states.json).
- Rescaled DESK settings (plan): spatial_conv kernel 5 -> 3, tile_cells 6 -> 3, cube gap-fill radius_px 25 -> 13,
  smooth_sigma_cells / smooth_sigma_s (cells) halved. **Open question:** the plan's "smooth_half_width 2 -> 1" -- in
  esk_desk_config `target.smooth_half_width` is in YEARS and must equal `esk.spacetime.augment_half_width` (2, used by
  these teachers); halving it would break that coupling. Probably the plan meant a cell-counted width; check before Stage D.
- The DESK balance strata (`desk.balance.spatial_bins`) are also extent-relative bins (no change needed).
- Interp (desk_interp) training points are ready at `field54_eval/data/backcast40_interp/summer_train_points`.
