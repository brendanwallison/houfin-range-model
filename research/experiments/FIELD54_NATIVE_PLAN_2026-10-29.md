# Native 54 km field_d: production build + masked twin (plan, pre-registered 2026-10-29)

User (2026-10-29): use field_d_interp; train from scratch where needed; production has ALL species except House Finch
(sealed included), all locations, all years; a from-scratch build is native 54 km, and may get a non-production twin
under the current experimental protocols; it must be done right, covariate aggregation included.
Pipeline map: agent report 2026-10-29 (stages 0-5 below cite its file:line findings). Records: ASSIM_FIELD_2026-10-27.md
(field_d_interp: withheld regional .664 sedentary common / .547 all species), GRID54_INPUTS_2026-10-29.md.
Considerations: G6 (what production ships: raw z), G8 (history length), M7 (scales).
Interim, separate: the existing backcast40 field_d_interp cube, block-averaged to 54 km, is being put through the
ladder (GRID54_INPUTS_FDINTERP_2026-10-29.md) -- an averaged Z, not this native build.

## Two builds, one code path
| | production (field54_prod) | masked twin (field54_eval) |
|---|---|---|
| species | all except House Finch (sealed included) | non-sealed, House Finch out (as backcast40) |
| summer rows | all BBS 1966-2025, all places | backcast40 masks mapped to 54 km (below); pre-1996 training-place rows added (interp) |
| winter | all CBC lists | backcast40 winter mask mapped to 54 km |
| teachers | ESK at 54 km from all rows | ESK at 54 km from training rows (as z_repair_teacher) |
| DESK | all rows, 500 epochs, no epoch selection on outcomes | desk_interp recipe at 54 km |
| field | z_assim field_d on all rows | z_assim field_d, --summer interp |
| scored? | no (nothing withheld) | yes: withheld places, k2, regional/all/local, sedentary common first |
Mask mapping (twin): a 54 km cell is a holdout cell if ANY of its 27 km children is holdout; buffer likewise; a
withheld-panel evaluation cell keeps its 54 km parent out of every training source. Conservative: nothing that
touched a withheld place trains.

## Stage A -- covariates, natively from source (TACC CPU; outputs in their own 54 km data root)
Every family is rebuilt from the native source with the 54 km reference grid (ref_grid_54km.tif, 67 x 112, same origin
as 27 km, each cell exactly 2 x 2 children):
- Climate (240 ch): EXACT, no climr rerun. The 27 km run kept per-sub-point output
  ($SCRATCH/houfin/data/climate/_chunks/out_*/climate_points.csv; 5 x 5 points per 27 km cell, fine-land filtered).
  A 54 km cell's 10 x 10 sub-point lattice is the union of its children's (same fine transform), so q10/q50/q90 are
  recomputed over the union with climatena.quantile_aggregate (parent = (row // 2, col // 2)). Check: the 27 km
  recompute from the same points reproduces climate_q*.csv exactly.
- Elevation (3): rerun elevation.py with fine_factor 30 (= 2 x 15; same lattice); recompute p10/p50/p90.
- BUI (7): rerun bui.py with fine_factor 54 (= 2 x 27); recompute quantiles, bui_avail mean, non-CONUS fill.
- HYDE 3.5 (3): density = area-weighted mean of the 5' source, counts = sums; log1p only at load (never average logs).
- LUH-3 (33): area fractions = area-weighted mean. Management rates (fertl, irrig, fharv) and secma / secmb are per
  unit of their parent land use: aggregate as the parent-area-weighted mean. The 27 km build used a plain average
  (regrid.py:238-248), so this is a correction; it is applied at 54 km and flagged as a 27 km inconsistency (the
  twin's comparison with the 27 km cube carries it).
- Soil (16): area-weighted mean of SoilGrids 5 km, then nearest-land hole fill on the 54 km mask.
Then build_states.py (linear blend between snapshots, flat ends, tau = 2 backward EMA; 302 channels, 1882-2025) with
the 54 km data overlay. Checks: exactly-aggregable channels (LUH fractions, HYDE counts, soil) equal the area-weighted
mean / sum of the 27 km children to float tolerance (where the 27 km rule was the same); quantile channels lie within
the children's range; channel schema identical to the 27 km states.

## Stage B -- observations at 54 km
BBS points: route -> 54 km cell with the 54 km land mask (coastal snap rule of coarsen_model_grid), cell-year =
mean over routes, then log1p (bbs_community_points.build_bbs_rows). CBC: circle row/col from the 54 km ref raster.
Species lists from the cache meta (production: + sealed, - House Finch). Checks: route and circle counts conserved;
House Finch absent (assert).

## Stage C -- teachers (ESK, 54 km points)
z_repair_teacher recipe (64 dims, 16k diverse landmarks, window-mean augmentation, seed 20261026), summer and winter,
on 54 km points. Native: the community vectors are 54 km route means, so the basis is rebuilt, not projected.
tile_cells (landmark strata) halved to keep the km scale. z_repair_aux projections, cbc_ruzicka, winter operator.

## Stage D -- DESK at 54 km
Recipe: desk_final_clean_v2 overlay as in z_repair_control / z_coord_run (interp arm). Settings counted in cells are
rescaled to keep their km extent: spatial_conv.kernel 5 -> 3 (135 -> 162 km), tile_cells 6 -> 3, smooth_half_width
2 -> 1, smooth_sigma_* halved, cube gap-fill radius_px 25 -> 13. 500 fixed epochs, no selection on withheld outcomes.
Export raw; z_ema_cube at the trained half-life. GPU: TACC gpu-a100-small (27 km took 10.4 h locally / ~72 min A100;
54 km has ~4x fewer cells).

## Stage E -- field fit at 54 km (z_assim field_d)
Remove the year >= 1996 assert for production (z_assim.py:67); TASK / root / control cube become arguments (no
hard-coded backcast40). Basis unchanged in km (128 RBF at 200 km, 10-yr cubic B-spline, rank 16). GMRF: lam_lap is per
4-neighbour pair, so at 54 km it is ~2x smoother per km; re-run the inner grid on the twin's inner backcast (training
data only) for lam_lap and lam_u rather than guess a rescale.
D input: field_d_interp used the RAW control (a DESK that never saw pre-1996 summer). Our DESK is supervised 1966-2025,
for which EMA is its trained quantity (ASSIM_FIELD: desk_interp EMA beats raw by +.08). The twin fits the field on
BOTH D = raw and D = EMA and scores both; production uses the winner. The field output itself is not EMA'd (it is
fitted to the teacher coordinates directly).
Output: cube with a written cube_meta.json (z_assim.export writes none today).

## Stage F -- age-model inputs and use
Summer 64 dims (as s64) -> Z_disp (generate_all_path_features, cell-integrated kernels) -> model_inputs (obsroute) ->
the ladder / injection arms. Compare with the averaged-Z interim build.

## Twin decision rules (pre-registered; withheld places, k2, regional, paired 6x6-block bootstrap)
1. Native 54 km vs averaged 27 km: score field54_eval and block-averaged field_d_interp on the SAME 54 km panel.
   "Native at least as good" if the difference >= -0.02 (sedentary common and all species).
2. D = EMA vs raw: the larger regional score wins; ties (< .01) go to EMA (the trained quantity).
3. Sanity: field54_eval beats the 54 km control DESK (raw / EMA) at withheld places, as at 27 km.
Production is built with the winning choices; it is not scored (nothing withheld).

## Order and compute
A (TACC CPU) and B (local CPU) now; C after B (CPU, minutes); D after A-C (TACC A100, twin then production); E (CPU,
~0.5 h per fit + inner grid); F as for the existing grid54 inputs.

## Amendment (2026-10-29, before any native stage ran): Z_disp land-share floor
GRID54_INPUTS_FDINTERP found integrate_paths divides rounding noise by ~0 where a dispersal band's land share is
< 1e-4 (181 / 51,816 cell-band pairs at 54 km, 1,031 / 206,508 at 27 km; up to 1.45x the largest Z averaged; present
in every existing input). The native build applies a land-share floor (bands below it take the cell's own Z),
as a default-off switch, and reports the affected pairs; existing inputs are left unchanged.

## Amendment (2026-10-30, before Stage D): smooth_half_width stays 2
Stage D listed "smooth_half_width 2 -> 1" as a cell-counted setting. It is not: target.smooth_half_width is the
point-denoising half-width in YEARS and must equal the teachers' esk.spacetime.augment_half_width
(config/esk_desk_config.json _smooth_half_width_coupling); halving it would invalidate the 54 km teachers' basis.
Only CELL-counted settings are rescaled (spatial_conv.kernel 5 -> 3, tile_cells 6 -> 3, smooth_sigma_cells halved,
cube gap-fill radius_px 25 -> 13). Flagged by the Stage B/C build (FIELD54_OBS_TEACHERS_2026-10-30.md). Also from that
build: landmark strata are extent-relative (no tile change needed), and plan rule 1 scores native vs averaged on the
54 km panel (219 withheld cells).
