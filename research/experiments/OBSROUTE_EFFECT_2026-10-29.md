# BBS observer-route random effect + first-year effect (age model)

Date: 2026-10-29. Status: code, tests and inputs only. **No model fit was run.**

Code lives in the age-model working copy
`/home/breallis/houfin/work/houfin/research/GRID54_LADDER_20261029/code` (WSL). This repo's `src/model` is
stale for the age model. Commits there, on top of `ac521d8`:

| commit | content |
|---|---|
| `dd78b76` | `src/data/preprocess/bbs.py`: save the observer arrays to the npz |
| `ca81f61` | `src/model/age_priors.py`, `src/model/data_loading.py`, `src/data/combine/model_inputs.py`: the effect, default off |
| `7045db3` | `tests/test_observer_effect.py`, `tests/test_full_model_smoke.py` (synthetic observer arrays + smoke arm), `config/overlays/map_obsroute_effect.json` (template) |
| `d0e59ff` | `research/age_model/exp/obsroute_sigma.py` (sigma estimator), `research/age_model/exp/build_observer_ids.py` (side file for existing inputs) |

## Why

The likelihood treated every BBS route-year as an independent NB2 draw around the 27 km cell mean.
Persistent differences between routes and observers had nowhere to go except the habitat fields.
Observer turnover is about 30% of measured community change energy (E021; CONSIDERATIONS M3, M18;
E028/E028b). Sixty years from one route also counted as 60 independent pieces of evidence about that
cell's level. Standard BBS models include observer-route effects and a first-year effect: Sauer & Link
(2011), and bbsBayes2 `first_diff_hier_bbs_CV.stan`, which uses
`eta*first_year + sdobs*obs_raw[observer]` with `obs_raw` a `sum_to_zero_vector`, `eta ~ std_normal()`
and `sdobs ~ normal(0,0.3)`.

## Design

For real observations: `mu_i = density * pop_scalar * exp(eta * first_year_i + sigma_u * eps[obsroute_i])`.
Pseudo-zeros (id -1) are left exactly as before. The switch is
`population_model.observation.observer_effect`, and it is **off unless set** (absent key = off).

- **Grouping.** Effects are per observer × route, bbsBayes2's `obs_route`. Mexico has no ObsN, so each
  Mexico route gets one id and first_year is 0 there.
- **Centring: zero-sum `eps ~ ZeroSumNormal(1)`, non-centred.** The capacity level is anchored on a
  *median* in route counts (docs/PRIORS.md). With a zero-sum effect the geometric mean of the multipliers
  is exactly 1, and because the effects are symmetric the median is about 1 (test). So
  `density*pop_scalar` still means "a typical observer-route's count". A `-sigma^2/2` lognormal
  correction would preserve the arithmetic mean instead, which moves the median observer-route down by
  `exp(-sigma^2/2)` (-38% at sigma 0.97) and shifts the gauge. Zero-sum also removes the common-shift
  direction that would otherwise trade against the capacity level. (Established: algebra and test.)
- **sigma_u is FIXED (config) by default.** Under a joint mode (AutoDelta MAP, then float64 L-BFGS) a
  hierarchical scale is degenerate in both parameterisations:
  - Centred `u ~ N(0, sigma)`: the mode drives sigma to 0, because the density `prod N(u_g; 0, sigma)`
    at `u = 0` grows without bound as sigma shrinks.
  - Non-centred `u = sigma*eps`: the penalty is `eps^2/2 = u^2/(2 sigma^2)`. For a fixed fitted `u` it
    falls as sigma grows, so sigma inflates until its own prior stops it.

  The lineage has already hit the second case. `observation.route_effects` with a free `obs_route_sd`
  went to sd 7.5 in run_41, and map_run59a then pinned it at 0.7. That earlier route effect is still
  non-zero-sum. With sigma fixed, the eps prior is an ordinary ridge, so there is no degeneracy.
  `sigma_mode: "prior"` samples `sigma ~ HalfNormal(sigma_prior_scale)` and is meant for HMC only.
- **eta** is a fixed effect with prior `Normal(0, eta_prior_scale=1)`, matching bbsBayes2. Density then
  refers to an experienced observer. `pinned_sites` can pin `observer_first_year_eta` or
  `observer_route_sigma`.
- **Diagnostics.** `expected_obs` remains the mean the NB2 sees, which is what the residual diagnostics
  read. When the effect is on, `expected_obs_before_observer` is also stored.
- **Ordering.** The multiplier is applied after `route_effects` and before `year_effects`, the NB
  concentration terms and route-occupancy zero inflation. Route-occupancy zero inflation preserves the
  mean, so the multiplier passes through it unchanged.

## Ingest (27 km)

`bbs.py --out-name bbs_data_obsroute_for_python.npz --mexico-from-npz bbs_data_for_python.npz` writes
`${HOUFIN_DATA}/bbs_2026_release/bbs_data_obsroute_for_python.npz` (sha256 `2e8c0a7b…85ca9a`). The
existing file was not touched.

- The Mexico release is absent locally, so its 236 tier-1 rows are carried over from the old npz, with
  each Mexico cell standing in for a route (58 ids).
- Every existing key is **bit-identical** to `bbs_data_for_python.npz`. The snapshot build and the repo
  build are identical to each other.
- Joining on `[CountryNum, StateNum, Route, Year]`, **0 / 123,191** real observations failed to find an
  observer, and there were 0 duplicate observers per route-year. Any future unmatched row would get its
  own singleton id and first_year 0; this is in the code and tested.

| quantity | value |
|---|---|
| real obs / pseudo-zeros | 123,191 / 299,668 |
| routes | 4,947 (median 3 observer-routes per route) |
| observer-routes | 19,127 (58 Mexico) |
| obs per observer-route | min 1, q25 1, median 3, q75 8, q90 17, q99 34, max 55; mean 6.4 |
| observer-routes with 1 obs / >= 3 obs | 5,140 / 11,351 (the >= 3 hold 91.5% of obs) |
| first-year obs | 19,069 = 15.5% of real obs. This equals the number of US/CA observer-routes, as it must |

**Alignment through the ingest.** I ran `model_inputs`' own observation section on the new npz. It
reproduces `obs_rows`, `obs_cols`, `observed_results`, `obs_quality` and `obs_time_indices` of the
existing ingest (`numpyro_input_cbc1902_r64`) element by element: 192,123 obs, of which 68,932 are
pseudo-zeros. Every route maps to exactly one cell and every observer-route to exactly one route.
`build_observer_ids.py` independently verifies the same match and writes a side file for already-ingested
input directories.

**54 km.** The coarsening script carries extra per-observation arrays, so this npz can be coarsened as it
is. `model_inputs` re-densifies the ids after its year filter.

## sigma_u estimate (`obsroute_sigma.py`; JSON: `OBSROUTE_SIGMA_2026-10-29.json` here)

**Method.** I fit a marginal-likelihood NB2 GLMM to House Finch counts at US/Canada observer-routes with
at least 3 years:

    y ~ NB2(exp(a[ref] + eta*fy + u[g]), phi),   u ~ N(0, sigma)

- `a` is a fixed effect per reference cell.
- `u` is integrated out by adaptive Gauss-Hermite quadrature, so the scale is identified. A joint mode
  would not identify it.
- Only references that hold at least 2 observer-routes and a non-zero total are kept.
- With about 2 observations per fixed effect, the ML scale is biased low (Neyman-Scott; only within-
  reference contrasts survive). The simulation runs measured this bias as **-25% to -40%**. I corrected
  it by iterated simulation calibration (3 rounds × 10 refits). The 95% CI is a cell-cluster bootstrap
  (200 replicates) scaled by the calibration factor.

| reference | obs / obs-routes / cells | raw sigma (boot 95%) | **calibrated sigma** (95%) | eta (boot 95%) | NB phi |
|---|---|---|---|---|---|
| 27 km cell × year (primary) | 8,656 / 1,218 / 299 | 0.735 (0.650-0.812) | **0.97 (0.86-1.07)** | +0.014 (-0.07, +0.10) | 58 |
| cell × 5-yr block (sensitivity) | 17,962 / 3,614 / 1,212 | 0.547 (0.492-0.588) | **0.84 (0.75-0.90)** | -0.019 (-0.05, +0.02) | 4.7 |

**Comparison with bbsBayes2.** bbsBayes2's `sdobs ~ half-normal(0, 0.3)` has a median of 0.20 and a 97.5%
point of 0.59. Our estimate of about 0.9-1.0 is in its far tail, where P(sd > 0.97) ≈ 0.001.

- **Likely mechanism.** Our `u` absorbs the route-within-cell siting that bbsBayes2 gives a separate
  `ste` effect (`sdste ~ student_t(3,0,1)`), as well as the observer. House Finch is a patchy, town-
  associated species (cf. E072 route occupancy), so persistent route levels
  within a 27 km cell are large.
- **Why the two references differ.** The block-5 reference adds same-route observer turnover, and it
  gives a smaller sigma than the cell-year reference. This suggests the route-siting part dominates over
  the observer part. That is likely, not established; the two CIs do not overlap by much.
- **Template default.** `map_obsroute_effect.json` uses sigma_u = 0.97, the pre-specified primary.
  0.84 is the sensitivity value.
- **eta.** For House Finch it is about 0, and both CIs include 0. The USGS analysis cited in `bbs.py`
  reports first-year counts running low for most species, but not measurably for this one. This is
  established within the CI.

**Production implication (for the planner, not a decision).** At sigma_u ≈ 1, an observer-route's
persistent level is only loosely tied to its cell's mean. Count *levels* at single-route cells (80% of
the 3,959 cells with real counts have one route) will then inform habitat much less than now. Trends within an
observer-route and the pattern of occupancy carry the information instead. This is the intended effect,
but its size is large. It interacts with `route_effects` (do not enable both without a reason, since they
are nested) and with `conc_density_power` / route occupancy, which currently absorb part of the same
patchiness.

## Tests (snapshot; OMP capped, CPU)

`tests/test_observer_effect.py`: 13 tests, all pass.

- Toy ingest table: lengths, observer × route grouping, re-flagging an observer on a new route, Mexico
  one id per route, unmatched → singleton, first-year logic.
- Order alignment under shuffling; `dense_ids` keeps -1.
- Real-npz alignment: pseudo rows -1/-1/0, and one first year per US/CA observer-route.
- **Switch off: the full `build_model_2d` log density equals the pre-change commit `ac521d8` exactly**
  (float equality, on synthetic inputs).
- **sigma 0, eta 0: the log density equals off + log N(0; 0, 1)**, with no eps site and an identical
  `expected_obs`.
- The multiplier is exact on real obs and pseudo-zeros are unchanged.
- In fixed mode AutoDelta has no sigma coordinate; in prior mode it has one.
- The full-model gradient is finite.
- Missing arrays raise an error.
- Zero-sum keeps the median multiplier at 1 (n = 4001, within 0.05).
- **Float64 central finite differences match `jax.grad`** (rtol 1e-5) in both fixed and prior modes.

Existing tests also pass: `test_hmc_common`, `test_data_pure_cores`, `test_model_inputs_guards`, and
`test_full_model_smoke` (base, route arms, run59a pinned-route test, new `map_obsroute_effect` arm),
72 + 17 tests in total. I did not run the full 180-overlay smoke matrix.

## Not done

- No MAP, refine or HMC.
- `build_model_2d` has not been evaluated on the real 27 km inputs with the effect on. Only synthetic
  full-model inputs and an ingest replay were used.
- To use the effect on an existing input directory, run `build_observer_ids.py` first, or re-ingest
  from the new npz.
