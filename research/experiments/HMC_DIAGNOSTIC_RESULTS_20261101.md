# HMC geometry and performance diagnostics — live results

Full unconstrained nominal posterior, including transformation Jacobians and observer nuisance parameters. Geometry rows are matrix-free Krylov estimates with residuals, not certified full condition numbers or full negative-eigenvalue counts. A finite geometry result is not an HMC pass. All queued cases are shown.

| Fit | Diagnostic | Queue | Chain 0 iterations | Complete |
|---|---|---|---|---|
| step33_nestcbc_s41 | geometry | done | not started | True |
| step33_nestcbc_abs_s41 | geometry | done | not started | True |
| step34_abs_natdep_s41 | geometry | done | not started | True |
| step34_abs_v2_natdep_s41 | geometry | done | not started | True |
| step33_nestcbc_abs_s41 | nuts | done | 256/256 | True |
| step34_abs_natdep_s41 | nuts | pending | not started | False |
| step33_nestcbc_abs_s41 | nuts_block | pending | not started | False |
| step34_abs_natdep_s41 | nuts_block | pending | not started | False |

One diagonal run provides timing calibration first. Other sampler comparisons are held for measured compute-cost support; held_cost is a scheduling decision, not a model-quality failure. Warmup and sampling checkpoint every four transitions with adaptation and RNG preserved. Six hours bounds each attempt, not promised completion.

Geometry measures raw and conditional-prior-standardized curvature, numerical derivative/roughness checks and nearby curvature variation. The sampler screens compare diagonal and block-dense ecological metrics with diagonal observer nuisance blocks. Block screens have longer warmup; wall-time efficiency includes initialization, warmup and compilation, while leapfrog efficiency counts retained-draw steps only.

NUTS screens report divergences, acceptance, adapted step sizes, depth saturation, E-BFMI, NumPyro ESS/split-Rhat, and ESS per leapfrog and total wall time. These bounded near-MAP pilots are not full inference: eventual adoption requires longer dispersed chains, rank-normalized bulk/tail diagnostics, Monte Carlo error for biological estimands, and the biological pathology checklist.


## Compute-cost forecasts

Depth-saturated scenarios based on each GPU gradient timing and, once available, observed NUTS leapfrog timing. Compilation/setup overhead is additional.

| Fit | Metric | Saturated hours | Attempt budget hours | Timing calibration |
|---|---|---:|---:|---|
| step33_nestcbc_abs_s41 | nuts | 1.10 | 6.0 | True |
| step34_abs_natdep_s41 | nuts | 1.10 | 6.0 | False |
| step33_nestcbc_abs_s41 | nuts_block | 2.75 | 6.0 | False |
| step34_abs_natdep_s41 | nuts_block | 2.76 | 6.0 | False |
## geometry / step33_nestcbc_s41

Dimension 19274; gradient norm 20.93; gradient median 0.1213 seconds.

| Coordinates | Smallest Ritz estimate | Largest Ritz estimate |
|---|---:|---:|
| raw_unconstrained | 1.288 | 6.015e+05 |
| conditional_prior_scaled | 1.198 | 4.246e+05 |
## geometry / step33_nestcbc_abs_s41

Dimension 19274; gradient norm 79.7; gradient median 0.1224 seconds.

| Coordinates | Smallest Ritz estimate | Largest Ritz estimate |
|---|---:|---:|
| raw_unconstrained | 1.318 | 1.246e+06 |
| conditional_prior_scaled | 1.241 | 9.822e+05 |
## geometry / step34_abs_natdep_s41

Dimension 19275; gradient norm 13.73; gradient median 0.1233 seconds.

| Coordinates | Smallest Ritz estimate | Largest Ritz estimate |
|---|---:|---:|
| raw_unconstrained | 1.34 | 8.081e+05 |
| conditional_prior_scaled | 1.247 | 6.165e+05 |
## geometry / step34_abs_v2_natdep_s41

Dimension 19275; gradient norm 29.42; gradient median 0.123 seconds.

| Coordinates | Smallest Ritz estimate | Largest Ritz estimate |
|---|---:|---:|
| raw_unconstrained | 1.404 | 9.811e+05 |
| conditional_prior_scaled | 1.327 | 7.412e+05 |
## nuts / step33_nestcbc_abs_s41

[
  {
    "divergences": 0,
    "draws": 128,
    "mean_acceptance": 0.9672882290113528,
    "step_size_median": 0.013983374776120588,
    "mean_leapfrog_steps": 63.0,
    "depth_cap_fraction": 1.0,
    "E_BFMI": 0.03157871827709185,
    "wall_seconds_including_warmup": 2028.3170976638794
  },
  {
    "divergences": 0,
    "draws": 128,
    "mean_acceptance": 0.9711706489721219,
    "step_size_median": 0.014020562449170099,
    "mean_leapfrog_steps": 63.0,
    "depth_cap_fraction": 1.0,
    "E_BFMI": 0.03627963694480069,
    "wall_seconds_including_warmup": 1999.9786076545715
  }
]

