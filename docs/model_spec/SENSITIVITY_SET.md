# Sensitivity set reported alongside LA1

This file lists the variants the paper reports alongside the candidate. Each row says how it differs from the candidate, in config keys and CLI flags only. Everything not listed is identical to the candidate's resolved config and HMC data flags (RESOLVED_CONFIG.json).

- **Source of run IDs and results.** `research/experiments/SANDBOX_INVASION_SHAPE_2026-11.md` (main repo, working copy of 2026-11-08). Line numbers are given as `S:<line>`.
- **Overlays.** The variant overlays are the session-scratchpad `*_local.json` files named in the "overlay" column (builders `mk_lc_ov.py`, `gpuq*/` task scripts). Each key difference below was checked by diffing those overlays against `J200VW_ALL_local.json` [E]. Results are quoted from the records [E as quoted]; they were not re-measured here.

## Candidate

| id | data | kernel | Allee `pinned_sites.n50_raw` | `capacity_floor_internal` | habitat structure | objective |
|---|---|---|---|---|---|---|
| **LA1** (restart LA1p 201,573.2, S:10542) | all (holdouts removed) | J200 | −3.756179 (n50 = .0231 RC = ×.2 of the pin) | .0231039 | `capacity_demography` + Sa–K `coupling_priors` coef/map .9, sd .1, one-sided | 201,574.8 |
| LC19 / LC19p | held-out masked twin (`observation_mask.holdout_years` + `holdout_blocks` present) | J200 | same | same | same | 158,315.2 / held-out 48,024.3 (S:10354-10380) |

## Kernel axis (J330 / J200 / J110 decisive; J60A30 a lower bound only)

All kernel variants change three overlay keys: `input_dir`, `raw_z_dir` and `dispersal.juvenile_mdd_km`. The V&L scale b changes with them, and is recorded in each input's KERNEL_DISCRETISATION.json.

| kernel | `input_dir` (under `grid54/inputs_shape/`) | `raw_z_dir` | `juvenile_mdd_km` (V&L b) | other diffs | runs at the candidate setting (×.2, floor .0231, K-only sd .1) |
|---|---|---|---|---|---|
| J200 | `J200_established1951_vl_nativeMX` | `grid54/kernel_sens/J200/latent_avian_paths` | 200 (24.937 km) | — | LA1 (all data); LC19 (masked) |
| J330 | `established1951_vl_nativeMX` | `field54/stageF/latent_avian_paths_field54prod_ema_s64` | 330 (41.056 km) | `_final` comment text only | **LA2** all data 201,637.5 (init LC20; LA2p queued) (S:10517); LC20 / LC20p masked: D −.081, ETF rise ×1.68, held-out 48,194 (S:10354) |
| J110 | `J110_established1951_vl_nativeMX` | `grid54/kernel_sens/J110/latent_avian_paths` | 110 (13.715 km) | `native_departure._comment` says "Beta(17,3)" but the values are still 1.5/1.5 | **LA3** all data 201,719.4 (init LC21p) (S:10517); LC21p masked: D −.173, ×1.41, held-out 47,830 (S:10448) |
| J60A30 (lower bound) | `J60A30_established1951_nativeMX` (**no `_vl`** V&L-faithful rebuild [L from the name; no KERNEL_DISCRETISATION.json]) | `grid54/kernel_sens/J60A30/latent_avian_paths` | 60 | `dispersal.adult_mdd_km` = 30 | **no ×.2 run.** Nearest: LC18 (×.5, floor .0231, masked): ETF λ 1.63 in 1910, D −1.049 (S:10334); LC10 / LC10p (pin, floor .0462) |

## Allee axis (`pinned_sites.n50_raw`; n50 in route counts = softplus(raw))

| setting | `n50_raw` | n50 (RC) | floor in the run | J200 | J330 | J110 | all-data |
|---|---|---|---|---|---|---|---|
| pin (11-06) | −2.1 | .1155 | .0462 (+ floor diff) | LC3 / LC3q | LC4 | LC8 | none |
| ×.5, floor tied | −2.822444 | .05776 | .0231039 (= candidate) | LC11 | LC15 / LC15p | LC17 / LC17p | LC31r (J200, pending; LC31 died), **LC32** (J330) 201,630.5, **LC33** (J110) 201,933.0 (S:10547, 10542) |
| ×.5, floor .0462 | −2.822444 | .05776 | .0462078 | LC1 / LC1q | LC2r | LC9 | — |
| ×.33, floor .0231 | −3.247857 | .03812 | .0231039 | LC29 | LC30 | — | — |
| ×.33, floor .0462 | −3.247857 | .03812 | .0462078 | LC5t / LC5q | LC6s | LC13 | — |
| ×.33, floor tied | −3.247857 | .03812 | .0152486 | LC12 | LC16 | LC25r | — |
| **×.2 (candidate)** | −3.756179 | .02310 | .0231039 | LC19 / LA1 | LC20 / LA2 | LC21p / LA3 | LA1–3 |

The matrix of D / ETF rise / held-out by setting and kernel is at S:10338-10353 (decisive-kernel matrix) and S:10417-10430 (held-out by setting).

## Floor axis at the candidate Allee (×.2) (`capacity_floor_internal`)

| floor (internal) | = × ñ50 | J200 | J330 | J110 | all-data |
|---|---|---|---|---|---|
| .0462078 (original) | 40 | LC26 (S:10413) | LC27 (S:10434) | LC28 (S:10465) | — |
| **.0231039 (candidate)** | 20 | LC19 | LC20 | LC21p | LA1–3 |
| .0115520 (quarter) | 10 | LB1 (TACC); LBL1 (local, early-stopped: ignore its held-out) | LB2 | LB3 | LA4 (J200) 201,601.4 at 141 iterations, early stop; LA4p queued. LA5 / LA6 (J330 / J110) queued (S:10531-10552) |

The records' "floor = k × n50" multiples mix units: n50 is in route counts and the floor is internal (NOTATION V8/V9). The "= × ñ50" column uses ñ50 = n50/20.

## Habitat-structure axis (K-only vs four free fields)

Four-free differs from the candidate in these keys:
- `manifold_prior.form` = `exchangeable` (instead of `capacity_demography`);
- `manifold_prior.fixed_coupling` = null;
- `manifold_prior.coupling_logit_scale` = 1.0;
- `manifold_prior.coupling_priors` = {coef, map: mean .9, sd .1, one_sided} with `pairs` = "all" (6 pairs) instead of Sa–K only.

Source: S:9778-9781, "FOUR FREE FIELDS". [L] The exact overlay was not diffed here, and no four-free overlay is in the scratchpad listing that was checked.

| run | kernel | Allee | floor | result vs K-only twin (S:) |
|---|---|---|---|---|
| LC7 | J200 | pin | .0462 | −52 objective, +31 held-out, ETF rise ×1.79 vs ×1.38 (S:10260) |
| LC14 | J200 | ×.33 | .0462 | −33, +45, ×1.90 vs ×1.43 (S:10293) |
| FR1–FR4r, FRA1–FRA3 | J200 / J330 | pin / ×.5 / ×.33 | .0462 / older base | older inputs; S:9837-9935 |

There is **no four-free run at ×.2** [E: grep of the records].

## Related variant (not in the requested set)

| VWC: tighter coupling | key diff: `coupling_priors.coef.sd` = `map.sd` = .05 | LC22r (J200), LC23 (J330), LC24 (J110); masked; "no consistent gain; not pursued" (S:10381-10433) |
|---|---|---|

## Notes

- All runs share the CLI data flags of the candidate (records: "full combination"). The exceptions are runs before the specimen-count / collector stream was added. Every LC/LA run above (LC1 onwards) is from after it [L: S:~10100-10170].
- Masked (LC/LB) and all-data (LA, LC31–33) objectives are not comparable: the all-data runs include about 26.7k more BBS rows.
- The restart spread of held-out scores is about 6–23 nats (S:10403).
