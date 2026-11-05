# Full stored-run pathology evidence and coverage audit

Snapshot: 2026-11-01T21:26:42.167872+00:00

135 GRID54 directories; 110 accepted refinements; 86 accepted runs with real-observation logs. Synthetic and unknown-observation-provenance runs are excluded from the comparison tables, retained in JSON and listed below.

**Correction to the morning note:** its few highlighted runs were a selective update. The automated biological readout covers rate/region/era summaries, founders and existing metrics; it does not establish that all fifteen pathologies were fully evaluated. The earlier retrospective table also covered only cached fields. This document makes that gap explicit rather than labeling unavailable checks as passes.

Every table includes every accepted real-data run in this stored GRID54 archive. Rows are descriptive evidence, not a pooled ranking: input basis, life cycle, observation model and fitted mechanisms differ. Earlier non-GRID54 archives and detailed map interpretation remain outside this snapshot. The JSON contains complete configs, available metrics, parameter norms, precision results and errors.

## Checklist coverage and outstanding work

| ID | Evidence collected here | What remains unestablished |
|---|---|---|
| P1 historical habitat/growth | Early/modern lambda and suitable area; regional trajectories where replayed | Independent validation of historical change and spatial-map review |
| P2 expansion too fast | Fitted and frozen release-only front errors at both saved thresholds | Missing front replays for some historical runs; common fixed-threshold comparisons across models |
| P3 fecundity/growth | Fmax and local lambda distributions; realized nest output where available | Correct lifecycle-specific biological calibration for every historical arm |
| P4 juvenile ceiling | Resident Sj/Sa distribution and population share above .95 where replayed | Effective mover-weighted juvenile survival and observation-based weighting |
| P5 compensation | Joint Sa/Sj/F/K/founder evidence, matched seed/assumption comparisons | Identification of a unique demographic explanation |
| P6 western sinks | Land, fitted-population and observed-positive-record sink shares | Explicit immigration flux/replacement deficits; these weights are distinct |
| P7 tiny K | Cached K and K at positive modern records in route-count units | Geographic continuity/map assessment and sensitivity of the absence mechanism |
| P8 Allee-dead cells | Cached land share and positive-record share | Perturbed-establishment/starting-density experiments for every arm |
| P9 founders | Fixed/fitted founder conversion; founder bracket comparisons | Joint early-CBC detection likelihood and release-history validation |
| P10 movement tails | Fixed versus learned kernel configuration and saved scalar/vector parameters | Realized departure-weighted distances, tail masses and independent movement comparison |
| P11 exodus/regathering | Departure/settlement structural switches | Actual departure curves and before/after redistribution maps |
| P12 free Q escape route | Journey/decoupling structure, existing q_asymmetry metrics | Mover/stayer-weighted annual survival decomposition |
| P13 projection flexibility | Shared/private/split config, manifold metrics and parameter norms | Consistent coefficient-angle/map/extrapolation audit |
| P14 scale inflation | Parameter norms/maxima, amplitude and observer-effect settings | Effective field magnitude versus coefficient/amplitude compensation in common units |
| P15 flexible corrections | Saved trend/drift/process-noise config and metrics | Spatial contribution maps and ablations demonstrating whether corrections bypass spread |

**No run is certified free of all pathologies.** A disabled mechanism can remove a particular escape route structurally; that is not a measured biological pass. Missing means unavailable, not zero.

## Comparison strata

- Input 1: `/work/07980/bwa386/ls6/houfin/processed/field54/stageF/model_inputs/numpyro_input_field54prod_ema_s64_grid54_obsroute`
- Input 2: `/work/07980/bwa386/ls6/houfin/processed/field54/stageF/model_inputs/numpyro_input_field54prod_v2_ema_s64_grid54_obsroute`
- Input 3: `/work/07980/bwa386/ls6/houfin/processed/grid54/model_inputs/numpyro_input_fdinterp_s64_grid54_obsroute`
- Input 4: `/work/07980/bwa386/ls6/houfin/processed/grid54/model_inputs/numpyro_input_s64_grid54_b44_obsroute`
- Input 5: `/work/07980/bwa386/ls6/houfin/processed/grid54/model_inputs/numpyro_input_s64_grid54_mdd110_obsroute`
- Input 6: `/work/07980/bwa386/ls6/houfin/processed/grid54/model_inputs/numpyro_input_s64_grid54_mdd200_obsroute`
- Input 7: `/work/07980/bwa386/ls6/houfin/processed/grid54/model_inputs/numpyro_input_s64_grid54_obsroute`

## P1–P5: demography and habitat history

Modern medians are weighted by fitted population. Lambda history below uses the existing whole-land diagnostic, not an independent estimate of habitat change. F-nest is the saved realized annual fecundity at nest locations, not fledglings per nesting attempt.

| Run | Input | Yearling | Early→modern lambda | Suitable area early→modern | Sa | Sj/Sa | Pop share ratio>.95 | Fmax | F-nest |
|---|---:|---|---|---|---:|---:|---:|---:|---:|
| step10_pn05 | 7 | False | 0.838→1.31 | 0.116→0.485 | missing | missing | missing | missing | missing |
| step10_pn10 | 7 | False | 0.482→1.16 | 0.0843→0.465 | missing | missing | missing | missing | missing |
| step10_pn20 | 7 | False | 0.535→0.987 | 0.06→0.519 | missing | missing | missing | missing | missing |
| step12_lam10 | 7 | False | 0.535→1.1 | 0.119→0.465 | missing | missing | missing | missing | missing |
| step12_lam10_s42 | 7 | False | 0.414→1.07 | 0.0947→0.441 | missing | missing | missing | missing | missing |
| step12_lam10_s43 | 7 | False | 0.405→1.07 | 0.0938→0.442 | missing | missing | missing | missing | missing |
| step13_obsreg | 7 | False | 0.921→1.39 | 0.157→0.494 | missing | missing | missing | missing | missing |
| step14b_dispera | 7 | False | 0.562→1.12 | 0.0973→0.406 | missing | missing | missing | missing | missing |
| step15_lowdens | 7 | False | 0.788→1.03 | 0.126→0.523 | missing | missing | missing | missing | missing |
| step16_sthab | 7 | False | 0.452→1.09 | 0.106→0.434 | missing | missing | missing | missing | missing |
| step17_r64_regpen | 7 | False | 0.456→1.2 | 0.104→0.457 | missing | missing | missing | missing | missing |
| step17_r64_total | 7 | False | 0.472→1.21 | 0.103→0.464 | missing | missing | missing | missing | missing |
| step19b_anchor64 | 7 | False | 0.45→1.18 | 0.101→0.455 | missing | missing | missing | missing | missing |
| step19b_anchor64pen | 7 | False | 0.442→1.21 | 0.1→0.458 | missing | missing | missing | missing | missing |
| step20_jboth | 7 | False | 0.766→1.13 | 0.142→0.548 | missing | missing | missing | missing | missing |
| step20_jcost | 7 | False | 0.458→1.2 | 0.104→0.457 | missing | missing | missing | missing | missing |
| step20_jdec | 7 | False | 0.734→1.09 | 0.127→0.508 | missing | missing | missing | missing | missing |
| step21_j1 | 7 | False | 0.454→1.2 | 0.104→0.457 | missing | missing | missing | missing | missing |
| step22_b44 | 4 | False | 0.497→1.24 | 0.101→0.458 | missing | missing | missing | missing | missing |
| step22_j2 | 7 | False | 0.454→1.2 | 0.103→0.458 | missing | missing | missing | missing | missing |
| step24_attr | 7 | False | 0.685→1.38 | 0.107→0.475 | missing | missing | missing | missing | missing |
| step25_natdep | 7 | False | 0.482→1.24 | 0.115→0.472 | 0.623 | 0.964 | 0.555 | 13.2 | missing |
| step26_mdd110 | 5 | False | 0.733→1.42 | 0.255→0.55 | 0.703 | 0.918 | 0.0161 | 16.8 | missing |
| step26_mdd200 | 6 | False | 0.499→1.28 | 0.113→0.464 | 0.626 | 0.934 | 0.456 | 15.9 | missing |
| step27_dirch | 7 | False | 0.51→1.17 | 0.116→0.507 | missing | missing | missing | missing | missing |
| step29_yb_native | 1 | True | 0.419→1.76 | 0.101→0.443 | 0.517 | 0.757 | 0 | 10.5 | missing |
| step29_yb_native_s42 | 1 | True | 0.419→1.76 | 0.101→0.442 | 0.514 | 0.758 | 0 | 10.5 | missing |
| step30_comp | 7 | False | 0.453→1.2 | 0.103→0.459 | missing | missing | missing | missing | missing |
| step31_nest | 1 | True | 0.433→1.6 | 0.111→0.451 | 0.399 | 0.995 | 0.778 | 9.39 | 3.59 |
| step31_nest_bloose | 1 | True | 0.431→1.67 | 0.108→0.45 | 0.408 | 0.994 | 0.772 | 9.72 | 3.55 |
| step31_nestonly | 1 | True | 0.425→1.6 | 0.11→0.454 | 0.396 | 0.995 | 0.779 | 9.38 | 3.6 |
| step32_cbc_native | 1 | False | 0.573→1.44 | 0.131→0.507 | missing | missing | missing | missing | missing |
| step32_cbc_yb | 1 | True | 0.441→1.62 | 0.103→0.446 | 0.387 | 0.991 | 0.768 | 9.48 | missing |
| step32_cbchi_yb | 1 | True | 0.445→1.4 | 0.098→0.43 | missing | missing | missing | missing | missing |
| step32_cbclo_yb | 1 | True | 0.447→1.6 | 0.103→0.446 | missing | missing | missing | missing | missing |
| step33_nestcbc_abs_s41 | 1 | True | 0.411→1.15 | 0.108→0.414 | 0.56 | 0.479 | 0 | 8.21 | 3.57 |
| step33_nestcbc_natdep_s41 | 1 | True | 0.528→1.58 | 0.15→0.49 | 0.374 | 0.987 | 0.742 | 9.93 | 3.76 |
| step33_nestcbc_natdep_s42 | 1 | True | 0.527→1.55 | 0.151→0.487 | 0.371 | 0.988 | 0.745 | 9.82 | 3.77 |
| step33_nestcbc_s41 | 1 | True | 0.432→1.6 | 0.11→0.451 | 0.4 | 0.995 | 0.778 | 9.35 | 3.58 |
| step33_nestcbc_s42 | 1 | True | 0.432→1.6 | 0.11→0.45 | 0.399 | 0.995 | 0.777 | 9.35 | 3.59 |
| step34_abs_broodhi | 1 | True | 0.376→1.25 | 0.0954→0.425 | 0.379 | 0.98 | 0.683 | 7.23 | 3.51 |
| step34_abs_broodlo | 1 | True | 0.385→1.13 | 0.0892→0.403 | 0.534 | 0.641 | 0 | 6.24 | 2.84 |
| step34_abs_comp_natdep_s41 | 1 | True | 0.675→1.58 | 0.24→0.58 | 0.574 | 0.437 | 0 | 12.4 | 4.15 |
| step34_abs_comp_natdep_s42 | 1 | True | 0.536→1.36 | 0.173→0.497 | 0.518 | 0.544 | 0 | 9.65 | 3.8 |
| step34_abs_comp_s42 | 1 | True | 0.44→1.22 | 0.12→0.429 | 0.554 | 0.499 | 0 | 8.55 | 3.64 |
| step34_abs_foundhi | 1 | True | 0.403→1.14 | 0.101→0.406 | 0.529 | 0.545 | 0 | 7.69 | 3.47 |
| step34_abs_foundlo | 1 | True | 0.419→1.16 | 0.11→0.417 | 0.568 | 0.467 | 0 | 8.34 | 3.57 |
| step34_abs_hazard_natdep_s41 | 1 | True | 0.661→1.57 | 0.233→0.583 | 0.563 | 0.45 | 0 | 12.2 | 4.17 |
| step34_abs_hazard_natdep_s42 | 1 | True | 0.539→1.36 | 0.174→0.5 | 0.512 | 0.554 | 0 | 9.66 | 3.82 |
| step34_abs_hazard_s41 | 1 | True | 0.418→1.18 | 0.11→0.42 | 0.565 | 0.473 | 0 | 8.4 | 3.6 |
| step34_abs_hazard_s42 | 1 | True | 0.42→1.17 | 0.111→0.418 | 0.565 | 0.474 | 0 | 8.38 | 3.59 |
| step34_abs_natdep_s41 | 1 | True | 0.551→1.38 | 0.179→0.516 | 0.555 | 0.479 | 0 | 10 | 3.8 |
| step34_abs_natdep_s42 | 1 | True | 0.633→1.51 | 0.219→0.568 | 0.577 | 0.437 | 0 | 11.6 | 4 |
| step34_abs_s42 | 1 | True | 0.418→1.17 | 0.112→0.418 | 0.57 | 0.465 | 0 | 8.39 | 3.58 |
| step34_abs_v2_hazard_s41 | 2 | True | 0.423→1.11 | 0.108→0.396 | 0.509 | 0.594 | 0 | 7.19 | 3.29 |
| step34_abs_v2_hazard_s42 | 2 | True | 0.452→1.14 | 0.127→0.408 | 0.558 | 0.495 | 0 | 8.12 | 3.53 |
| step34_abs_v2_natdep_s41 | 2 | True | 0.544→1.28 | 0.158→0.472 | 0.386 | 0.962 | 0.592 | 7.44 | 3.41 |
| step34_abs_v2_natdep_s42 | 2 | True | 0.564→1.27 | 0.177→0.482 | 0.469 | 0.661 | 0 | 8.36 | 3.58 |
| step34_abs_v2_s41 | 2 | True | 0.421→1.1 | 0.107→0.394 | 0.515 | 0.582 | 0 | 7.14 | 3.26 |
| step34_abs_v2_s42 | 2 | True | 0.451→1.13 | 0.126→0.408 | 0.566 | 0.48 | 0 | 8.15 | 3.52 |
| step34_broodhi | 1 | True | 0.423→1.57 | 0.11→0.448 | 0.361 | 0.995 | 0.767 | 10.1 | 4.08 |
| step34_broodlo | 1 | True | 0.445→1.61 | 0.113→0.453 | 0.442 | 0.995 | 0.786 | 8.4 | 3.09 |
| step34_foundhi | 1 | True | 0.442→1.66 | 0.112→0.457 | 0.41 | 0.995 | 0.775 | 9.44 | 3.51 |
| step34_foundlo | 1 | True | 0.432→1.6 | 0.111→0.451 | 0.399 | 0.995 | 0.778 | 9.35 | 3.58 |
| step34_rel_comp_natdep_s41 | 1 | True | 0.527→1.6 | 0.15→0.495 | 0.376 | 0.988 | 0.748 | 10 | 3.77 |
| step34_rel_comp_natdep_s42 | 1 | True | 0.517→1.65 | 0.163→0.521 | 0.363 | 0.994 | 0.825 | 10.4 | 3.97 |
| step34_rel_comp_s41 | 1 | True | 0.435→1.61 | 0.113→0.454 | 0.399 | 0.995 | 0.779 | 9.44 | 3.6 |
| step34_rel_hazard_natdep_s41 | 1 | True | 0.527→1.6 | 0.15→0.494 | 0.377 | 0.988 | 0.748 | 9.99 | 3.75 |
| step34_rel_hazard_natdep_s42 | 1 | True | 0.535→1.6 | 0.154→0.494 | 0.375 | 0.988 | 0.751 | 9.97 | 3.76 |
| step34_rel_hazard_s41 | 1 | True | 0.431→1.61 | 0.111→0.454 | 0.398 | 0.995 | 0.779 | 9.42 | 3.6 |
| step34_rel_hazard_s42 | 1 | True | 0.434→1.6 | 0.112→0.449 | 0.398 | 0.995 | 0.779 | 9.39 | 3.59 |
| step34_rel_v2_hazard_s41 | 2 | True | 0.473→1.53 | 0.124→0.442 | 0.389 | 0.994 | 0.766 | 9.27 | 3.59 |
| step34_rel_v2_hazard_s42 | 2 | True | 0.488→1.57 | 0.13→0.451 | 0.407 | 0.997 | 0.799 | 9.07 | 3.45 |
| step34_rel_v2_natdep_s41 | 2 | True | 0.589→1.51 | 0.161→0.484 | 0.369 | 0.989 | 0.747 | 9.64 | 3.78 |
| step34_rel_v2_natdep_s42 | 2 | True | 0.567→1.54 | 0.163→0.491 | 0.367 | 0.992 | 0.785 | 9.77 | 3.8 |
| step34_rel_v2_s41 | 2 | True | 0.471→1.53 | 0.124→0.443 | 0.39 | 0.995 | 0.767 | 9.27 | 3.58 |
| step3_native | 1 | False | 0.585→1.44 | 0.136→0.498 | 0.647 | 0.998 | 0.825 | 16.6 | missing |
| step3_native_s42 | 1 | False | 0.539→1.43 | 0.151→0.5 | missing | missing | missing | missing | missing |
| step3_native_v2 | 2 | False | 0.612→1.44 | 0.147→0.501 | 0.647 | 0.998 | 0.816 | 16.4 | missing |
| step3_native_v2_s42 | 2 | False | 0.556→1.41 | 0.164→0.496 | missing | missing | missing | missing | missing |
| step3_seed43 | 7 | False | 0.452→1.22 | 0.101→0.467 | missing | missing | missing | missing | missing |
| step4_amp8 | 7 | False | 0.601→1.17 | 0.197→0.455 | missing | missing | missing | missing | missing |
| step8a_kernel_s42 | 7 | False | 0.584→1.13 | 0.097→0.408 | missing | missing | missing | missing | missing |
| step8a_native | 1 | False | 0.507→1.22 | 0.112→0.453 | 0.641 | 0.995 | 0.757 | 12.5 | missing |
| step8a_native_v2 | 2 | False | 0.525→1.21 | 0.124→0.451 | 0.647 | 0.997 | 0.763 | 12.6 | missing |
| step9_seed42 | 3 | False | 0.603→1.04 | 0.175→0.383 | missing | missing | missing | missing | missing |

## P2/P6–P9: fronts, sinks, K, establishment and founders

Front errors are model-minus-observed years at the saved 10% threshold; negative means early. K-low is the share of finite-K positive modern survey records with K < .05 route-count units. These are whole-domain records; western observed sink share uses the separate western audit. Allee-dead shares are whole-land and whole-domain positive-record shares.

| Run | Fitted front | Frozen release-only | West sink pop | West sink positive records | K-low positive records | Allee-dead land / positive records | Founders birds |
|---|---:|---:|---:|---:|---:|---|---:|
| step10_pn05 | missing | missing | missing | 0.327 | 0 | 0.000232 / 0 | 88.9 |
| step10_pn10 | missing | missing | missing | 0.25 | 0.000398 | 0.000926 / 0 | 49.4 |
| step10_pn20 | missing | missing | missing | 0.284 | 0 | 0.000463 / 0.00131 | 168 |
| step12_lam10 | missing | missing | missing | 0.299 | 0 | 0.000926 / 0.00108 | 39.6 |
| step12_lam10_s42 | missing | missing | missing | 0.311 | 0.000796 | 0.000232 / 0 | 42 |
| step12_lam10_s43 | missing | missing | missing | 0.311 | 0.000398 | 0.000463 / 5.69e-05 | 43.1 |
| step13_obsreg | missing | missing | missing | 0.409 | 0 | 0.000232 / 0 | 91.5 |
| step14b_dispera | missing | missing | missing | 0.332 | 0.000398 | 0.000232 / 0.000228 | 37.2 |
| step15_lowdens | missing | missing | missing | 0.366 | 0.000114 | 0.000232 / 5.69e-05 | 1.06e+03 |
| step16_sthab | missing | missing | missing | 0.302 | 0.00171 | 0.000463 / 0.000512 | 35 |
| step17_r64_regpen | missing | missing | missing | 0.306 | 0.000796 | 0.000232 / 0 | 36.7 |
| step17_r64_total | missing | missing | missing | 0.303 | 0.000796 | 0.000695 / 0.00279 | 36.4 |
| step19b_anchor64 | missing | missing | missing | 0.313 | 0.00119 | 0.000463 / 0 | 41.8 |
| step19b_anchor64pen | missing | missing | missing | 0.293 | 0.000796 | 0.000463 / 0 | 37.6 |
| step20_jboth | missing | missing | missing | 0.21 | 0.000341 | 0.000463 / 0.000626 | 13.2 |
| step20_jcost | missing | missing | missing | 0.306 | 0.000796 | 0 / 0 | 35.1 |
| step20_jdec | missing | missing | missing | 0.246 | 0.000569 | 0.000695 / 0.00125 | 22.7 |
| step21_j1 | missing | missing | missing | 0.306 | 0.000796 | 0.000695 / 0.00148 | 35.2 |
| step22_b44 | missing | missing | missing | 0.305 | 0.000569 | 0.000463 / 0 | 36.4 |
| step22_j2 | missing | missing | missing | 0.305 | 0.000796 | 0.000926 / 0.00171 | 37.2 |
| step24_attr | missing | missing | missing | 0.247 | 0.000683 | 0.000926 / 5.69e-05 | 17.6 |
| step25_natdep | missing | missing | 0.0843 | 0.236 | 0.000796 | 0.000695 / 0.00114 | 27.3 |
| step26_mdd110 | missing | missing | 0.0412 | 0.114 | 0.0355 | 0.000463 / 5.69e-05 | 193 |
| step26_mdd200 | missing | missing | 0.154 | 0.316 | 0.00432 | 0.00139 / 0.000228 | 171 |
| step27_dirch | missing | missing | missing | 0.249 | 0.000683 | 0.00185 / 0.00205 | 29.4 |
| step29_yb_native | missing | missing | 0.0729 | 0.202 | 0.00262 | 0.000463 / 0.000683 | 44 |
| step29_yb_native_s42 | missing | missing | 0.0734 | 0.202 | 0.00262 | 0.000695 / 0.000455 | 44.6 |
| step30_comp | missing | missing | missing | 0.304 | 0.000796 | 0.000232 / 0 | 36.3 |
| step31_nest | missing | missing | 0.0841 | 0.207 | 0.00182 | 0.000232 / 5.69e-05 | 29.8 |
| step31_nest_bloose | missing | missing | 0.0826 | 0.213 | 0.00119 | 0.00116 / 0.00182 | 36.1 |
| step31_nestonly | -3 | -38 | 0.0824 | 0.207 | 0.00119 | 0.000695 / 0.000341 | 29.6 |
| step32_cbc_native | missing | missing | missing | 0.174 | 0.000114 | 0.000463 / 0 | 77.5 |
| step32_cbc_yb | missing | missing | 0.0768 | 0.201 | 0.000569 | 0.000463 / 0.000512 | 77.5 |
| step32_cbchi_yb | missing | missing | missing | 0.203 | 0.000171 | 0.000695 / 0.000341 | 352 |
| step32_cbclo_yb | missing | missing | missing | 0.203 | 0.000171 | 0.000232 / 0 | 17.2 |
| step33_nestcbc_abs_s41 | -3 | -36 | 0.104 | 0.24 | 0.0133 | 0.0139 / 0.0101 | 77.5 |
| step33_nestcbc_natdep_s41 | -3 | -37 | 0.0288 | 0.106 | 0.000683 | 0.000695 / 5.69e-05 | 77.5 |
| step33_nestcbc_natdep_s42 | -3 | -37 | 0.0301 | 0.105 | 0.000683 | 0.00116 / 0.000512 | 77.5 |
| step33_nestcbc_s41 | -3 | -39 | 0.0847 | 0.207 | 0.00165 | 0.000232 / 0 | 77.5 |
| step33_nestcbc_s42 | -3 | -39 | 0.0849 | 0.207 | 0.00182 | 0 / 0 | 77.5 |
| step34_abs_broodhi | -3 | -38 | 0.0988 | 0.237 | 0.00182 | 0.00185 / 0.00267 | 77.5 |
| step34_abs_broodlo | -3 | -36 | 0.102 | 0.243 | 0.00375 | 0.00278 / 0.0025 | 77.5 |
| step34_abs_comp_natdep_s41 | -3 | -37 | 0.015 | 0.0257 | 0.00711 | 0.0868 / 0.0498 | 77.5 |
| step34_abs_comp_natdep_s42 | -3 | -37 | 0.027 | 0.0848 | 0.00279 | 0.0215 / 0.0132 | 77.5 |
| step34_abs_comp_s42 | -3 | -36.5 | 0.0949 | 0.222 | 0.0135 | 0.0144 / 0.0111 | 77.5 |
| step34_abs_foundhi | -3 | -38 | 0.11 | 0.248 | 0.0114 | 0.0095 / 0.00768 | 352 |
| step34_abs_foundlo | -3 | -35 | 0.104 | 0.23 | 0.0129 | 0.0146 / 0.0132 | 17.2 |
| step34_abs_hazard_natdep_s41 | -3 | -37 | 0.0138 | 0.0251 | 0.00637 | 0.0829 / 0.0448 | 77.5 |
| step34_abs_hazard_natdep_s42 | -3 | -37 | 0.0256 | 0.0768 | 0.00319 | 0.0197 / 0.0127 | 77.5 |
| step34_abs_hazard_s41 | -3 | -36 | 0.102 | 0.23 | 0.0129 | 0.013 / 0.0118 | 77.5 |
| step34_abs_hazard_s42 | -3 | -36 | 0.103 | 0.226 | 0.0132 | 0.0137 / 0.0144 | 77.5 |
| step34_abs_natdep_s41 | -3 | -37 | 0.0224 | 0.0703 | 0.0041 | 0.0287 / 0.0185 | 77.5 |
| step34_abs_natdep_s42 | -3 | -37 | 0.0158 | 0.0307 | 0.0054 | 0.0771 / 0.0458 | 77.5 |
| step34_abs_s42 | -3 | -36 | 0.103 | 0.227 | 0.0132 | 0.0144 / 0.0139 | 77.5 |
| step34_abs_v2_hazard_s41 | -3 | -36 | 0.107 | 0.246 | 0.0037 | 0.00255 / 0.000796 | 77.5 |
| step34_abs_v2_hazard_s42 | -3 | -36 | 0.102 | 0.231 | 0.0166 | 0.0155 / 0.0184 | 77.5 |
| step34_abs_v2_natdep_s41 | -3 | -38 | 0.0348 | 0.134 | 0.00182 | 0.00718 / 0.00233 | 77.5 |
| step34_abs_v2_natdep_s42 | -3 | -37 | 0.0293 | 0.106 | 0.00256 | 0.0134 / 0.0111 | 77.5 |
| step34_abs_v2_s41 | -3 | -36 | 0.109 | 0.246 | 0.00404 | 0.00278 / 0.00205 | 77.5 |
| step34_abs_v2_s42 | -3 | -36 | 0.104 | 0.231 | 0.0176 | 0.0155 / 0.0163 | 77.5 |
| step34_broodhi | -3 | -39 | 0.0874 | 0.208 | 0.00131 | 0 / 0 | 77.5 |
| step34_broodlo | -3 | -38 | 0.0813 | 0.202 | 0.00222 | 0.00162 / 0.00154 | 77.5 |
| step34_foundhi | -3 | -40 | 0.0862 | 0.209 | 0.00171 | 0.000463 / 5.69e-05 | 352 |
| step34_foundlo | -3 | -38 | 0.0845 | 0.207 | 0.00165 | 0.000695 / 5.69e-05 | 17.2 |
| step34_rel_comp_natdep_s41 | -3 | -37 | 0.0278 | 0.104 | 0.000683 | 0.000463 / 0.000569 | 77.5 |
| step34_rel_comp_natdep_s42 | -4 | -37 | 0.0248 | 0.0699 | 0.000796 | 0.000463 / 0 | 77.5 |
| step34_rel_comp_s41 | -3 | -39 | 0.0836 | 0.207 | 0.00182 | 0.000695 / 0.00176 | 77.5 |
| step34_rel_hazard_natdep_s41 | -3 | -37 | 0.0278 | 0.104 | 0.000683 | 0.000695 / 0.000626 | 77.5 |
| step34_rel_hazard_natdep_s42 | -3 | -37 | 0.0275 | 0.0997 | 0.000683 | 0.00116 / 5.69e-05 | 77.5 |
| step34_rel_hazard_s41 | -3 | -39 | 0.0835 | 0.207 | 0.00125 | 0.000926 / 0.00205 | 77.5 |
| step34_rel_hazard_s42 | -3 | -39 | 0.0858 | 0.21 | 0.00216 | 0.000463 / 0.000228 | 77.5 |
| step34_rel_v2_hazard_s41 | -3 | -38 | 0.0838 | 0.208 | 0.00091 | 0.000695 / 0.000626 | 77.5 |
| step34_rel_v2_hazard_s42 | -3 | -38 | 0.084 | 0.202 | 0.00233 | 0.000695 / 0.000228 | 77.5 |
| step34_rel_v2_natdep_s41 | -3 | -37 | 0.0316 | 0.116 | 0.00114 | 0.000926 / 0.000171 | 77.5 |
| step34_rel_v2_natdep_s42 | -3 | -37 | 0.0314 | 0.105 | 0.00114 | 0.000926 / 0.00131 | 77.5 |
| step34_rel_v2_s41 | -3 | -39 | 0.0836 | 0.207 | 0.00091 | 0.000232 / 0.000114 | 77.5 |
| step3_native | missing | missing | 0.0537 | 0.173 | 0.000284 | 0 / 0 | 0.912 |
| step3_native_s42 | missing | missing | missing | 0.173 | 0.00336 | 0.000232 / 0.000228 | 0.202 |
| step3_native_v2 | missing | missing | 0.0616 | 0.177 | 0.000114 | 0.000463 / 0 | 1.63 |
| step3_native_v2_s42 | missing | missing | missing | 0.173 | 0.00375 | 0 / 0 | 0.35 |
| step3_seed43 | missing | missing | missing | 0.295 | 0.000796 | 0 / 0 | 36.9 |
| step4_amp8 | missing | missing | missing | 0.272 | 0 | 0.000463 / 0 | 19.9 |
| step8a_kernel_s42 | missing | missing | missing | 0.333 | 0.000739 | 0 / 0 | 37.3 |
| step8a_native | missing | missing | 0.077 | 0.207 | 0.000796 | 0.000463 / 0.00165 | 2.36 |
| step8a_native_v2 | missing | missing | 0.0779 | 0.209 | 0.00091 | 0.000695 / 0.00125 | 5.93 |
| step9_seed42 | missing | missing | missing | 0.314 | 0.0025 | 0.000695 / 0.000853 | 3.95 |

## P10–P15: structural escape routes and numerical coverage

This table records structure, not a pass/fail verdict. Parameter norms and complete manifold/trend/Q diagnostics are retained in JSON. Precision means fixed-point float32/float64 replay exists; it does not imply optimizer-basin robustness.

| Run | Kernel bands | Departure / settlement | Journey | Shared field / split | Observer scale | Added corrections | Parameter summary | Precision replay |
|---|---|---|---|---|---|---|---|---|
| step10_pn05 | fixed | veit/none | coupled/default | tied/False | fixed | process_noise; K trend retained | available | missing |
| step10_pn10 | fixed | veit/none | coupled/default | tied/False | fixed | process_noise; K trend retained | available | missing |
| step10_pn20 | fixed | veit/none | coupled/default | tied/False | fixed | process_noise; K trend retained | available | missing |
| step12_lam10 | fixed | veit/none | coupled/default | tied/False | fixed | process_noise; K trend retained | available | missing |
| step12_lam10_s42 | fixed | veit/none | coupled/default | tied/False | fixed | process_noise; K trend retained | available | missing |
| step12_lam10_s43 | fixed | veit/none | coupled/default | tied/False | fixed | process_noise; K trend retained | available | missing |
| step13_obsreg | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step14b_dispera | learned | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step15_lowdens | fixed | veit/none | coupled/default | tied/False | fixed | process_noise; K trend retained | available | missing |
| step16_sthab | fixed | veit/none | coupled/default | tied/False | fixed | st_correction; K trend retained | available | missing |
| step17_r64_regpen | fixed | veit/none | coupled/default | tied/False | fixed | process_noise; K trend retained | available | missing |
| step17_r64_total | fixed | veit/none | coupled/default | tied/False | fixed | process_noise; K trend retained | available | missing |
| step19b_anchor64 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step19b_anchor64pen | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step20_jboth | fixed | veit/none | decouple,band_cost | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step20_jcost | fixed | veit/none | band_cost | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step20_jdec | fixed | veit/none | decouple | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step21_j1 | fixed | veit/none | distance_hazard | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step22_b44 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step22_j2 | fixed | veit/none | water_hazard | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step24_attr | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step25_natdep | fixed | veit+native/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step26_mdd110 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step26_mdd200 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step27_dirch | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step29_yb_native | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step29_yb_native_s42 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step30_comp | fixed | veit/none | coupled/default+compounding | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step31_nest | fixed | veit/none | distance_hazard | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step31_nest_bloose | fixed | veit/none | distance_hazard | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step31_nestonly | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | available |
| step32_cbc_native | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step32_cbc_yb | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step32_cbchi_yb | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step32_cbclo_yb | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step33_nestcbc_abs_s41 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | available |
| step33_nestcbc_natdep_s41 | fixed | veit+native/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | available |
| step33_nestcbc_natdep_s42 | fixed | veit+native/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step33_nestcbc_s41 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | available |
| step33_nestcbc_s42 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_abs_broodhi | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_abs_broodlo | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_abs_comp_natdep_s41 | fixed | veit+native/none | coupled/default+compounding | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_abs_comp_natdep_s42 | fixed | veit+native/none | coupled/default+compounding | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_abs_comp_s42 | fixed | veit/none | coupled/default+compounding | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_abs_foundhi | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_abs_foundlo | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_abs_hazard_natdep_s41 | fixed | veit+native/none | distance_hazard | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_abs_hazard_natdep_s42 | fixed | veit+native/none | distance_hazard | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_abs_hazard_s41 | fixed | veit/none | distance_hazard | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_abs_hazard_s42 | fixed | veit/none | distance_hazard | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_abs_natdep_s41 | fixed | veit+native/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_abs_natdep_s42 | fixed | veit+native/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_abs_s42 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_abs_v2_hazard_s41 | fixed | veit/none | distance_hazard | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_abs_v2_hazard_s42 | fixed | veit/none | distance_hazard | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_abs_v2_natdep_s41 | fixed | veit+native/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_abs_v2_natdep_s42 | fixed | veit+native/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_abs_v2_s41 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_abs_v2_s42 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_broodhi | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_broodlo | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_foundhi | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_foundlo | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_rel_comp_natdep_s41 | fixed | veit+native/none | coupled/default+compounding | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_rel_comp_natdep_s42 | fixed | veit+native/none | coupled/default+compounding | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_rel_comp_s41 | fixed | veit/none | coupled/default+compounding | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_rel_hazard_natdep_s41 | fixed | veit+native/none | distance_hazard | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_rel_hazard_natdep_s42 | fixed | veit+native/none | distance_hazard | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_rel_hazard_s41 | fixed | veit/none | distance_hazard | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_rel_hazard_s42 | fixed | veit/none | distance_hazard | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_rel_v2_hazard_s41 | fixed | veit/none | distance_hazard | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_rel_v2_hazard_s42 | fixed | veit/none | distance_hazard | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_rel_v2_natdep_s41 | fixed | veit+native/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_rel_v2_natdep_s42 | fixed | veit+native/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_rel_v2_s41 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step3_native | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step3_native_s42 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step3_native_v2 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step3_native_v2_s42 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step3_seed43 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step4_amp8 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step8a_kernel_s42 | learned | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step8a_native | learned | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step8a_native_v2 | learned | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step9_seed42 | learned | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |

## Prediction and run availability

Legacy temporal/spatial log scores use their saved diagnostics and are not interchangeable with the local paired scorer. Paired comparison reference, uncertainty and parity live in the existing rescore artifacts; a direct matched contrast is needed before ranking configurations.

| Run | Temporal log score | Spatial log score | Biological replay | Front replay | Positive-record context |
|---|---:|---:|---|---|---|
| step10_pn05 | -1.91 | -1.66 | missing | missing | available |
| step10_pn10 | -1.86 | -1.52 | missing | missing | available |
| step10_pn20 | -1.89 | -1.51 | missing | missing | available |
| step12_lam10 | -1.9 | -1.66 | missing | missing | available |
| step12_lam10_s42 | -1.84 | -1.51 | missing | missing | available |
| step12_lam10_s43 | -1.84 | -1.51 | missing | missing | available |
| step13_obsreg | -1.92 | -1.65 | missing | missing | available |
| step14b_dispera | -1.84 | -1.49 | missing | missing | available |
| step15_lowdens | -1.84 | -1.51 | missing | missing | available |
| step16_sthab | -1.84 | -1.49 | missing | missing | available |
| step17_r64_regpen | -1.84 | -1.49 | missing | missing | available |
| step17_r64_total | -1.84 | -1.49 | missing | missing | available |
| step19b_anchor64 | -1.85 | -1.49 | missing | missing | available |
| step19b_anchor64pen | -1.84 | -1.49 | missing | missing | available |
| step20_jboth | -1.84 | -1.5 | missing | missing | available |
| step20_jcost | -1.84 | -1.49 | missing | missing | available |
| step20_jdec | -1.85 | -1.52 | missing | missing | available |
| step21_j1 | -1.84 | -1.49 | missing | missing | available |
| step22_b44 | -1.85 | -1.49 | missing | missing | available |
| step22_j2 | -1.84 | -1.49 | missing | missing | available |
| step24_attr | -1.84 | -1.47 | missing | missing | available |
| step25_natdep | -1.84 | -1.5 | available | missing | available |
| step26_mdd110 | -1.85 | -1.48 | available | missing | available |
| step26_mdd200 | -1.85 | -1.51 | available | missing | available |
| step27_dirch | -1.84 | -1.47 | missing | missing | available |
| step29_yb_native | -1.79 | -1.42 | available | missing | available |
| step29_yb_native_s42 | -1.79 | -1.42 | available | missing | available |
| step30_comp | -1.84 | -1.49 | missing | missing | available |
| step31_nest | -1.78 | -1.43 | available | missing | available |
| step31_nest_bloose | -1.78 | -1.43 | available | missing | available |
| step31_nestonly | -1.78 | -1.43 | available | available | available |
| step32_cbc_native | -1.79 | -1.44 | missing | missing | available |
| step32_cbc_yb | -1.78 | -1.41 | available | missing | available |
| step32_cbchi_yb | -1.8 | -1.45 | missing | missing | available |
| step32_cbclo_yb | -1.78 | -1.41 | missing | missing | available |
| step33_nestcbc_abs_s41 | -1.79 | -1.43 | available | available | available |
| step33_nestcbc_natdep_s41 | -1.77 | -1.41 | available | available | available |
| step33_nestcbc_natdep_s42 | -1.77 | -1.41 | available | available | available |
| step33_nestcbc_s41 | -1.78 | -1.43 | available | available | available |
| step33_nestcbc_s42 | -1.78 | -1.43 | available | available | available |
| step34_abs_broodhi | -1.78 | -1.42 | available | available | available |
| step34_abs_broodlo | -1.78 | -1.42 | available | available | available |
| step34_abs_comp_natdep_s41 | -1.78 | -1.39 | available | available | available |
| step34_abs_comp_natdep_s42 | -1.78 | -1.42 | available | available | available |
| step34_abs_comp_s42 | -1.79 | -1.43 | available | available | available |
| step34_abs_foundhi | -1.79 | -1.42 | available | available | available |
| step34_abs_foundlo | -1.79 | -1.43 | available | available | available |
| step34_abs_hazard_natdep_s41 | -1.78 | -1.39 | available | available | available |
| step34_abs_hazard_natdep_s42 | -1.78 | -1.41 | available | available | available |
| step34_abs_hazard_s41 | -1.79 | -1.43 | available | available | available |
| step34_abs_hazard_s42 | -1.79 | -1.43 | available | available | available |
| step34_abs_natdep_s41 | -1.77 | -1.39 | available | available | available |
| step34_abs_natdep_s42 | -1.78 | -1.4 | available | available | available |
| step34_abs_s42 | -1.79 | -1.43 | available | available | available |
| step34_abs_v2_hazard_s41 | -1.78 | -1.42 | available | available | available |
| step34_abs_v2_hazard_s42 | -1.78 | -1.42 | available | available | available |
| step34_abs_v2_natdep_s41 | -1.77 | -1.4 | available | available | available |
| step34_abs_v2_natdep_s42 | -1.77 | -1.4 | available | available | available |
| step34_abs_v2_s41 | -1.78 | -1.42 | available | available | available |
| step34_abs_v2_s42 | -1.78 | -1.42 | available | available | available |
| step34_broodhi | -1.78 | -1.43 | available | available | available |
| step34_broodlo | -1.78 | -1.43 | available | available | available |
| step34_foundhi | -1.78 | -1.43 | available | available | available |
| step34_foundlo | -1.78 | -1.43 | available | available | available |
| step34_rel_comp_natdep_s41 | -1.77 | -1.41 | available | available | available |
| step34_rel_comp_natdep_s42 | -1.78 | -1.45 | available | available | available |
| step34_rel_comp_s41 | -1.78 | -1.43 | available | available | available |
| step34_rel_hazard_natdep_s41 | -1.77 | -1.41 | available | available | available |
| step34_rel_hazard_natdep_s42 | -1.77 | -1.41 | available | available | available |
| step34_rel_hazard_s41 | -1.78 | -1.43 | available | available | available |
| step34_rel_hazard_s42 | -1.78 | -1.43 | available | available | available |
| step34_rel_v2_hazard_s41 | -1.78 | -1.42 | available | available | available |
| step34_rel_v2_hazard_s42 | -1.78 | -1.43 | available | available | available |
| step34_rel_v2_natdep_s41 | -1.77 | -1.41 | available | available | available |
| step34_rel_v2_natdep_s42 | -1.77 | -1.41 | available | available | available |
| step34_rel_v2_s41 | -1.78 | -1.42 | available | available | available |
| step3_native | -1.79 | -1.41 | available | missing | available |
| step3_native_s42 | -1.79 | -1.4 | missing | missing | available |
| step3_native_v2 | -1.79 | -1.43 | available | missing | available |
| step3_native_v2_s42 | -1.79 | -1.41 | missing | missing | available |
| step3_seed43 | -1.84 | -1.49 | missing | missing | available |
| step4_amp8 | -1.88 | -1.61 | missing | missing | available |
| step8a_kernel_s42 | -1.84 | -1.48 | missing | missing | available |
| step8a_native | -1.79 | -1.43 | available | missing | available |
| step8a_native_v2 | -1.78 | -1.42 | available | missing | available |
| step9_seed42 | -1.85 | -1.54 | missing | missing | available |

## Excluded or unfinished runs

- step11_reset1966: unaccepted/pending; queue=not identified; checkpoint=False.
- step11_reset3: unaccepted/pending; queue=not identified; checkpoint=False.
- step14_dispera: unaccepted/pending; queue=not identified; checkpoint=False.
- step17_r64_free: unaccepted/pending; queue=not identified; checkpoint=False.
- step17_r64_freepen: unaccepted/pending; queue=not identified; checkpoint=False.
- step17_r64_regional: unaccepted/pending; queue=not identified; checkpoint=False.
- step18_warp64: unaccepted/pending; queue=not identified; checkpoint=False.
- step18_warp64pen: unaccepted/pending; queue=not identified; checkpoint=False.
- step19_anchor64: unaccepted/pending; queue=not identified; checkpoint=False.
- step19_anchor64pen: unaccepted/pending; queue=not identified; checkpoint=False.
- step20_jboth8a: unaccepted/pending; queue=not identified; checkpoint=False.
- step21_j1null: synthetic observations; queue=not identified; checkpoint=True.
- step21_j1rec: synthetic observations; queue=not identified; checkpoint=True.
- step22_j2null: synthetic observations; queue=not identified; checkpoint=True.
- step22_j2rec: synthetic observations; queue=not identified; checkpoint=True.
- step24_attrrec: synthetic observations; queue=not identified; checkpoint=True.
- step24_attrxallee: synthetic observations; queue=not identified; checkpoint=True.
- step24_attrxfound: synthetic observations; queue=not identified; checkpoint=True.
- step25_natdepnull: synthetic observations; queue=not identified; checkpoint=True.
- step25_natdeprec: synthetic observations; queue=not identified; checkpoint=True.
- step25_natdepxfec: synthetic observations; queue=not identified; checkpoint=True.
- step27_dirchnull: synthetic observations; queue=not identified; checkpoint=True.
- step27_dirchrec: synthetic observations; queue=not identified; checkpoint=True.
- step27_dirchxj2: synthetic observations; queue=not identified; checkpoint=True.
- step30_compcur: synthetic observations; queue=not identified; checkpoint=True.
- step30_compnull: synthetic observations; queue=not identified; checkpoint=True.
- step30_comprec: synthetic observations; queue=not identified; checkpoint=True.
- step30_compxfec: synthetic observations; queue=not identified; checkpoint=True.
- step31_j1null_nonest: synthetic observations; queue=not identified; checkpoint=True.
- step31_j1rec_nonest: synthetic observations; queue=not identified; checkpoint=True.
- step31_nestdrop: synthetic observations; queue=not identified; checkpoint=True.
- step31_nestnull: synthetic observations; queue=not identified; checkpoint=True.
- step31_nestrec: synthetic observations; queue=not identified; checkpoint=True.
- step31_nestrec_bloose: synthetic observations; queue=not identified; checkpoint=True.
- step31_nestrec_bwrong: synthetic observations; queue=not identified; checkpoint=True.
- step31_nestsub: synthetic observations; queue=not identified; checkpoint=True.
- step34_abs_comp_s41: unaccepted/pending; queue=failed; checkpoint=False.
- step34_rel_comp_s42: unaccepted/pending; queue=running; checkpoint=False.
- step34_rel_v2_s42: unaccepted/pending; queue=running; checkpoint=False.
- step3_seed42: unaccepted/pending; queue=not identified; checkpoint=False.
- step4_amp4: unaccepted/pending; queue=not identified; checkpoint=False.
- step5_inj1950: unaccepted/pending; queue=not identified; checkpoint=False.
- step5_inj1955: unaccepted/pending; queue=not identified; checkpoint=False.
- step5_inj1960: unaccepted/pending; queue=not identified; checkpoint=False.
- step7_allee005: unaccepted/pending; queue=not identified; checkpoint=False.
- step7_allee010: unaccepted/pending; queue=not identified; checkpoint=False.
- step7_allee020: unaccepted/pending; queue=not identified; checkpoint=False.
- step8a_kernel: unaccepted/pending; queue=not identified; checkpoint=False.
- step8b_brake: unaccepted/pending; queue=not identified; checkpoint=False.
