# Full stored-run pathology evidence and coverage audit

Snapshot: 2026-11-03T15:52:42.146510+00:00

281 GRID54 directories; 165 accepted refinements; 131 accepted runs with real-observation logs. Synthetic and unknown-observation-provenance runs are excluded from the comparison tables, retained in JSON and listed below.

**Correction to the morning note:** its few highlighted runs were a selective update. The automated biological readout covers rate/region/era summaries, founders and existing metrics; it does not establish that all fifteen pathologies were fully evaluated. The earlier retrospective table also covered only cached fields. This document makes that gap explicit rather than labeling unavailable checks as passes.

Every table includes every accepted real-data run in this stored GRID54 archive. Rows are descriptive evidence, not a pooled ranking: input basis, life cycle, observation model and fitted mechanisms differ. Earlier non-GRID54 archives and detailed map interpretation remain outside this snapshot. The JSON contains complete configs, available metrics, parameter norms, precision results and errors.

Actual coverage: **98/131 detailed biological replays; 86/131 front replays; 131/131 positive-record K/Allee contexts; 131/131 parameter summaries; 4/131 fixed-point precision replays.** These counts describe available evidence, not completed biological validation.

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
- Input 8: `/work/07980/bwa386/ls6/houfin/research/grid54/inputs/established1951`

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
| step34_rel_comp_s42 | 1 | True | 0.437→1.6 | 0.113→0.451 | 0.399 | 0.995 | 0.779 | 9.41 | 3.59 |
| step34_rel_hazard_natdep_s41 | 1 | True | 0.527→1.6 | 0.15→0.494 | 0.377 | 0.988 | 0.748 | 9.99 | 3.75 |
| step34_rel_hazard_natdep_s42 | 1 | True | 0.535→1.6 | 0.154→0.494 | 0.375 | 0.988 | 0.751 | 9.97 | 3.76 |
| step34_rel_hazard_s41 | 1 | True | 0.431→1.61 | 0.111→0.454 | 0.398 | 0.995 | 0.779 | 9.42 | 3.6 |
| step34_rel_hazard_s42 | 1 | True | 0.434→1.6 | 0.112→0.449 | 0.398 | 0.995 | 0.779 | 9.39 | 3.59 |
| step34_rel_v2_hazard_s41 | 2 | True | 0.473→1.53 | 0.124→0.442 | 0.389 | 0.994 | 0.766 | 9.27 | 3.59 |
| step34_rel_v2_hazard_s42 | 2 | True | 0.488→1.57 | 0.13→0.451 | 0.407 | 0.997 | 0.799 | 9.07 | 3.45 |
| step34_rel_v2_natdep_s41 | 2 | True | 0.589→1.51 | 0.161→0.484 | 0.369 | 0.989 | 0.747 | 9.64 | 3.78 |
| step34_rel_v2_natdep_s42 | 2 | True | 0.567→1.54 | 0.163→0.491 | 0.367 | 0.992 | 0.785 | 9.77 | 3.8 |
| step34_rel_v2_s41 | 2 | True | 0.471→1.53 | 0.124→0.443 | 0.39 | 0.995 | 0.767 | 9.27 | 3.58 |
| step34_rel_v2_s42 | 2 | True | 0.473→1.55 | 0.126→0.448 | 0.401 | 0.996 | 0.786 | 9.26 | 3.56 |
| step35_est1951_ctl_hi_s41 | 8 | True | 0.37→1.3 | 0.0862→0.435 | 0.437 | 0.98 | 0.684 | 6.22 | 2.94 |
| step35_est1951_ctl_lo_s41 | 8 | True | 0.376→1.25 | 0.091→0.428 | 0.423 | 0.98 | 0.689 | 6.21 | 2.99 |
| step35_est1951_ctl_mid_s41 | 8 | True | 0.376→1.26 | 0.0906→0.426 | 0.426 | 0.977 | 0.664 | 6.33 | 3 |
| step35_est1951_ctl_mid_s42 | 8 | True | 0.376→1.24 | 0.0906→0.416 | 0.419 | 0.973 | 0.642 | 6.37 | 3.05 |
| step35_est1951_nat_hi_s41 | 8 | True | 0.49→1.39 | 0.144→0.498 | 0.445 | 0.801 | 0 | 8.18 | 3.44 |
| step35_est1951_nat_lo_s41 | 8 | True | 0.473→1.28 | 0.132→0.468 | 0.389 | 0.962 | 0.597 | 7.24 | 3.34 |
| step35_est1951_nat_mid_s41 | 8 | True | 0.488→1.42 | 0.146→0.495 | 0.415 | 0.875 | 0 | 8.51 | 3.53 |
| step35_est1951_nat_mid_s42 | 8 | True | 0.488→1.48 | 0.153→0.514 | 0.432 | 0.814 | 0 | 9.15 | 3.66 |
| step36_est1951_nonest_ctl_s41 | 8 | True | 0.417→1.57 | 0.0982→0.437 | 0.548 | 0.679 | 0 | 9.79 | missing |
| step36_est1951_nonest_nat_s41 | 8 | True | missing→missing | missing→missing | missing | missing | missing | missing | missing |
| step36_nonest_ctl_s41 | 1 | True | 0.431→1.41 | 0.108→0.437 | 0.597 | 0.494 | 0 | 10.1 | missing |
| step36_nonest_ctl_s42 | 1 | True | 0.428→1.4 | 0.108→0.435 | 0.589 | 0.5 | 0 | 10 | missing |
| step36_nonest_nat_s41 | 1 | True | 0.681→1.54 | 0.184→0.528 | 0.593 | 0.383 | 0 | 13.3 | missing |
| step36_nonest_nat_s42 | 1 | True | 0.571→1.42 | 0.144→0.487 | 0.543 | 0.518 | 0 | 10.6 | missing |
| step38_cbc1951_broods_free_s41 | 8 | True | 0.536→1.87 | 0.137→0.534 | 0.25 | 0.999 | 0.772 | 13.3 | 6.42 |
| step38_cbc1951_circle2_free_s41 | 8 | True | 0.55→1.86 | 0.142→0.545 | 0.28 | 0.999 | 0.755 | 11.5 | 5.58 |
| step38_cbc1951_free_s41 | 8 | True | 0.555→1.87 | 0.142→0.535 | 0.278 | 0.999 | 0.764 | 11.8 | 5.62 |
| step38_cbc1951_free_s42 | 8 | True | 0.56→1.94 | 0.138→0.549 | 0.286 | 1 | 0.788 | 11.8 | 5.45 |
| step38_cbc1951_lo_s41 | 8 | True | 0.573→1.95 | 0.141→0.543 | 0.282 | 1 | 0.8 | 12.1 | 5.47 |
| step38_cbc1951_mid_s41 | 8 | True | 0.573→2.03 | 0.138→0.547 | 0.297 | 1 | 0.816 | 12.3 | 5.3 |
| step38_cbc1951_mid_s42 | 8 | True | 0.547→1.98 | 0.135→0.55 | 0.289 | 1 | 0.796 | 12 | 5.39 |
| step38_cbc1951_noncbc_free_s41 | 8 | True | 0.409→1.16 | 0.107→0.414 | 0.562 | 0.477 | 0 | 8.31 | 3.57 |
| step38_cbc1951_noncbc_free_s42 | 8 | True | 0.416→1.17 | 0.111→0.418 | 0.566 | 0.474 | 0 | 8.37 | 3.57 |
| step39_adult_priority_fixed_s41 | 8 | True | 4.47→7.14 | 0.721→0.836 | 0.731 | 0.999 | 0.946 | 14.9 | 1.29 |
| step39_adult_priority_fixed_s42 | 8 | True | 4.2→6.83 | 0.756→0.862 | 0.702 | 0.996 | 0.907 | 15.6 | 1.46 |
| step39_kfloor1p25_s41 | 8 | True | 0.414→1.17 | 0.109→0.417 | 0.57 | 0.466 | 0 | 8.47 | 3.6 |
| step39_kfloor1p25_s42 | 8 | True | 0.415→1.18 | 0.109→0.418 | 0.569 | 0.468 | 0 | 8.46 | 3.6 |
| step39_kfloor4p0_s41 | 8 | True | 0.373→1.22 | 0.0906→0.415 | 0.416 | 0.975 | 0.656 | 6.32 | 3.04 |
| step39_kfloor4p0_s42 | 8 | True | 0.413→1.17 | 0.108→0.416 | 0.575 | 0.454 | 0 | 8.61 | 3.66 |
| step39_shared_fixed_s41 | 8 | True | 5.31→7.81 | 0.832→0.918 | 0.686 | 0.999 | 0.936 | 18.3 | 1.62 |
| step39_shared_fixed_s42 | 8 | True | 5.01→7.69 | 0.762→0.874 | 0.672 | 0.999 | 0.947 | 18.2 | 1.69 |
| step39_veit_fixed_s41 | 8 | True | 0.414→1.17 | 0.11→0.417 | 0.567 | 0.473 | 0 | 8.38 | 3.57 |
| step39_veit_fixed_s42 | 8 | True | 0.415→1.17 | 0.111→0.418 | 0.567 | 0.472 | 0 | 8.41 | 3.58 |
| step3_native | 1 | False | 0.585→1.44 | 0.136→0.498 | 0.647 | 0.998 | 0.825 | 16.6 | missing |
| step3_native_s42 | 1 | False | 0.539→1.43 | 0.151→0.5 | missing | missing | missing | missing | missing |
| step3_native_v2 | 2 | False | 0.612→1.44 | 0.147→0.501 | 0.647 | 0.998 | 0.816 | 16.4 | missing |
| step3_native_v2_s42 | 2 | False | 0.556→1.41 | 0.164→0.496 | missing | missing | missing | missing | missing |
| step3_seed43 | 7 | False | 0.452→1.22 | 0.101→0.467 | missing | missing | missing | missing | missing |
| step40_rangev2_historical_s41 | 8 | True | 0.397→1.12 | 0.0959→0.399 | 0.524 | 0.538 | 0 | 7.74 | 3.52 |
| step40_rangev2_historical_s42 | 8 | True | 0.404→1.12 | 0.101→0.402 | 0.531 | 0.533 | 0 | 7.8 | 3.49 |
| step40_rangev2_invasion_s41 | 8 | True | 0.405→1.09 | 0.101→0.384 | 0.481 | 0.65 | 0 | 7.19 | 3.51 |
| step40_rangev2_invasion_s42 | 8 | True | 0.406→1.1 | 0.102→0.386 | 0.484 | 0.636 | 0 | 7.38 | 3.57 |
| step41_rational_adult20_s41 | 8 | True | 0.468→1.42 | 0.115→0.492 | 0.442 | 0.976 | 0.671 | 6.41 | 2.81 |
| step41_rational_adult20_s42 | 8 | True | 0.468→1.42 | 0.115→0.492 | 0.442 | 0.975 | 0.665 | 6.42 | 2.82 |
| step41_rational_shared_s42 | 8 | True | 0.431→1.37 | 0.112→0.477 | 0.438 | 0.98 | 0.73 | 6.49 | 2.91 |
| step41_veit_adult20_s41 | 8 | True | 0.441→1.16 | 0.117→0.425 | 0.608 | 0.4 | 0 | 8.62 | 3.55 |
| step41_veit_adult20_s42 | 8 | True | 0.441→1.16 | 0.118→0.426 | 0.609 | 0.399 | 0 | 8.63 | 3.55 |
| step42_power2_veit_s42 | 8 | True | 0.356→0.973 | 0.091→0.409 | 0.794 | 0.323 | 0 | 5.7 | 3.29 |
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
| step34_rel_comp_s42 | -3 | -39 | 0.0848 | 0.207 | 0.00216 | 0.000695 / 0.000739 | 77.5 |
| step34_rel_hazard_natdep_s41 | -3 | -37 | 0.0278 | 0.104 | 0.000683 | 0.000695 / 0.000626 | 77.5 |
| step34_rel_hazard_natdep_s42 | -3 | -37 | 0.0275 | 0.0997 | 0.000683 | 0.00116 / 5.69e-05 | 77.5 |
| step34_rel_hazard_s41 | -3 | -39 | 0.0835 | 0.207 | 0.00125 | 0.000926 / 0.00205 | 77.5 |
| step34_rel_hazard_s42 | -3 | -39 | 0.0858 | 0.21 | 0.00216 | 0.000463 / 0.000228 | 77.5 |
| step34_rel_v2_hazard_s41 | -3 | -38 | 0.0838 | 0.208 | 0.00091 | 0.000695 / 0.000626 | 77.5 |
| step34_rel_v2_hazard_s42 | -3 | -38 | 0.084 | 0.202 | 0.00233 | 0.000695 / 0.000228 | 77.5 |
| step34_rel_v2_natdep_s41 | -3 | -37 | 0.0316 | 0.116 | 0.00114 | 0.000926 / 0.000171 | 77.5 |
| step34_rel_v2_natdep_s42 | -3 | -37 | 0.0314 | 0.105 | 0.00114 | 0.000926 / 0.00131 | 77.5 |
| step34_rel_v2_s41 | -3 | -39 | 0.0836 | 0.207 | 0.00091 | 0.000232 / 0.000114 | 77.5 |
| step34_rel_v2_s42 | -3 | -39 | 0.0812 | 0.204 | 0.00233 | 0.000232 / 0 | 77.5 |
| step35_est1951_ctl_hi_s41 | -3 | -31 | 0.101 | 0.244 | 0.00154 | 0.00116 / 0.00142 | 1.12e+03 |
| step35_est1951_ctl_lo_s41 | -3 | -29 | 0.0968 | 0.244 | 0.00154 | 0.00162 / 0.00148 | 280 |
| step35_est1951_ctl_mid_s41 | -3 | -29 | 0.098 | 0.242 | 0.00159 | 0.00162 / 0.000853 | 560 |
| step35_est1951_ctl_mid_s42 | -3 | -29 | 0.0998 | 0.245 | 0.00159 | 0.00139 / 0.00119 | 560 |
| step35_est1951_nat_hi_s41 | -3 | -31 | 0.028 | 0.114 | 0.00239 | 0.0146 / 0.00745 | 1.12e+03 |
| step35_est1951_nat_lo_s41 | -3 | -28 | 0.0387 | 0.145 | 0.00119 | 0.00625 / 0.00518 | 280 |
| step35_est1951_nat_mid_s41 | -3 | -29 | 0.0269 | 0.103 | 0.00245 | 0.0127 / 0.00694 | 560 |
| step35_est1951_nat_mid_s42 | -3 | -30 | 0.0244 | 0.0807 | 0.00336 | 0.0222 / 0.0144 | 560 |
| step36_est1951_nonest_ctl_s41 | -3 | -30 | 0.0786 | 0.216 | 0.00728 | 0.0095 / 0.00802 | 560 |
| step36_est1951_nonest_nat_s41 | missing | missing | missing | missing | 0.00284 | 0.0234 / 0.0221 | 560 |
| step36_nonest_ctl_s41 | -3 | -38 | 0.0783 | 0.205 | 0.00768 | 0.0127 / 0.0128 | 77.5 |
| step36_nonest_ctl_s42 | -3 | -38 | 0.0774 | 0.202 | 0.00802 | 0.0146 / 0.0149 | 77.5 |
| step36_nonest_nat_s41 | -3 | -37 | 0.0148 | 0.0634 | 0.00392 | 0.0366 / 0.0214 | 77.5 |
| step36_nonest_nat_s42 | -3 | -37 | 0.0236 | 0.117 | 0.00091 | 0.0095 / 0.00779 | 77.5 |
| step38_cbc1951_broods_free_s41 | -6 | -31 | 0.138 | 0.197 | 0.000512 | 0.00208 / 0.00626 | 184 |
| step38_cbc1951_circle2_free_s41 | -6 | -31 | 0.136 | 0.201 | 0.000512 | 0.00116 / 0.00125 | 193 |
| step38_cbc1951_free_s41 | -6 | -31 | 0.14 | 0.195 | 0 | 0.00162 / 0.000853 | 196 |
| step38_cbc1951_free_s42 | -6 | -31 | 0.12 | 0.163 | 0 | 0.000695 / 0.00648 | 155 |
| step38_cbc1951_lo_s41 | -6 | -31 | 0.118 | 0.178 | 0 | 0.00162 / 0.000284 | 280 |
| step38_cbc1951_mid_s41 | -6 | -32 | 0.11 | 0.17 | 0 | 0.000695 / 0.000228 | 560 |
| step38_cbc1951_mid_s42 | -6 | -32 | 0.117 | 0.167 | 0 | 0.00116 / 0.00148 | 560 |
| step38_cbc1951_noncbc_free_s41 | -3 | -26 | 0.104 | 0.24 | 0.0131 | 0.0141 / 0.0113 | 104 |
| step38_cbc1951_noncbc_free_s42 | -3 | -25 | 0.104 | 0.225 | 0.013 | 0.0151 / 0.0154 | 98.4 |
| step39_adult_priority_fixed_s41 | -3 | -11 | 0.000171 | 0 | 0.00131 | 0.142 / 0.00683 | 103 |
| step39_adult_priority_fixed_s42 | -3 | -12 | 9.22e-05 | 0 | 0.00114 | 0.164 / 0.00677 | 103 |
| step39_kfloor1p25_s41 | -3 | -26 | 0.102 | 0.226 | 0 | 0.013 / 0.016 | 103 |
| step39_kfloor1p25_s42 | -3 | -26 | 0.103 | 0.225 | 0 | 0.0132 / 0.0156 | 103 |
| step39_kfloor4p0_s41 | -3 | -27 | 0.0993 | 0.246 | 0 | 0.00208 / 0.000569 | 103 |
| step39_kfloor4p0_s42 | -3 | -25 | 0.103 | 0.227 | 0 | 0.0132 / 0.0157 | 103 |
| step39_shared_fixed_s41 | -3 | -9.5 | 0.000261 | 0 | 0.000796 | 0.191 / 0.00614 | 103 |
| step39_shared_fixed_s42 | -3 | -10 | 0.000254 | 0.000649 | 0.000739 | 0.161 / 0.0062 | 103 |
| step39_veit_fixed_s41 | -3 | -26 | 0.103 | 0.227 | 0.013 | 0.0151 / 0.0154 | 103 |
| step39_veit_fixed_s42 | -3 | -26 | 0.103 | 0.227 | 0.0134 | 0.0155 / 0.0156 | 103 |
| step3_native | missing | missing | 0.0537 | 0.173 | 0.000284 | 0 / 0 | 0.912 |
| step3_native_s42 | missing | missing | missing | 0.173 | 0.00336 | 0.000232 / 0.000228 | 0.202 |
| step3_native_v2 | missing | missing | 0.0616 | 0.177 | 0.000114 | 0.000463 / 0 | 1.63 |
| step3_native_v2_s42 | missing | missing | missing | 0.173 | 0.00375 | 0 / 0 | 0.35 |
| step3_seed43 | missing | missing | missing | 0.295 | 0.000796 | 0 / 0 | 36.9 |
| step40_rangev2_historical_s41 | -3 | -26 | 0.129 | 0.242 | 0.0137 | 0.0123 / 0.0131 | 118 |
| step40_rangev2_historical_s42 | -3 | -26 | 0.127 | 0.238 | 0.0143 | 0.0148 / 0.0138 | 116 |
| step40_rangev2_invasion_s41 | -3 | -27 | 0.132 | 0.264 | 0.0118 | 0.00579 / 0.00466 | 387 |
| step40_rangev2_invasion_s42 | -3 | -27 | 0.133 | 0.263 | 0.0124 | 0.00602 / 0.00392 | 377 |
| step41_rational_adult20_s41 | -3 | -26 | 0.0606 | 0.188 | 0.000683 | 0.00162 / 0.000455 | 103 |
| step41_rational_adult20_s42 | -3 | -26 | 0.0597 | 0.189 | 0.000683 | 0.00255 / 0.00165 | 103 |
| step41_rational_shared_s42 | -3 | -27 | 0.0649 | 0.196 | 0.00233 | 0.00324 / 0.00193 | 103 |
| step41_veit_adult20_s41 | -3 | -24 | 0.1 | 0.218 | 0.0117 | 0.0181 / 0.0164 | 103 |
| step41_veit_adult20_s42 | -3 | -24 | 0.0995 | 0.218 | 0.0117 | 0.0188 / 0.0181 | 103 |
| step42_power2_veit_s42 | -3 | -26 | 0.101 | 0.23 | 0.00722 | 0.00371 / 0.00341 | 103 |
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
| step34_rel_comp_s42 | fixed | veit/none | coupled/default+compounding | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_rel_hazard_natdep_s41 | fixed | veit+native/none | distance_hazard | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_rel_hazard_natdep_s42 | fixed | veit+native/none | distance_hazard | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_rel_hazard_s41 | fixed | veit/none | distance_hazard | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_rel_hazard_s42 | fixed | veit/none | distance_hazard | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_rel_v2_hazard_s41 | fixed | veit/none | distance_hazard | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_rel_v2_hazard_s42 | fixed | veit/none | distance_hazard | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_rel_v2_natdep_s41 | fixed | veit+native/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_rel_v2_natdep_s42 | fixed | veit+native/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_rel_v2_s41 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step34_rel_v2_s42 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step35_est1951_ctl_hi_s41 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step35_est1951_ctl_lo_s41 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step35_est1951_ctl_mid_s41 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step35_est1951_ctl_mid_s42 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step35_est1951_nat_hi_s41 | fixed | veit+native/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step35_est1951_nat_lo_s41 | fixed | veit+native/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step35_est1951_nat_mid_s41 | fixed | veit+native/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step35_est1951_nat_mid_s42 | fixed | veit+native/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step36_est1951_nonest_ctl_s41 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step36_est1951_nonest_nat_s41 | fixed | veit+native/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step36_nonest_ctl_s41 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step36_nonest_ctl_s42 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step36_nonest_nat_s41 | fixed | veit+native/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step36_nonest_nat_s42 | fixed | veit+native/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step38_cbc1951_broods_free_s41 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step38_cbc1951_circle2_free_s41 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step38_cbc1951_free_s41 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step38_cbc1951_free_s42 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step38_cbc1951_lo_s41 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step38_cbc1951_mid_s41 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step38_cbc1951_mid_s42 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step38_cbc1951_noncbc_free_s41 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step38_cbc1951_noncbc_free_s42 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step39_adult_priority_fixed_s41 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step39_adult_priority_fixed_s42 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step39_kfloor1p25_s41 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step39_kfloor1p25_s42 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step39_kfloor4p0_s41 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step39_kfloor4p0_s42 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step39_shared_fixed_s41 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step39_shared_fixed_s42 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step39_veit_fixed_s41 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step39_veit_fixed_s42 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step3_native | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step3_native_s42 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step3_native_v2 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step3_native_v2_s42 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step3_seed43 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step40_rangev2_historical_s41 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step40_rangev2_historical_s42 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step40_rangev2_invasion_s41 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step40_rangev2_invasion_s42 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step41_rational_adult20_s41 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step41_rational_adult20_s42 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step41_rational_shared_s42 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step41_veit_adult20_s41 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step41_veit_adult20_s42 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
| step42_power2_veit_s42 | fixed | veit/none | coupled/default | tied/False | fixed | no extra enabled flag; K trend retained | available | missing |
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
| step34_rel_comp_s42 | -1.78 | -1.43 | available | available | available |
| step34_rel_hazard_natdep_s41 | -1.77 | -1.41 | available | available | available |
| step34_rel_hazard_natdep_s42 | -1.77 | -1.41 | available | available | available |
| step34_rel_hazard_s41 | -1.78 | -1.43 | available | available | available |
| step34_rel_hazard_s42 | -1.78 | -1.43 | available | available | available |
| step34_rel_v2_hazard_s41 | -1.78 | -1.42 | available | available | available |
| step34_rel_v2_hazard_s42 | -1.78 | -1.43 | available | available | available |
| step34_rel_v2_natdep_s41 | -1.77 | -1.41 | available | available | available |
| step34_rel_v2_natdep_s42 | -1.77 | -1.41 | available | available | available |
| step34_rel_v2_s41 | -1.78 | -1.42 | available | available | available |
| step34_rel_v2_s42 | -1.78 | -1.43 | available | available | available |
| step35_est1951_ctl_hi_s41 | -1.78 | -1.43 | available | available | available |
| step35_est1951_ctl_lo_s41 | -1.78 | -1.42 | available | available | available |
| step35_est1951_ctl_mid_s41 | -1.78 | -1.42 | available | available | available |
| step35_est1951_ctl_mid_s42 | -1.78 | -1.42 | available | available | available |
| step35_est1951_nat_hi_s41 | -1.78 | -1.41 | available | available | available |
| step35_est1951_nat_lo_s41 | -1.77 | -1.4 | available | available | available |
| step35_est1951_nat_mid_s41 | -1.78 | -1.42 | available | available | available |
| step35_est1951_nat_mid_s42 | -1.78 | -1.42 | available | available | available |
| step36_est1951_nonest_ctl_s41 | -1.79 | -1.43 | available | available | available |
| step36_est1951_nonest_nat_s41 | missing | missing | missing | missing | available |
| step36_nonest_ctl_s41 | -1.78 | -1.41 | available | available | available |
| step36_nonest_ctl_s42 | -1.78 | -1.41 | available | available | available |
| step36_nonest_nat_s41 | -1.77 | -1.38 | available | available | available |
| step36_nonest_nat_s42 | -1.77 | -1.4 | available | available | available |
| step38_cbc1951_broods_free_s41 | -1.87 | -1.57 | available | available | available |
| step38_cbc1951_circle2_free_s41 | -1.88 | -1.61 | available | available | available |
| step38_cbc1951_free_s41 | -1.87 | -1.59 | available | available | available |
| step38_cbc1951_free_s42 | -1.86 | -1.57 | available | available | available |
| step38_cbc1951_lo_s41 | -1.86 | -1.56 | available | available | available |
| step38_cbc1951_mid_s41 | -1.86 | -1.55 | available | available | available |
| step38_cbc1951_mid_s42 | -1.86 | -1.56 | available | available | available |
| step38_cbc1951_noncbc_free_s41 | -1.79 | -1.43 | available | available | available |
| step38_cbc1951_noncbc_free_s42 | -1.79 | -1.43 | available | available | available |
| step39_adult_priority_fixed_s41 | -1.78 | -1.35 | available | available | available |
| step39_adult_priority_fixed_s42 | -1.78 | -1.36 | available | available | available |
| step39_kfloor1p25_s41 | -1.79 | -1.43 | available | available | available |
| step39_kfloor1p25_s42 | -1.79 | -1.43 | available | available | available |
| step39_kfloor4p0_s41 | -1.78 | -1.42 | available | available | available |
| step39_kfloor4p0_s42 | -1.79 | -1.42 | available | available | available |
| step39_shared_fixed_s41 | -1.78 | -1.36 | available | available | available |
| step39_shared_fixed_s42 | -1.78 | -1.35 | available | available | available |
| step39_veit_fixed_s41 | -1.79 | -1.43 | available | available | available |
| step39_veit_fixed_s42 | -1.79 | -1.43 | available | available | available |
| step3_native | -1.79 | -1.41 | available | missing | available |
| step3_native_s42 | -1.79 | -1.4 | missing | missing | available |
| step3_native_v2 | -1.79 | -1.43 | available | missing | available |
| step3_native_v2_s42 | -1.79 | -1.41 | missing | missing | available |
| step3_seed43 | -1.84 | -1.49 | missing | missing | available |
| step40_rangev2_historical_s41 | -1.78 | -1.43 | available | available | available |
| step40_rangev2_historical_s42 | -1.79 | -1.43 | available | available | available |
| step40_rangev2_invasion_s41 | -1.78 | -1.42 | available | available | available |
| step40_rangev2_invasion_s42 | -1.78 | -1.42 | available | available | available |
| step41_rational_adult20_s41 | -1.77 | -1.4 | available | available | available |
| step41_rational_adult20_s42 | -1.77 | -1.4 | available | available | available |
| step41_rational_shared_s42 | -1.77 | -1.38 | available | available | available |
| step41_veit_adult20_s41 | -1.79 | -1.42 | available | available | available |
| step41_veit_adult20_s42 | -1.79 | -1.42 | available | available | available |
| step42_power2_veit_s42 | -1.79 | -1.43 | available | available | available |
| step4_amp8 | -1.88 | -1.61 | missing | missing | available |
| step8a_kernel_s42 | -1.84 | -1.48 | missing | missing | available |
| step8a_native | -1.79 | -1.43 | available | missing | available |
| step8a_native_v2 | -1.78 | -1.42 | available | missing | available |
| step9_seed42 | -1.85 | -1.54 | missing | missing | available |

## S2–S4/S6: actual contribution and population budgets

Added after the initial audit. Exact frozen-trajectory replays, actual-state local replacement, BIDE flow accounting, Runge breeding contribution and current-stage-weighted contribution are distinct from the intrinsic-growth classifications above. Reported immigration and export refer to cell boundaries. Full era/cell arrays, nonlinear isolation and validation receipts are in CONTRIBUTION_RESULTS_20261101.md and its linked artifact root.

| Run | Region | C breeding | R breeding | C state | No regional immigration: 2025/fitted |
|---|---|---:|---:|---:|---:|
| step29_yb_native | west | 0.983 | 0.477 | 0.987 | 0.99 |
| step29_yb_native | plains | 0.831 | 0.504 | 0.876 | 0.675 |
| step29_yb_native | broad_plains | 0.937 | 0.391 | 0.929 | 0.696 |
| step29_yb_native | east | 0.992 | 0.784 | 0.993 | 0.905 |
| step29_yb_native_s42 | west | 0.983 | 0.478 | 0.987 | 0.99 |
| step29_yb_native_s42 | plains | 0.83 | 0.501 | 0.875 | 0.674 |
| step29_yb_native_s42 | broad_plains | 0.937 | 0.389 | 0.929 | 0.694 |
| step29_yb_native_s42 | east | 0.992 | 0.783 | 0.993 | 0.904 |
| step31_nest | west | 0.986 | 0.463 | 0.99 | 0.99 |
| step31_nest | plains | 0.836 | 0.538 | 0.869 | 0.66 |
| step31_nest | broad_plains | 0.934 | 0.405 | 0.927 | 0.673 |
| step31_nest | east | 0.99 | 0.728 | 0.992 | 0.872 |
| step31_nest_bloose | west | 0.985 | 0.456 | 0.989 | 0.991 |
| step31_nest_bloose | plains | 0.828 | 0.546 | 0.865 | 0.67 |
| step31_nest_bloose | broad_plains | 0.931 | 0.411 | 0.924 | 0.68 |
| step31_nest_bloose | east | 0.99 | 0.753 | 0.992 | 0.885 |
| step31_nestonly | west | 0.986 | 0.458 | 0.99 | 0.99 |
| step31_nestonly | plains | 0.833 | 0.542 | 0.867 | 0.658 |
| step31_nestonly | broad_plains | 0.937 | 0.413 | 0.931 | 0.681 |
| step31_nestonly | east | 0.99 | 0.728 | 0.992 | 0.871 |
| step32_cbc_yb | west | 0.986 | 0.359 | 0.99 | 0.991 |
| step32_cbc_yb | plains | 0.793 | 0.504 | 0.856 | 0.657 |
| step32_cbc_yb | broad_plains | 0.926 | 0.382 | 0.929 | 0.685 |
| step32_cbc_yb | east | 0.987 | 0.743 | 0.991 | 0.892 |
| step32_cbchi_yb | west | 0.986 | 0.283 | 0.989 | 0.985 |
| step32_cbchi_yb | plains | 0.808 | 0.45 | 0.865 | 0.622 |
| step32_cbchi_yb | broad_plains | 0.939 | 0.326 | 0.938 | 0.651 |
| step32_cbchi_yb | east | 0.987 | 0.669 | 0.991 | 0.869 |
| step32_cbclo_yb | west | 0.986 | 0.351 | 0.99 | 0.991 |
| step32_cbclo_yb | plains | 0.791 | 0.502 | 0.856 | 0.653 |
| step32_cbclo_yb | broad_plains | 0.925 | 0.378 | 0.929 | 0.68 |
| step32_cbclo_yb | east | 0.987 | 0.738 | 0.991 | 0.889 |
| step33_nestcbc_abs_s41 | west | 0.988 | 0.491 | 0.99 | 0.983 |
| step33_nestcbc_abs_s41 | plains | 0.878 | 0.49 | 0.906 | 0.62 |
| step33_nestcbc_abs_s41 | broad_plains | 0.959 | 0.391 | 0.951 | 0.654 |
| step33_nestcbc_abs_s41 | east | 0.99 | 0.726 | 0.991 | 0.869 |
| step33_nestcbc_natdep_s41 | west | 0.988 | 0.811 | 0.993 | 0.991 |
| step33_nestcbc_natdep_s41 | plains | 0.891 | 0.618 | 0.916 | 0.723 |
| step33_nestcbc_natdep_s41 | broad_plains | 0.969 | 0.589 | 0.969 | 0.799 |
| step33_nestcbc_natdep_s41 | east | 0.987 | 0.661 | 0.99 | 0.853 |
| step33_nestcbc_natdep_s42 | west | 0.988 | 0.808 | 0.993 | 0.991 |
| step33_nestcbc_natdep_s42 | plains | 0.889 | 0.614 | 0.914 | 0.718 |
| step33_nestcbc_natdep_s42 | broad_plains | 0.967 | 0.58 | 0.967 | 0.789 |
| step33_nestcbc_natdep_s42 | east | 0.986 | 0.662 | 0.99 | 0.853 |
| step33_nestcbc_s41 | west | 0.986 | 0.462 | 0.99 | 0.99 |
| step33_nestcbc_s41 | plains | 0.835 | 0.537 | 0.868 | 0.659 |
| step33_nestcbc_s41 | broad_plains | 0.934 | 0.404 | 0.927 | 0.671 |
| step33_nestcbc_s41 | east | 0.99 | 0.727 | 0.992 | 0.871 |
| step33_nestcbc_s42 | west | 0.986 | 0.461 | 0.99 | 0.99 |
| step33_nestcbc_s42 | plains | 0.835 | 0.537 | 0.868 | 0.658 |
| step33_nestcbc_s42 | broad_plains | 0.933 | 0.404 | 0.927 | 0.671 |
| step33_nestcbc_s42 | east | 0.99 | 0.726 | 0.992 | 0.871 |
| step34_abs_broodhi | west | 0.988 | 0.462 | 0.99 | 0.988 |
| step34_abs_broodhi | plains | 0.845 | 0.556 | 0.88 | 0.666 |
| step34_abs_broodhi | broad_plains | 0.945 | 0.426 | 0.939 | 0.682 |
| step34_abs_broodhi | east | 0.989 | 0.739 | 0.991 | 0.879 |
| step34_abs_broodlo | west | 0.989 | 0.498 | 0.99 | 0.985 |
| step34_abs_broodlo | plains | 0.878 | 0.551 | 0.907 | 0.658 |
| step34_abs_broodlo | broad_plains | 0.959 | 0.447 | 0.954 | 0.7 |
| step34_abs_broodlo | east | 0.992 | 0.772 | 0.993 | 0.897 |
| step34_abs_comp_natdep_s41 | west | 0.987 | 0.825 | 0.994 | 0.989 |
| step34_abs_comp_natdep_s41 | plains | 0.917 | 0.525 | 0.951 | 0.693 |
| step34_abs_comp_natdep_s41 | broad_plains | 0.979 | 0.548 | 0.983 | 0.791 |
| step34_abs_comp_natdep_s41 | east | 0.989 | 0.652 | 0.991 | 0.867 |
| step34_abs_comp_natdep_s42 | west | 0.988 | 0.837 | 0.992 | 0.987 |
| step34_abs_comp_natdep_s42 | plains | 0.921 | 0.586 | 0.946 | 0.696 |
| step34_abs_comp_natdep_s42 | broad_plains | 0.983 | 0.577 | 0.983 | 0.788 |
| step34_abs_comp_natdep_s42 | east | 0.988 | 0.654 | 0.99 | 0.853 |
| step34_abs_comp_s42 | west | 0.987 | 0.503 | 0.989 | 0.986 |
| step34_abs_comp_s42 | plains | 0.881 | 0.501 | 0.906 | 0.639 |
| step34_abs_comp_s42 | broad_plains | 0.955 | 0.399 | 0.948 | 0.669 |
| step34_abs_comp_s42 | east | 0.99 | 0.737 | 0.991 | 0.88 |
| step34_abs_foundhi | west | 0.989 | 0.489 | 0.99 | 0.984 |
| step34_abs_foundhi | plains | 0.876 | 0.512 | 0.906 | 0.638 |
| step34_abs_foundhi | broad_plains | 0.958 | 0.404 | 0.951 | 0.665 |
| step34_abs_foundhi | east | 0.99 | 0.742 | 0.991 | 0.879 |
| step34_abs_foundlo | west | 0.988 | 0.492 | 0.99 | 0.983 |
| step34_abs_foundlo | plains | 0.879 | 0.489 | 0.907 | 0.619 |
| step34_abs_foundlo | broad_plains | 0.957 | 0.388 | 0.95 | 0.649 |
| step34_abs_foundlo | east | 0.99 | 0.725 | 0.991 | 0.869 |
| step34_abs_hazard_natdep_s41 | west | 0.988 | 0.839 | 0.994 | 0.989 |
| step34_abs_hazard_natdep_s41 | plains | 0.93 | 0.553 | 0.959 | 0.715 |
| step34_abs_hazard_natdep_s41 | broad_plains | 0.986 | 0.58 | 0.988 | 0.816 |
| step34_abs_hazard_natdep_s41 | east | 0.989 | 0.655 | 0.991 | 0.87 |
| step34_abs_hazard_natdep_s42 | west | 0.989 | 0.841 | 0.993 | 0.987 |
| step34_abs_hazard_natdep_s42 | plains | 0.924 | 0.586 | 0.948 | 0.696 |
| step34_abs_hazard_natdep_s42 | broad_plains | 0.984 | 0.582 | 0.983 | 0.79 |
| step34_abs_hazard_natdep_s42 | east | 0.987 | 0.645 | 0.99 | 0.849 |
| step34_abs_hazard_s41 | west | 0.988 | 0.494 | 0.99 | 0.984 |
| step34_abs_hazard_s41 | plains | 0.88 | 0.493 | 0.908 | 0.625 |
| step34_abs_hazard_s41 | broad_plains | 0.957 | 0.392 | 0.95 | 0.654 |
| step34_abs_hazard_s41 | east | 0.99 | 0.73 | 0.992 | 0.872 |
| step34_abs_hazard_s42 | west | 0.988 | 0.496 | 0.99 | 0.984 |
| step34_abs_hazard_s42 | plains | 0.882 | 0.496 | 0.909 | 0.627 |
| step34_abs_hazard_s42 | broad_plains | 0.957 | 0.394 | 0.949 | 0.655 |
| step34_abs_hazard_s42 | east | 0.99 | 0.729 | 0.992 | 0.873 |
| step34_abs_natdep_s41 | west | 0.989 | 0.84 | 0.993 | 0.986 |
| step34_abs_natdep_s41 | plains | 0.923 | 0.561 | 0.949 | 0.676 |
| step34_abs_natdep_s41 | broad_plains | 0.988 | 0.57 | 0.989 | 0.787 |
| step34_abs_natdep_s41 | east | 0.99 | 0.673 | 0.992 | 0.862 |
| step34_abs_natdep_s42 | west | 0.989 | 0.825 | 0.995 | 0.988 |
| step34_abs_natdep_s42 | plains | 0.908 | 0.519 | 0.944 | 0.663 |
| step34_abs_natdep_s42 | broad_plains | 0.982 | 0.547 | 0.985 | 0.779 |
| step34_abs_natdep_s42 | east | 0.988 | 0.637 | 0.99 | 0.849 |
| step34_abs_s42 | west | 0.988 | 0.493 | 0.99 | 0.983 |
| step34_abs_s42 | plains | 0.88 | 0.489 | 0.907 | 0.62 |
| step34_abs_s42 | broad_plains | 0.957 | 0.388 | 0.95 | 0.649 |
| step34_abs_s42 | east | 0.99 | 0.726 | 0.992 | 0.869 |
| step34_abs_v2_hazard_s41 | west | 0.989 | 0.482 | 0.99 | 0.985 |
| step34_abs_v2_hazard_s41 | plains | 0.877 | 0.558 | 0.909 | 0.67 |
| step34_abs_v2_hazard_s41 | broad_plains | 0.959 | 0.461 | 0.956 | 0.713 |
| step34_abs_v2_hazard_s41 | east | 0.988 | 0.733 | 0.99 | 0.881 |
| step34_abs_v2_hazard_s42 | west | 0.989 | 0.511 | 0.991 | 0.984 |
| step34_abs_v2_hazard_s42 | plains | 0.892 | 0.525 | 0.918 | 0.653 |
| step34_abs_v2_hazard_s42 | broad_plains | 0.962 | 0.44 | 0.958 | 0.693 |
| step34_abs_v2_hazard_s42 | east | 0.988 | 0.712 | 0.989 | 0.868 |
| step34_abs_v2_natdep_s41 | west | 0.99 | 0.835 | 0.995 | 0.989 |
| step34_abs_v2_natdep_s41 | plains | 0.911 | 0.645 | 0.936 | 0.734 |
| step34_abs_v2_natdep_s41 | broad_plains | 0.986 | 0.625 | 0.987 | 0.823 |
| step34_abs_v2_natdep_s41 | east | 0.984 | 0.676 | 0.987 | 0.861 |
| step34_abs_v2_natdep_s42 | west | 0.99 | 0.847 | 0.994 | 0.988 |
| step34_abs_v2_natdep_s42 | plains | 0.915 | 0.612 | 0.941 | 0.696 |
| step34_abs_v2_natdep_s42 | broad_plains | 0.986 | 0.614 | 0.986 | 0.807 |
| step34_abs_v2_natdep_s42 | east | 0.985 | 0.645 | 0.987 | 0.844 |
| step34_abs_v2_s41 | west | 0.989 | 0.481 | 0.99 | 0.985 |
| step34_abs_v2_s41 | plains | 0.876 | 0.554 | 0.909 | 0.665 |
| step34_abs_v2_s41 | broad_plains | 0.959 | 0.457 | 0.956 | 0.708 |
| step34_abs_v2_s41 | east | 0.988 | 0.731 | 0.99 | 0.879 |
| step34_abs_v2_s42 | west | 0.989 | 0.508 | 0.991 | 0.984 |
| step34_abs_v2_s42 | plains | 0.891 | 0.521 | 0.917 | 0.647 |
| step34_abs_v2_s42 | broad_plains | 0.962 | 0.435 | 0.958 | 0.685 |
| step34_abs_v2_s42 | east | 0.988 | 0.709 | 0.989 | 0.865 |
| step34_broodhi | west | 0.986 | 0.439 | 0.99 | 0.989 |
| step34_broodhi | plains | 0.82 | 0.52 | 0.856 | 0.643 |
| step34_broodhi | broad_plains | 0.925 | 0.374 | 0.917 | 0.635 |
| step34_broodhi | east | 0.989 | 0.71 | 0.991 | 0.858 |
| step34_broodlo | west | 0.986 | 0.487 | 0.991 | 0.99 |
| step34_broodlo | plains | 0.854 | 0.552 | 0.881 | 0.673 |
| step34_broodlo | broad_plains | 0.943 | 0.431 | 0.937 | 0.704 |
| step34_broodlo | east | 0.991 | 0.746 | 0.993 | 0.883 |
| step34_foundhi | west | 0.986 | 0.469 | 0.99 | 0.99 |
| step34_foundhi | plains | 0.843 | 0.545 | 0.875 | 0.667 |
| step34_foundhi | broad_plains | 0.94 | 0.41 | 0.932 | 0.684 |
| step34_foundhi | east | 0.989 | 0.736 | 0.991 | 0.872 |
| step34_foundlo | west | 0.986 | 0.462 | 0.99 | 0.99 |
| step34_foundlo | plains | 0.835 | 0.537 | 0.868 | 0.658 |
| step34_foundlo | broad_plains | 0.934 | 0.404 | 0.927 | 0.671 |
| step34_foundlo | east | 0.99 | 0.726 | 0.992 | 0.87 |
| step34_rel_comp_natdep_s41 | west | 0.988 | 0.81 | 0.993 | 0.991 |
| step34_rel_comp_natdep_s41 | plains | 0.89 | 0.623 | 0.917 | 0.729 |
| step34_rel_comp_natdep_s41 | broad_plains | 0.968 | 0.59 | 0.968 | 0.799 |
| step34_rel_comp_natdep_s41 | east | 0.987 | 0.668 | 0.99 | 0.857 |
| step34_rel_comp_natdep_s42 | west | 0.987 | 0.794 | 0.993 | 0.991 |
| step34_rel_comp_natdep_s42 | plains | 0.89 | 0.624 | 0.92 | 0.743 |
| step34_rel_comp_natdep_s42 | broad_plains | 0.963 | 0.581 | 0.966 | 0.8 |
| step34_rel_comp_natdep_s42 | east | 0.989 | 0.675 | 0.992 | 0.863 |
| step34_rel_comp_s41 | west | 0.986 | 0.464 | 0.99 | 0.99 |
| step34_rel_comp_s41 | plains | 0.836 | 0.539 | 0.868 | 0.659 |
| step34_rel_comp_s41 | broad_plains | 0.934 | 0.407 | 0.927 | 0.674 |
| step34_rel_comp_s41 | east | 0.99 | 0.726 | 0.992 | 0.871 |
| step34_rel_hazard_natdep_s41 | west | 0.988 | 0.809 | 0.993 | 0.991 |
| step34_rel_hazard_natdep_s41 | plains | 0.891 | 0.624 | 0.917 | 0.73 |
| step34_rel_hazard_natdep_s41 | broad_plains | 0.967 | 0.59 | 0.968 | 0.799 |
| step34_rel_hazard_natdep_s41 | east | 0.987 | 0.669 | 0.991 | 0.858 |
| step34_rel_hazard_natdep_s42 | west | 0.988 | 0.808 | 0.993 | 0.991 |
| step34_rel_hazard_natdep_s42 | plains | 0.89 | 0.623 | 0.916 | 0.728 |
| step34_rel_hazard_natdep_s42 | broad_plains | 0.966 | 0.588 | 0.966 | 0.795 |
| step34_rel_hazard_natdep_s42 | east | 0.987 | 0.667 | 0.99 | 0.856 |
| step34_rel_hazard_s41 | west | 0.986 | 0.463 | 0.99 | 0.99 |
| step34_rel_hazard_s41 | plains | 0.835 | 0.541 | 0.868 | 0.658 |
| step34_rel_hazard_s41 | broad_plains | 0.935 | 0.409 | 0.928 | 0.675 |
| step34_rel_hazard_s41 | east | 0.99 | 0.727 | 0.992 | 0.871 |
| step34_rel_hazard_s42 | west | 0.986 | 0.462 | 0.99 | 0.99 |
| step34_rel_hazard_s42 | plains | 0.836 | 0.535 | 0.869 | 0.659 |
| step34_rel_hazard_s42 | broad_plains | 0.935 | 0.404 | 0.928 | 0.675 |
| step34_rel_hazard_s42 | east | 0.99 | 0.727 | 0.992 | 0.872 |
| step34_rel_v2_hazard_s41 | west | 0.986 | 0.45 | 0.991 | 0.99 |
| step34_rel_v2_hazard_s41 | plains | 0.843 | 0.561 | 0.875 | 0.68 |
| step34_rel_v2_hazard_s41 | broad_plains | 0.945 | 0.448 | 0.942 | 0.719 |
| step34_rel_v2_hazard_s41 | east | 0.987 | 0.711 | 0.99 | 0.867 |
| step34_rel_v2_hazard_s42 | west | 0.987 | 0.482 | 0.992 | 0.99 |
| step34_rel_v2_hazard_s42 | plains | 0.855 | 0.564 | 0.882 | 0.686 |
| step34_rel_v2_hazard_s42 | broad_plains | 0.947 | 0.458 | 0.944 | 0.728 |
| step34_rel_v2_hazard_s42 | east | 0.986 | 0.706 | 0.988 | 0.861 |
| step34_rel_v2_natdep_s41 | west | 0.988 | 0.833 | 0.994 | 0.99 |
| step34_rel_v2_natdep_s41 | plains | 0.905 | 0.642 | 0.926 | 0.745 |
| step34_rel_v2_natdep_s41 | broad_plains | 0.981 | 0.619 | 0.984 | 0.833 |
| step34_rel_v2_natdep_s41 | east | 0.983 | 0.647 | 0.987 | 0.839 |
| step34_rel_v2_natdep_s42 | west | 0.987 | 0.796 | 0.993 | 0.991 |
| step34_rel_v2_natdep_s42 | plains | 0.878 | 0.615 | 0.908 | 0.723 |
| step34_rel_v2_natdep_s42 | broad_plains | 0.964 | 0.569 | 0.965 | 0.789 |
| step34_rel_v2_natdep_s42 | east | 0.985 | 0.657 | 0.989 | 0.847 |
| step34_rel_v2_s41 | west | 0.986 | 0.45 | 0.991 | 0.99 |
| step34_rel_v2_s41 | plains | 0.841 | 0.559 | 0.874 | 0.677 |
| step34_rel_v2_s41 | broad_plains | 0.944 | 0.447 | 0.941 | 0.717 |
| step34_rel_v2_s41 | east | 0.987 | 0.711 | 0.99 | 0.867 |
| step35_est1951_ctl_hi_s41 | west | 0.987 | 0.49 | 0.989 | 0.99 |
| step35_est1951_ctl_hi_s41 | plains | 0.863 | 0.611 | 0.893 | 0.699 |
| step35_est1951_ctl_hi_s41 | broad_plains | 0.948 | 0.477 | 0.944 | 0.712 |
| step35_est1951_ctl_hi_s41 | east | 0.992 | 0.796 | 0.994 | 0.904 |
| step35_est1951_ctl_lo_s41 | west | 0.988 | 0.485 | 0.99 | 0.989 |
| step35_est1951_ctl_lo_s41 | plains | 0.863 | 0.591 | 0.892 | 0.687 |
| step35_est1951_ctl_lo_s41 | broad_plains | 0.949 | 0.462 | 0.944 | 0.704 |
| step35_est1951_ctl_lo_s41 | east | 0.991 | 0.771 | 0.992 | 0.893 |
| step35_est1951_ctl_mid_s41 | west | 0.988 | 0.483 | 0.99 | 0.989 |
| step35_est1951_ctl_mid_s41 | plains | 0.862 | 0.588 | 0.892 | 0.688 |
| step35_est1951_ctl_mid_s41 | broad_plains | 0.95 | 0.462 | 0.945 | 0.711 |
| step35_est1951_ctl_mid_s41 | east | 0.991 | 0.775 | 0.993 | 0.898 |
| step35_est1951_ctl_mid_s42 | west | 0.988 | 0.477 | 0.99 | 0.989 |
| step35_est1951_ctl_mid_s42 | plains | 0.861 | 0.588 | 0.893 | 0.693 |
| step35_est1951_ctl_mid_s42 | broad_plains | 0.949 | 0.46 | 0.945 | 0.711 |
| step35_est1951_ctl_mid_s42 | east | 0.991 | 0.772 | 0.993 | 0.899 |
| step35_est1951_nat_hi_s41 | west | 0.99 | 0.828 | 0.993 | 0.989 |
| step35_est1951_nat_hi_s41 | plains | 0.901 | 0.61 | 0.929 | 0.683 |
| step35_est1951_nat_hi_s41 | broad_plains | 0.978 | 0.595 | 0.976 | 0.784 |
| step35_est1951_nat_hi_s41 | east | 0.99 | 0.724 | 0.991 | 0.884 |
| step35_est1951_nat_lo_s41 | west | 0.991 | 0.836 | 0.993 | 0.988 |
| step35_est1951_nat_lo_s41 | plains | 0.918 | 0.652 | 0.939 | 0.727 |
| step35_est1951_nat_lo_s41 | broad_plains | 0.984 | 0.627 | 0.983 | 0.81 |
| step35_est1951_nat_lo_s41 | east | 0.987 | 0.702 | 0.989 | 0.875 |
| step35_est1951_nat_mid_s41 | west | 0.99 | 0.839 | 0.993 | 0.989 |
| step35_est1951_nat_mid_s41 | plains | 0.909 | 0.619 | 0.935 | 0.703 |
| step35_est1951_nat_mid_s41 | broad_plains | 0.981 | 0.612 | 0.979 | 0.805 |
| step35_est1951_nat_mid_s41 | east | 0.987 | 0.702 | 0.989 | 0.876 |
| step35_est1951_nat_mid_s42 | west | 0.99 | 0.835 | 0.994 | 0.989 |
| step35_est1951_nat_mid_s42 | plains | 0.912 | 0.596 | 0.939 | 0.702 |
| step35_est1951_nat_mid_s42 | broad_plains | 0.985 | 0.595 | 0.984 | 0.808 |
| step35_est1951_nat_mid_s42 | east | 0.99 | 0.704 | 0.992 | 0.88 |
| step36_est1951_nonest_ctl_s41 | west | 0.983 | 0.496 | 0.987 | 0.99 |
| step36_est1951_nonest_ctl_s41 | plains | 0.844 | 0.501 | 0.891 | 0.673 |
| step36_est1951_nonest_ctl_s41 | broad_plains | 0.943 | 0.395 | 0.936 | 0.701 |
| step36_est1951_nonest_ctl_s41 | east | 0.991 | 0.761 | 0.992 | 0.9 |
| step36_nonest_ctl_s41 | west | 0.984 | 0.474 | 0.987 | 0.988 |
| step36_nonest_ctl_s41 | plains | 0.847 | 0.463 | 0.896 | 0.635 |
| step36_nonest_ctl_s41 | broad_plains | 0.943 | 0.366 | 0.939 | 0.67 |
| step36_nonest_ctl_s41 | east | 0.99 | 0.728 | 0.992 | 0.884 |
| step36_nonest_ctl_s42 | west | 0.984 | 0.468 | 0.987 | 0.988 |
| step36_nonest_ctl_s42 | plains | 0.843 | 0.455 | 0.894 | 0.628 |
| step36_nonest_ctl_s42 | broad_plains | 0.942 | 0.363 | 0.938 | 0.667 |
| step36_nonest_ctl_s42 | east | 0.99 | 0.727 | 0.992 | 0.884 |
| step36_nonest_nat_s41 | west | 0.983 | 0.806 | 0.99 | 0.99 |
| step36_nonest_nat_s41 | plains | 0.901 | 0.512 | 0.946 | 0.677 |
| step36_nonest_nat_s41 | broad_plains | 0.975 | 0.538 | 0.98 | 0.787 |
| step36_nonest_nat_s41 | east | 0.985 | 0.646 | 0.988 | 0.864 |
| step36_nonest_nat_s42 | west | 0.985 | 0.82 | 0.99 | 0.99 |
| step36_nonest_nat_s42 | plains | 0.901 | 0.59 | 0.942 | 0.72 |
| step36_nonest_nat_s42 | broad_plains | 0.974 | 0.589 | 0.977 | 0.815 |
| step36_nonest_nat_s42 | east | 0.985 | 0.673 | 0.989 | 0.875 |
| step38_cbc1951_broods_free_s41 | west | 0.99 | 0.336 | 0.993 | 0.993 |
| step38_cbc1951_broods_free_s41 | plains | 0.818 | 0.64 | 0.848 | 0.611 |
| step38_cbc1951_broods_free_s41 | broad_plains | 0.903 | 0.382 | 0.887 | 0.488 |
| step38_cbc1951_broods_free_s41 | east | 0.974 | 0.469 | 0.978 | 0.631 |
| step38_cbc1951_circle2_free_s41 | west | 0.988 | 0.318 | 0.993 | 0.992 |
| step38_cbc1951_circle2_free_s41 | plains | 0.842 | 0.666 | 0.859 | 0.61 |
| step38_cbc1951_circle2_free_s41 | broad_plains | 0.915 | 0.42 | 0.905 | 0.535 |
| step38_cbc1951_circle2_free_s41 | east | 0.98 | 0.45 | 0.986 | 0.621 |
| step38_cbc1951_free_s41 | west | 0.991 | 0.342 | 0.994 | 0.993 |
| step38_cbc1951_free_s41 | plains | 0.834 | 0.659 | 0.859 | 0.619 |
| step38_cbc1951_free_s41 | broad_plains | 0.915 | 0.402 | 0.9 | 0.519 |
| step38_cbc1951_free_s41 | east | 0.976 | 0.468 | 0.98 | 0.643 |
| step38_cbc1951_free_s42 | west | 0.991 | 0.365 | 0.995 | 0.992 |
| step38_cbc1951_free_s42 | plains | 0.838 | 0.652 | 0.866 | 0.633 |
| step38_cbc1951_free_s42 | broad_plains | 0.917 | 0.409 | 0.904 | 0.539 |
| step38_cbc1951_free_s42 | east | 0.974 | 0.489 | 0.979 | 0.663 |
| step38_cbc1951_lo_s41 | west | 0.992 | 0.346 | 0.997 | 0.993 |
| step38_cbc1951_lo_s41 | plains | 0.814 | 0.626 | 0.848 | 0.597 |
| step38_cbc1951_lo_s41 | broad_plains | 0.912 | 0.372 | 0.895 | 0.498 |
| step38_cbc1951_lo_s41 | east | 0.976 | 0.483 | 0.98 | 0.666 |
| step38_cbc1951_mid_s41 | west | 0.992 | 0.352 | 0.997 | 0.994 |
| step38_cbc1951_mid_s41 | plains | 0.815 | 0.636 | 0.853 | 0.615 |
| step38_cbc1951_mid_s41 | broad_plains | 0.917 | 0.373 | 0.898 | 0.515 |
| step38_cbc1951_mid_s41 | east | 0.98 | 0.499 | 0.984 | 0.69 |
| step38_cbc1951_mid_s42 | west | 0.992 | 0.35 | 0.996 | 0.993 |
| step38_cbc1951_mid_s42 | plains | 0.824 | 0.649 | 0.855 | 0.615 |
| step38_cbc1951_mid_s42 | broad_plains | 0.922 | 0.393 | 0.907 | 0.529 |
| step38_cbc1951_mid_s42 | east | 0.974 | 0.498 | 0.979 | 0.68 |
| step38_cbc1951_noncbc_free_s41 | west | 0.988 | 0.489 | 0.99 | 0.983 |
| step38_cbc1951_noncbc_free_s41 | plains | 0.877 | 0.489 | 0.906 | 0.62 |
| step38_cbc1951_noncbc_free_s41 | broad_plains | 0.958 | 0.39 | 0.951 | 0.653 |
| step38_cbc1951_noncbc_free_s41 | east | 0.99 | 0.729 | 0.991 | 0.87 |
| step38_cbc1951_noncbc_free_s42 | west | 0.988 | 0.493 | 0.99 | 0.984 |
| step38_cbc1951_noncbc_free_s42 | plains | 0.879 | 0.49 | 0.907 | 0.621 |
| step38_cbc1951_noncbc_free_s42 | broad_plains | 0.957 | 0.39 | 0.949 | 0.65 |
| step38_cbc1951_noncbc_free_s42 | east | 0.99 | 0.726 | 0.991 | 0.87 |
| step39_adult_priority_fixed_s41 | west | 0.972 | 0.972 | 0.982 | 1 |
| step39_adult_priority_fixed_s41 | plains | 0.989 | 0.989 | 0.999 | 1 |
| step39_adult_priority_fixed_s41 | broad_plains | 0.968 | 0.968 | 0.981 | 1 |
| step39_adult_priority_fixed_s41 | east | 1 | 1 | 1 | 1 |
| step39_adult_priority_fixed_s42 | west | 0.971 | 0.97 | 0.981 | 1 |
| step39_adult_priority_fixed_s42 | plains | 0.985 | 0.985 | 0.996 | 1 |
| step39_adult_priority_fixed_s42 | broad_plains | 0.964 | 0.964 | 0.978 | 1 |
| step39_adult_priority_fixed_s42 | east | 1 | 1 | 1 | 1 |
| step39_kfloor1p25_s41 | west | 0.988 | 0.493 | 0.99 | 0.984 |
| step39_kfloor1p25_s41 | plains | 0.878 | 0.494 | 0.907 | 0.622 |
| step39_kfloor1p25_s41 | broad_plains | 0.957 | 0.391 | 0.95 | 0.651 |
| step39_kfloor1p25_s41 | east | 0.99 | 0.729 | 0.991 | 0.871 |
| step39_kfloor1p25_s42 | west | 0.988 | 0.492 | 0.99 | 0.984 |
| step39_kfloor1p25_s42 | plains | 0.877 | 0.494 | 0.906 | 0.622 |
| step39_kfloor1p25_s42 | broad_plains | 0.957 | 0.39 | 0.95 | 0.651 |
| step39_kfloor1p25_s42 | east | 0.99 | 0.729 | 0.992 | 0.872 |
| step39_kfloor4p0_s41 | west | 0.988 | 0.481 | 0.99 | 0.989 |
| step39_kfloor4p0_s41 | plains | 0.86 | 0.59 | 0.891 | 0.692 |
| step39_kfloor4p0_s41 | broad_plains | 0.95 | 0.462 | 0.945 | 0.711 |
| step39_kfloor4p0_s41 | east | 0.99 | 0.761 | 0.992 | 0.893 |
| step39_kfloor4p0_s42 | west | 0.988 | 0.496 | 0.989 | 0.984 |
| step39_kfloor4p0_s42 | plains | 0.877 | 0.503 | 0.906 | 0.624 |
| step39_kfloor4p0_s42 | broad_plains | 0.956 | 0.394 | 0.949 | 0.65 |
| step39_kfloor4p0_s42 | east | 0.99 | 0.729 | 0.992 | 0.872 |
| step39_shared_fixed_s41 | west | 0.963 | 0.962 | 0.975 | 1 |
| step39_shared_fixed_s41 | plains | 0.976 | 0.976 | 0.99 | 1 |
| step39_shared_fixed_s41 | broad_plains | 0.957 | 0.957 | 0.972 | 1 |
| step39_shared_fixed_s41 | east | 1 | 1 | 1.01 | 1 |
| step39_shared_fixed_s42 | west | 0.96 | 0.96 | 0.974 | 1 |
| step39_shared_fixed_s42 | plains | 0.975 | 0.975 | 0.989 | 1 |
| step39_shared_fixed_s42 | broad_plains | 0.953 | 0.953 | 0.97 | 1 |
| step39_shared_fixed_s42 | east | 1 | 1 | 1.01 | 1 |
| step39_veit_fixed_s41 | west | 0.988 | 0.494 | 0.99 | 0.984 |
| step39_veit_fixed_s41 | plains | 0.879 | 0.491 | 0.907 | 0.622 |
| step39_veit_fixed_s41 | broad_plains | 0.957 | 0.391 | 0.95 | 0.652 |
| step39_veit_fixed_s41 | east | 0.99 | 0.727 | 0.991 | 0.871 |
| step39_veit_fixed_s42 | west | 0.988 | 0.493 | 0.99 | 0.984 |
| step39_veit_fixed_s42 | plains | 0.879 | 0.49 | 0.907 | 0.622 |
| step39_veit_fixed_s42 | broad_plains | 0.957 | 0.39 | 0.95 | 0.651 |
| step39_veit_fixed_s42 | east | 0.99 | 0.727 | 0.991 | 0.87 |
| step40_rangev2_historical_s41 | west | 0.99 | 0.503 | 0.991 | 0.982 |
| step40_rangev2_historical_s41 | plains | 0.904 | 0.523 | 0.925 | 0.646 |
| step40_rangev2_historical_s41 | broad_plains | 0.964 | 0.425 | 0.958 | 0.678 |
| step40_rangev2_historical_s41 | east | 0.991 | 0.733 | 0.991 | 0.883 |
| step40_rangev2_historical_s42 | west | 0.99 | 0.5 | 0.991 | 0.982 |
| step40_rangev2_historical_s42 | plains | 0.906 | 0.516 | 0.926 | 0.642 |
| step40_rangev2_historical_s42 | broad_plains | 0.965 | 0.419 | 0.959 | 0.677 |
| step40_rangev2_historical_s42 | east | 0.99 | 0.728 | 0.991 | 0.88 |
| step40_rangev2_invasion_s41 | west | 0.991 | 0.526 | 0.991 | 0.985 |
| step40_rangev2_invasion_s41 | plains | 0.922 | 0.59 | 0.939 | 0.693 |
| step40_rangev2_invasion_s41 | broad_plains | 0.964 | 0.496 | 0.959 | 0.732 |
| step40_rangev2_invasion_s41 | east | 0.991 | 0.747 | 0.991 | 0.889 |
| step40_rangev2_invasion_s42 | west | 0.991 | 0.525 | 0.991 | 0.984 |
| step40_rangev2_invasion_s42 | plains | 0.921 | 0.586 | 0.939 | 0.69 |
| step40_rangev2_invasion_s42 | broad_plains | 0.964 | 0.492 | 0.959 | 0.73 |
| step40_rangev2_invasion_s42 | east | 0.991 | 0.743 | 0.991 | 0.887 |
| step41_rational_adult20_s41 | west | 0.987 | 0.772 | 0.993 | 0.994 |
| step41_rational_adult20_s41 | plains | 0.924 | 0.778 | 0.935 | 0.746 |
| step41_rational_adult20_s41 | broad_plains | 0.974 | 0.76 | 0.975 | 0.829 |
| step41_rational_adult20_s41 | east | 0.993 | 0.861 | 0.996 | 0.927 |
| step41_rational_adult20_s42 | west | 0.987 | 0.772 | 0.993 | 0.994 |
| step41_rational_adult20_s42 | plains | 0.923 | 0.777 | 0.934 | 0.746 |
| step41_rational_adult20_s42 | broad_plains | 0.973 | 0.759 | 0.974 | 0.828 |
| step41_rational_adult20_s42 | east | 0.994 | 0.862 | 0.996 | 0.927 |
| step41_rational_shared_s42 | west | 0.987 | 0.694 | 0.993 | 0.993 |
| step41_rational_shared_s42 | plains | 0.915 | 0.715 | 0.928 | 0.732 |
| step41_rational_shared_s42 | broad_plains | 0.965 | 0.68 | 0.965 | 0.802 |
| step41_rational_shared_s42 | east | 0.995 | 0.843 | 0.997 | 0.932 |
| step41_veit_adult20_s41 | west | 0.988 | 0.664 | 0.99 | 0.984 |
| step41_veit_adult20_s41 | plains | 0.887 | 0.609 | 0.912 | 0.616 |
| step41_veit_adult20_s41 | broad_plains | 0.961 | 0.554 | 0.956 | 0.646 |
| step41_veit_adult20_s41 | east | 0.991 | 0.795 | 0.993 | 0.873 |
| step41_veit_adult20_s42 | west | 0.988 | 0.664 | 0.99 | 0.984 |
| step41_veit_adult20_s42 | plains | 0.887 | 0.61 | 0.912 | 0.617 |
| step41_veit_adult20_s42 | broad_plains | 0.961 | 0.553 | 0.955 | 0.644 |
| step41_veit_adult20_s42 | east | 0.992 | 0.797 | 0.993 | 0.874 |
| step42_power2_veit_s42 | west | 0.985 | 0.456 | 0.991 | 0.986 |
| step42_power2_veit_s42 | plains | 0.88 | 0.475 | 0.907 | 0.616 |
| step42_power2_veit_s42 | broad_plains | 0.958 | 0.36 | 0.956 | 0.672 |
| step42_power2_veit_s42 | east | 0.993 | 0.661 | 0.995 | 0.864 |

Contribution suite available for 91/131 accepted real-data rows in this snapshot. Missing rows are not passes.

## P10: realized successful permanent movement

Mean endpoint distance and tails among successful off-cell adult survivors and juvenile recruits to the next breeding census. These are conditional on successful movement, not distances of all birds or seasonal movements. Full finite-propagule comparisons are in MOVEMENT_ESTABLISHMENT_RESULTS_20261101.md.

| Run | Region | Mean km | Fraction >300 km | Fraction >1000 km |
|---|---|---:|---:|---:|
| step29_yb_native | west | 215 | 0.209 | 0.014 |
| step29_yb_native | plains | 255 | 0.263 | 0.0261 |
| step29_yb_native | east | 253 | 0.264 | 0.023 |
| step29_yb_native_s42 | west | 215 | 0.209 | 0.014 |
| step29_yb_native_s42 | plains | 255 | 0.263 | 0.0262 |
| step29_yb_native_s42 | east | 253 | 0.264 | 0.023 |
| step31_nest | west | 234 | 0.244 | 0.0158 |
| step31_nest | plains | 267 | 0.283 | 0.0294 |
| step31_nest | east | 254 | 0.272 | 0.0197 |
| step31_nest_bloose | west | 229 | 0.234 | 0.0151 |
| step31_nest_bloose | plains | 266 | 0.28 | 0.0291 |
| step31_nest_bloose | east | 256 | 0.274 | 0.0204 |
| step31_nestonly | west | 235 | 0.245 | 0.016 |
| step31_nestonly | plains | 269 | 0.285 | 0.0296 |
| step31_nestonly | east | 255 | 0.273 | 0.0198 |
| step32_cbc_yb | west | 241 | 0.253 | 0.0176 |
| step32_cbc_yb | plains | 265 | 0.279 | 0.0287 |
| step32_cbc_yb | east | 263 | 0.283 | 0.023 |
| step32_cbchi_yb | west | 241 | 0.252 | 0.0181 |
| step32_cbchi_yb | plains | 256 | 0.266 | 0.0262 |
| step32_cbchi_yb | east | 258 | 0.273 | 0.0233 |
| step32_cbclo_yb | west | 242 | 0.255 | 0.0178 |
| step32_cbclo_yb | plains | 266 | 0.28 | 0.0288 |
| step32_cbclo_yb | east | 264 | 0.285 | 0.023 |
| step33_nestcbc_abs_s41 | west | 215 | 0.208 | 0.0146 |
| step33_nestcbc_abs_s41 | plains | 242 | 0.242 | 0.0239 |
| step33_nestcbc_abs_s41 | east | 237 | 0.241 | 0.0186 |
| step33_nestcbc_natdep_s41 | west | 260 | 0.287 | 0.0209 |
| step33_nestcbc_natdep_s41 | plains | 271 | 0.29 | 0.0305 |
| step33_nestcbc_natdep_s41 | east | 259 | 0.28 | 0.021 |
| step33_nestcbc_natdep_s42 | west | 259 | 0.286 | 0.0208 |
| step33_nestcbc_natdep_s42 | plains | 271 | 0.289 | 0.0304 |
| step33_nestcbc_natdep_s42 | east | 259 | 0.279 | 0.0208 |
| step33_nestcbc_s41 | west | 235 | 0.245 | 0.0159 |
| step33_nestcbc_s41 | plains | 268 | 0.284 | 0.0296 |
| step33_nestcbc_s41 | east | 255 | 0.273 | 0.0198 |
| step33_nestcbc_s42 | west | 235 | 0.245 | 0.0159 |
| step33_nestcbc_s42 | plains | 268 | 0.284 | 0.0296 |
| step33_nestcbc_s42 | east | 255 | 0.273 | 0.0198 |
| step34_abs_broodhi | west | 232 | 0.239 | 0.016 |
| step34_abs_broodhi | plains | 260 | 0.271 | 0.028 |
| step34_abs_broodhi | east | 252 | 0.268 | 0.0194 |
| step34_abs_broodlo | west | 217 | 0.21 | 0.0146 |
| step34_abs_broodlo | plains | 242 | 0.242 | 0.0236 |
| step34_abs_broodlo | east | 236 | 0.238 | 0.0187 |
| step34_abs_comp_natdep_s41 | west | 231 | 0.234 | 0.0178 |
| step34_abs_comp_natdep_s41 | plains | 234 | 0.23 | 0.022 |
| step34_abs_comp_natdep_s41 | east | 236 | 0.235 | 0.0204 |
| step34_abs_comp_natdep_s42 | west | 244 | 0.258 | 0.0195 |
| step34_abs_comp_natdep_s42 | plains | 251 | 0.258 | 0.0258 |
| step34_abs_comp_natdep_s42 | east | 248 | 0.255 | 0.0221 |
| step34_abs_comp_s42 | west | 210 | 0.199 | 0.0133 |
| step34_abs_comp_s42 | plains | 235 | 0.233 | 0.0222 |
| step34_abs_comp_s42 | east | 232 | 0.233 | 0.017 |
| step34_abs_foundhi | west | 217 | 0.211 | 0.0147 |
| step34_abs_foundhi | plains | 244 | 0.246 | 0.0243 |
| step34_abs_foundhi | east | 239 | 0.244 | 0.0189 |
| step34_abs_foundlo | west | 215 | 0.207 | 0.0145 |
| step34_abs_foundlo | plains | 241 | 0.241 | 0.0239 |
| step34_abs_foundlo | east | 237 | 0.24 | 0.0185 |
| step34_abs_hazard_natdep_s41 | west | 230 | 0.233 | 0.0175 |
| step34_abs_hazard_natdep_s41 | plains | 233 | 0.23 | 0.0216 |
| step34_abs_hazard_natdep_s41 | east | 237 | 0.235 | 0.0211 |
| step34_abs_hazard_natdep_s42 | west | 247 | 0.262 | 0.0201 |
| step34_abs_hazard_natdep_s42 | plains | 254 | 0.262 | 0.0264 |
| step34_abs_hazard_natdep_s42 | east | 250 | 0.259 | 0.0228 |
| step34_abs_hazard_s41 | west | 214 | 0.206 | 0.0143 |
| step34_abs_hazard_s41 | plains | 240 | 0.239 | 0.0235 |
| step34_abs_hazard_s41 | east | 236 | 0.238 | 0.0183 |
| step34_abs_hazard_s42 | west | 214 | 0.206 | 0.0143 |
| step34_abs_hazard_s42 | plains | 240 | 0.239 | 0.0235 |
| step34_abs_hazard_s42 | east | 236 | 0.238 | 0.0183 |
| step34_abs_natdep_s41 | west | 241 | 0.251 | 0.0194 |
| step34_abs_natdep_s41 | plains | 246 | 0.248 | 0.0247 |
| step34_abs_natdep_s41 | east | 246 | 0.25 | 0.0227 |
| step34_abs_natdep_s42 | west | 237 | 0.243 | 0.0192 |
| step34_abs_natdep_s42 | plains | 241 | 0.24 | 0.024 |
| step34_abs_natdep_s42 | east | 245 | 0.247 | 0.0233 |
| step34_abs_s42 | west | 215 | 0.207 | 0.0145 |
| step34_abs_s42 | plains | 241 | 0.241 | 0.0239 |
| step34_abs_s42 | east | 237 | 0.24 | 0.0185 |
| step34_abs_v2_hazard_s41 | west | 218 | 0.213 | 0.0149 |
| step34_abs_v2_hazard_s41 | plains | 245 | 0.248 | 0.0242 |
| step34_abs_v2_hazard_s41 | east | 237 | 0.24 | 0.0185 |
| step34_abs_v2_hazard_s42 | west | 214 | 0.206 | 0.0144 |
| step34_abs_v2_hazard_s42 | plains | 242 | 0.243 | 0.0239 |
| step34_abs_v2_hazard_s42 | east | 236 | 0.24 | 0.0182 |
| step34_abs_v2_natdep_s41 | west | 252 | 0.271 | 0.0205 |
| step34_abs_v2_natdep_s41 | plains | 266 | 0.282 | 0.0292 |
| step34_abs_v2_natdep_s41 | east | 254 | 0.27 | 0.0204 |
| step34_abs_v2_natdep_s42 | west | 249 | 0.266 | 0.0205 |
| step34_abs_v2_natdep_s42 | plains | 260 | 0.272 | 0.0279 |
| step34_abs_v2_natdep_s42 | east | 253 | 0.264 | 0.0232 |
| step34_abs_v2_s41 | west | 219 | 0.214 | 0.0151 |
| step34_abs_v2_s41 | plains | 246 | 0.249 | 0.0246 |
| step34_abs_v2_s41 | east | 238 | 0.241 | 0.0187 |
| step34_abs_v2_s42 | west | 214 | 0.206 | 0.0146 |
| step34_abs_v2_s42 | plains | 243 | 0.244 | 0.0242 |
| step34_abs_v2_s42 | east | 237 | 0.241 | 0.0184 |
| step34_broodhi | west | 239 | 0.252 | 0.0165 |
| step34_broodhi | plains | 272 | 0.29 | 0.0307 |
| step34_broodhi | east | 258 | 0.279 | 0.0201 |
| step34_broodlo | west | 229 | 0.235 | 0.0151 |
| step34_broodlo | plains | 262 | 0.274 | 0.0282 |
| step34_broodlo | east | 251 | 0.265 | 0.0193 |
| step34_foundhi | west | 233 | 0.241 | 0.0155 |
| step34_foundhi | plains | 266 | 0.281 | 0.0291 |
| step34_foundhi | east | 254 | 0.272 | 0.0198 |
| step34_foundlo | west | 235 | 0.245 | 0.0159 |
| step34_foundlo | plains | 268 | 0.284 | 0.0296 |
| step34_foundlo | east | 255 | 0.273 | 0.0198 |
| step34_rel_comp_natdep_s41 | west | 259 | 0.286 | 0.0207 |
| step34_rel_comp_natdep_s41 | plains | 271 | 0.29 | 0.0303 |
| step34_rel_comp_natdep_s41 | east | 259 | 0.279 | 0.0209 |
| step34_rel_comp_natdep_s42 | west | 259 | 0.286 | 0.0206 |
| step34_rel_comp_natdep_s42 | plains | 272 | 0.292 | 0.0303 |
| step34_rel_comp_natdep_s42 | east | 257 | 0.277 | 0.0206 |
| step34_rel_comp_s41 | west | 234 | 0.243 | 0.0157 |
| step34_rel_comp_s41 | plains | 267 | 0.282 | 0.0293 |
| step34_rel_comp_s41 | east | 254 | 0.271 | 0.0195 |
| step34_rel_hazard_natdep_s41 | west | 259 | 0.285 | 0.0207 |
| step34_rel_hazard_natdep_s41 | plains | 271 | 0.289 | 0.0303 |
| step34_rel_hazard_natdep_s41 | east | 259 | 0.279 | 0.0209 |
| step34_rel_hazard_natdep_s42 | west | 259 | 0.285 | 0.0207 |
| step34_rel_hazard_natdep_s42 | plains | 271 | 0.289 | 0.0303 |
| step34_rel_hazard_natdep_s42 | east | 259 | 0.279 | 0.0209 |
| step34_rel_hazard_s41 | west | 235 | 0.244 | 0.0158 |
| step34_rel_hazard_s41 | plains | 268 | 0.284 | 0.0294 |
| step34_rel_hazard_s41 | east | 254 | 0.271 | 0.0197 |
| step34_rel_hazard_s42 | west | 234 | 0.244 | 0.0158 |
| step34_rel_hazard_s42 | plains | 267 | 0.283 | 0.0293 |
| step34_rel_hazard_s42 | east | 254 | 0.271 | 0.0196 |
| step34_rel_v2_hazard_s41 | west | 233 | 0.24 | 0.0158 |
| step34_rel_v2_hazard_s41 | plains | 268 | 0.285 | 0.0292 |
| step34_rel_v2_hazard_s41 | east | 254 | 0.271 | 0.0195 |
| step34_rel_v2_hazard_s42 | west | 232 | 0.239 | 0.0156 |
| step34_rel_v2_hazard_s42 | plains | 269 | 0.286 | 0.0294 |
| step34_rel_v2_hazard_s42 | east | 252 | 0.269 | 0.0191 |
| step34_rel_v2_natdep_s41 | west | 257 | 0.282 | 0.0207 |
| step34_rel_v2_natdep_s41 | plains | 273 | 0.293 | 0.031 |
| step34_rel_v2_natdep_s41 | east | 257 | 0.277 | 0.0201 |
| step34_rel_v2_natdep_s42 | west | 256 | 0.281 | 0.0202 |
| step34_rel_v2_natdep_s42 | plains | 274 | 0.293 | 0.0318 |
| step34_rel_v2_natdep_s42 | east | 259 | 0.279 | 0.0208 |
| step34_rel_v2_s41 | west | 233 | 0.241 | 0.0159 |
| step34_rel_v2_s41 | plains | 269 | 0.286 | 0.0294 |
| step34_rel_v2_s41 | east | 254 | 0.273 | 0.0196 |
| step35_est1951_ctl_hi_s41 | west | 224 | 0.225 | 0.0147 |
| step35_est1951_ctl_hi_s41 | plains | 254 | 0.261 | 0.0267 |
| step35_est1951_ctl_hi_s41 | east | 248 | 0.26 | 0.019 |
| step35_est1951_ctl_lo_s41 | west | 226 | 0.228 | 0.0151 |
| step35_est1951_ctl_lo_s41 | plains | 254 | 0.26 | 0.0266 |
| step35_est1951_ctl_lo_s41 | east | 247 | 0.259 | 0.0189 |
| step35_est1951_ctl_mid_s41 | west | 225 | 0.227 | 0.015 |
| step35_est1951_ctl_mid_s41 | plains | 254 | 0.26 | 0.0265 |
| step35_est1951_ctl_mid_s41 | east | 248 | 0.26 | 0.019 |
| step35_est1951_ctl_mid_s42 | west | 226 | 0.228 | 0.0151 |
| step35_est1951_ctl_mid_s42 | plains | 254 | 0.26 | 0.0265 |
| step35_est1951_ctl_mid_s42 | east | 248 | 0.26 | 0.019 |
| step35_est1951_nat_hi_s41 | west | 252 | 0.271 | 0.0205 |
| step35_est1951_nat_hi_s41 | plains | 265 | 0.281 | 0.0286 |
| step35_est1951_nat_hi_s41 | east | 261 | 0.275 | 0.0249 |
| step35_est1951_nat_lo_s41 | west | 255 | 0.277 | 0.0209 |
| step35_est1951_nat_lo_s41 | plains | 266 | 0.283 | 0.0288 |
| step35_est1951_nat_lo_s41 | east | 256 | 0.272 | 0.0211 |
| step35_est1951_nat_mid_s41 | west | 259 | 0.284 | 0.0217 |
| step35_est1951_nat_mid_s41 | plains | 272 | 0.293 | 0.0299 |
| step35_est1951_nat_mid_s41 | east | 264 | 0.282 | 0.025 |
| step35_est1951_nat_mid_s42 | west | 259 | 0.283 | 0.0218 |
| step35_est1951_nat_mid_s42 | plains | 270 | 0.29 | 0.0294 |
| step35_est1951_nat_mid_s42 | east | 263 | 0.28 | 0.0249 |
| step36_est1951_nonest_ctl_s41 | west | 210 | 0.2 | 0.0134 |
| step36_est1951_nonest_ctl_s41 | plains | 248 | 0.253 | 0.025 |
| step36_est1951_nonest_ctl_s41 | east | 248 | 0.255 | 0.0214 |
| step36_nonest_ctl_s41 | west | 208 | 0.194 | 0.0135 |
| step36_nonest_ctl_s41 | plains | 238 | 0.235 | 0.0229 |
| step36_nonest_ctl_s41 | east | 239 | 0.241 | 0.0204 |
| step36_nonest_ctl_s42 | west | 209 | 0.197 | 0.0138 |
| step36_nonest_ctl_s42 | plains | 239 | 0.237 | 0.023 |
| step36_nonest_ctl_s42 | east | 240 | 0.242 | 0.0206 |
| step36_nonest_nat_s41 | west | 231 | 0.233 | 0.0183 |
| step36_nonest_nat_s41 | plains | 222 | 0.208 | 0.0202 |
| step36_nonest_nat_s41 | east | 250 | 0.248 | 0.0279 |
| step36_nonest_nat_s42 | west | 238 | 0.246 | 0.0191 |
| step36_nonest_nat_s42 | plains | 237 | 0.233 | 0.023 |
| step36_nonest_nat_s42 | east | 254 | 0.258 | 0.0268 |
| step38_cbc1951_broods_free_s41 | west | 254 | 0.276 | 0.0192 |
| step38_cbc1951_broods_free_s41 | plains | 307 | 0.346 | 0.0403 |
| step38_cbc1951_broods_free_s41 | east | 265 | 0.288 | 0.0209 |
| step38_cbc1951_circle2_free_s41 | west | 256 | 0.281 | 0.0189 |
| step38_cbc1951_circle2_free_s41 | plains | 309 | 0.352 | 0.0398 |
| step38_cbc1951_circle2_free_s41 | east | 259 | 0.277 | 0.0201 |
| step38_cbc1951_free_s41 | west | 253 | 0.274 | 0.0188 |
| step38_cbc1951_free_s41 | plains | 304 | 0.341 | 0.0391 |
| step38_cbc1951_free_s41 | east | 260 | 0.279 | 0.0201 |
| step38_cbc1951_free_s42 | west | 252 | 0.274 | 0.0191 |
| step38_cbc1951_free_s42 | plains | 302 | 0.337 | 0.0389 |
| step38_cbc1951_free_s42 | east | 262 | 0.282 | 0.0209 |
| step38_cbc1951_lo_s41 | west | 252 | 0.274 | 0.0187 |
| step38_cbc1951_lo_s41 | plains | 302 | 0.337 | 0.0399 |
| step38_cbc1951_lo_s41 | east | 262 | 0.282 | 0.0209 |
| step38_cbc1951_mid_s41 | west | 250 | 0.271 | 0.0183 |
| step38_cbc1951_mid_s41 | plains | 303 | 0.338 | 0.0404 |
| step38_cbc1951_mid_s41 | east | 258 | 0.276 | 0.0202 |
| step38_cbc1951_mid_s42 | west | 253 | 0.276 | 0.0189 |
| step38_cbc1951_mid_s42 | plains | 307 | 0.346 | 0.0405 |
| step38_cbc1951_mid_s42 | east | 261 | 0.28 | 0.0207 |
| step38_cbc1951_noncbc_free_s41 | west | 215 | 0.208 | 0.0145 |
| step38_cbc1951_noncbc_free_s41 | plains | 242 | 0.242 | 0.0239 |
| step38_cbc1951_noncbc_free_s41 | east | 237 | 0.241 | 0.0186 |
| step38_cbc1951_noncbc_free_s42 | west | 215 | 0.208 | 0.0146 |
| step38_cbc1951_noncbc_free_s42 | plains | 242 | 0.242 | 0.024 |
| step38_cbc1951_noncbc_free_s42 | east | 237 | 0.241 | 0.0186 |
| step39_adult_priority_fixed_s41 | west | 317 | 0.374 | 0.0357 |
| step39_adult_priority_fixed_s41 | plains | 388 | 0.467 | 0.058 |
| step39_adult_priority_fixed_s41 | east | 294 | 0.333 | 0.0307 |
| step39_adult_priority_fixed_s42 | west | 316 | 0.375 | 0.0339 |
| step39_adult_priority_fixed_s42 | plains | 411 | 0.507 | 0.0613 |
| step39_adult_priority_fixed_s42 | east | 301 | 0.347 | 0.0317 |
| step39_kfloor1p25_s41 | west | 215 | 0.208 | 0.0145 |
| step39_kfloor1p25_s41 | plains | 241 | 0.241 | 0.0239 |
| step39_kfloor1p25_s41 | east | 237 | 0.24 | 0.0185 |
| step39_kfloor1p25_s42 | west | 215 | 0.208 | 0.0145 |
| step39_kfloor1p25_s42 | plains | 241 | 0.241 | 0.0239 |
| step39_kfloor1p25_s42 | east | 237 | 0.241 | 0.0186 |
| step39_kfloor4p0_s41 | west | 227 | 0.23 | 0.0153 |
| step39_kfloor4p0_s41 | plains | 254 | 0.26 | 0.0265 |
| step39_kfloor4p0_s41 | east | 247 | 0.26 | 0.0189 |
| step39_kfloor4p0_s42 | west | 215 | 0.207 | 0.0144 |
| step39_kfloor4p0_s42 | plains | 240 | 0.24 | 0.0237 |
| step39_kfloor4p0_s42 | east | 237 | 0.24 | 0.0184 |
| step39_shared_fixed_s41 | west | 262 | 0.271 | 0.0293 |
| step39_shared_fixed_s41 | plains | 322 | 0.363 | 0.0431 |
| step39_shared_fixed_s41 | east | 217 | 0.202 | 0.0177 |
| step39_shared_fixed_s42 | west | 271 | 0.282 | 0.0343 |
| step39_shared_fixed_s42 | plains | 312 | 0.343 | 0.042 |
| step39_shared_fixed_s42 | east | 220 | 0.208 | 0.0177 |
| step39_veit_fixed_s41 | west | 215 | 0.208 | 0.0146 |
| step39_veit_fixed_s41 | plains | 242 | 0.242 | 0.024 |
| step39_veit_fixed_s41 | east | 237 | 0.241 | 0.0186 |
| step39_veit_fixed_s42 | west | 215 | 0.208 | 0.0146 |
| step39_veit_fixed_s42 | plains | 242 | 0.242 | 0.024 |
| step39_veit_fixed_s42 | east | 237 | 0.241 | 0.0186 |
| step40_rangev2_historical_s41 | west | 218 | 0.214 | 0.0152 |
| step40_rangev2_historical_s41 | plains | 247 | 0.253 | 0.0244 |
| step40_rangev2_historical_s41 | east | 242 | 0.249 | 0.0193 |
| step40_rangev2_historical_s42 | west | 218 | 0.213 | 0.0151 |
| step40_rangev2_historical_s42 | plains | 247 | 0.253 | 0.0243 |
| step40_rangev2_historical_s42 | east | 241 | 0.248 | 0.0192 |
| step40_rangev2_invasion_s41 | west | 221 | 0.218 | 0.0154 |
| step40_rangev2_invasion_s41 | plains | 251 | 0.26 | 0.0247 |
| step40_rangev2_invasion_s41 | east | 246 | 0.256 | 0.02 |
| step40_rangev2_invasion_s42 | west | 221 | 0.217 | 0.0154 |
| step40_rangev2_invasion_s42 | plains | 252 | 0.261 | 0.0248 |
| step40_rangev2_invasion_s42 | east | 246 | 0.256 | 0.02 |
| step41_rational_adult20_s41 | west | 273 | 0.311 | 0.0224 |
| step41_rational_adult20_s41 | plains | 303 | 0.345 | 0.0347 |
| step41_rational_adult20_s41 | east | 284 | 0.323 | 0.0248 |
| step41_rational_adult20_s42 | west | 273 | 0.311 | 0.0223 |
| step41_rational_adult20_s42 | plains | 303 | 0.345 | 0.0346 |
| step41_rational_adult20_s42 | east | 285 | 0.323 | 0.0249 |
| step41_rational_shared_s42 | west | 228 | 0.23 | 0.0162 |
| step41_rational_shared_s42 | plains | 260 | 0.272 | 0.0269 |
| step41_rational_shared_s42 | east | 252 | 0.267 | 0.0205 |
| step41_veit_adult20_s41 | west | 267 | 0.3 | 0.022 |
| step41_veit_adult20_s41 | plains | 292 | 0.327 | 0.0333 |
| step41_veit_adult20_s41 | east | 280 | 0.315 | 0.0249 |
| step41_veit_adult20_s42 | west | 268 | 0.3 | 0.022 |
| step41_veit_adult20_s42 | plains | 292 | 0.327 | 0.0333 |
| step41_veit_adult20_s42 | east | 280 | 0.315 | 0.025 |
| step42_power2_veit_s42 | west | 200 | 0.179 | 0.0123 |
| step42_power2_veit_s42 | plains | 222 | 0.207 | 0.0201 |
| step42_power2_veit_s42 | east | 217 | 0.206 | 0.0154 |

## HMC geometry and efficiency (separate from biological P1–P15)

Full unconstrained posterior including transformation Jacobians and observer nuisance parameters. Curvature, roughness and sampler performance are separate evidence. Raw or prior-standardized Krylov Ritz ratios are not certified full Hessian condition numbers. No MAP-only result certifies HMC suitability. See HMC_DIAGNOSTIC_RESULTS_20261101.md for all cases and sampler metrics.

| Run | Geometry | Diagonal NUTS | Block-dense NUTS |
|---|---|---|---|
| step33_nestcbc_abs_s41 | complete | complete | pending |
| step33_nestcbc_s41 | complete | not available | not available |
| step34_abs_natdep_s41 | complete | pending | pending |
| step34_abs_v2_natdep_s41 | complete | not available | not available |

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
- step37_cbc1951_broods_free_s41: unaccepted/pending; queue=not identified; checkpoint=False.
- step37_cbc1951_broods_free_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step37_cbc1951_circle2_free_s41: unaccepted/pending; queue=not identified; checkpoint=False.
- step37_cbc1951_circle2_free_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step37_cbc1951_free_s41: unaccepted/pending; queue=not identified; checkpoint=False.
- step37_cbc1951_free_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step37_cbc1951_free_s43: unaccepted/pending; queue=not identified; checkpoint=False.
- step37_cbc1951_hi_s41: unaccepted/pending; queue=not identified; checkpoint=False.
- step37_cbc1951_hi_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step37_cbc1951_lo_s41: unaccepted/pending; queue=not identified; checkpoint=False.
- step37_cbc1951_lo_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step37_cbc1951_mid_s41: unaccepted/pending; queue=not identified; checkpoint=False.
- step37_cbc1951_mid_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step37_cbc1951_mid_s43: unaccepted/pending; queue=not identified; checkpoint=False.
- step37_cbc1951_noncbc_free_s41: unaccepted/pending; queue=not identified; checkpoint=False.
- step37_cbc1951_noncbc_free_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step37_cbc1951_sw600_free_s41: unaccepted/pending; queue=not identified; checkpoint=False.
- step37_cbc1951_sw600_free_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step37_cbc1951_sw600_mid_s41: unaccepted/pending; queue=not identified; checkpoint=False.
- step37_cbc1951_sw600_mid_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step38_cbc1951_broods_free_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step38_cbc1951_circle2_free_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step38_cbc1951_free_s43: unaccepted/pending; queue=not identified; checkpoint=False.
- step38_cbc1951_hi_s41: unaccepted/pending; queue=not identified; checkpoint=False.
- step38_cbc1951_hi_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step38_cbc1951_lo_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step38_cbc1951_mid_s43: unaccepted/pending; queue=not identified; checkpoint=False.
- step38_cbc1951_sw600_free_s41: unaccepted/pending; queue=not identified; checkpoint=True.
- step38_cbc1951_sw600_free_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step38_cbc1951_sw600_mid_s41: unaccepted/pending; queue=not identified; checkpoint=False.
- step38_cbc1951_sw600_mid_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step39_early_envelope_s41: unaccepted/pending; queue=not identified; checkpoint=False.
- step39_early_envelope_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step39_early_survey_s41: unaccepted/pending; queue=not identified; checkpoint=False.
- step39_early_survey_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step39_priority_floor4_s41: unaccepted/pending; queue=not identified; checkpoint=False.
- step39_priority_floor4_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step3_seed42: unaccepted/pending; queue=not identified; checkpoint=False.
- step40_range_historical_s41: unaccepted/pending; queue=not identified; checkpoint=False.
- step40_range_historical_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step40_range_invasion_s41: unaccepted/pending; queue=not identified; checkpoint=False.
- step40_range_invasion_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step41_rational_shared_s41: unknown observation provenance; queue=not identified; checkpoint=True.
- step42_power2_rational_adult20_s41: unknown observation provenance; queue=not identified; checkpoint=True.
- step42_power2_rational_adult20_s42: unknown observation provenance; queue=not identified; checkpoint=True.
- step42_power2_veit_s41: unknown observation provenance; queue=not identified; checkpoint=True.
- step43_factor_p1_rational_adult20_s41: unknown observation provenance; queue=not identified; checkpoint=True.
- step43_factor_p1_rational_adult20_s42: unknown observation provenance; queue=not identified; checkpoint=True.
- step43_factor_p2_rational_adult20_s41: unknown observation provenance; queue=not identified; checkpoint=True.
- step43_factor_p2_rational_adult20_s42: unknown observation provenance; queue=not identified; checkpoint=True.
- step43_factor_p2_veit_s41: unknown observation provenance; queue=not identified; checkpoint=True.
- step43_factor_p2_veit_s42: unknown observation provenance; queue=not identified; checkpoint=True.
- step43_native_floor0_s41: unaccepted/pending; queue=not identified; checkpoint=False.
- step43_native_floor0_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step43_native_floor4_s41: unaccepted/pending; queue=not identified; checkpoint=False.
- step43_native_floor4_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step43_reprovar_floor0_s41: unaccepted/pending; queue=not identified; checkpoint=False.
- step43_reprovar_floor0_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step43_reprovar_floor4_s41: unaccepted/pending; queue=not identified; checkpoint=False.
- step43_reprovar_floor4_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step43_settlement_floor0_s41: unaccepted/pending; queue=not identified; checkpoint=False.
- step43_settlement_floor0_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step43_settlement_floor4_s41: unaccepted/pending; queue=not identified; checkpoint=False.
- step43_settlement_floor4_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step44_allrho99_s41: unaccepted/pending; queue=not identified; checkpoint=False.
- step44_allrho99_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step44_allrho99_s43: unaccepted/pending; queue=not identified; checkpoint=False.
- step44_kprivate10_s41: unaccepted/pending; queue=not identified; checkpoint=False.
- step44_kprivate10_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step44_kprivate10_s43: unaccepted/pending; queue=not identified; checkpoint=False.
- step44_kprivate25_s41: unaccepted/pending; queue=not identified; checkpoint=False.
- step44_kprivate25_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step44_kprivate25_s43: unaccepted/pending; queue=not identified; checkpoint=False.
- step44_shared_s43: unaccepted/pending; queue=not identified; checkpoint=False.
- step45_cell_iid_s005_s41: unaccepted/pending; queue=not identified; checkpoint=False.
- step45_cell_iid_s005_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step45_cell_iid_s005_s43: unaccepted/pending; queue=not identified; checkpoint=False.
- step45_cell_iid_s010_s41: unaccepted/pending; queue=not identified; checkpoint=False.
- step45_cell_iid_s010_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step45_cell_iid_s010_s43: unaccepted/pending; queue=not identified; checkpoint=False.
- step45_cell_iid_s020_s41: unaccepted/pending; queue=not identified; checkpoint=False.
- step45_cell_iid_s020_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step45_cell_iid_s020_s43: unaccepted/pending; queue=not identified; checkpoint=False.
- step45_regional_s005_s41: unaccepted/pending; queue=not identified; checkpoint=False.
- step45_regional_s005_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step45_regional_s005_s43: unaccepted/pending; queue=not identified; checkpoint=False.
- step45_regional_s010_s41: unaccepted/pending; queue=not identified; checkpoint=False.
- step45_regional_s010_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step45_regional_s010_s43: unaccepted/pending; queue=not identified; checkpoint=False.
- step45_regional_s020_s41: unaccepted/pending; queue=not identified; checkpoint=False.
- step45_regional_s020_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step45_regional_s020_s43: unaccepted/pending; queue=not identified; checkpoint=False.
- step46_capacity05_s41: unaccepted/pending; queue=not identified; checkpoint=False.
- step46_capacity05_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step46_capacity05_s43: unaccepted/pending; queue=not identified; checkpoint=False.
- step46_uniform05_s41: unaccepted/pending; queue=not identified; checkpoint=False.
- step46_uniform05_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step46_uniform05_s43: unaccepted/pending; queue=not identified; checkpoint=False.
- step47_historical_floor4_s41: unaccepted/pending; queue=not identified; checkpoint=False.
- step47_historical_floor4_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step47_invasion_floor4_s41: unaccepted/pending; queue=not identified; checkpoint=False.
- step47_invasion_floor4_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step47_original_floor4_s41: unaccepted/pending; queue=not identified; checkpoint=False.
- step47_original_floor4_s42: unaccepted/pending; queue=not identified; checkpoint=False.
- step4_amp4: unaccepted/pending; queue=not identified; checkpoint=False.
- step5_inj1950: unaccepted/pending; queue=not identified; checkpoint=False.
- step5_inj1955: unaccepted/pending; queue=not identified; checkpoint=False.
- step5_inj1960: unaccepted/pending; queue=not identified; checkpoint=False.
- step7_allee005: unaccepted/pending; queue=not identified; checkpoint=False.
- step7_allee010: unaccepted/pending; queue=not identified; checkpoint=False.
- step7_allee020: unaccepted/pending; queue=not identified; checkpoint=False.
- step8a_kernel: unaccepted/pending; queue=not identified; checkpoint=False.
- step8b_brake: unaccepted/pending; queue=not identified; checkpoint=False.
