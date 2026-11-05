# Retrospective pathology checklist — cached diagnostics

Root: /mnt/e/Datasets/houfin/grid54/runs. 111 run directories; 81 checkpoints; 78 accepted refinements.

Historical fits use different habitat inputs, life cycles and observation models. This table is an inventory of tradeoffs, not a pooled ranking. Lambda change is not itself a pathology. Tiny K in unoccupied cells also requires map/observation context. Synthetic-observation runs identified in the logs are retained in JSON but excluded from this real-data table.

| Run | Yearling breeding | Lambda change | Suitable share early → modern | Modern K minimum | Allee-dead share of land | Temporal / spatial log score |
|---|---|---:|---|---:|---:|---|
| step0_s64 | False | +0.414 | 0.139 → 0.415 | 1.3e-10 | 0.072 | -1.832 / -1.230 |
| step10_pn05 | False | +0.467 | 0.116 → 0.485 | 0.0023 | 0.000 | -1.911 / -1.663 |
| step10_pn10 | False | +0.680 | 0.084 → 0.465 | 0.00045 | 0.001 | -1.862 / -1.520 |
| step10_pn20 | False | +0.452 | 0.060 → 0.519 | 0.00033 | 0.000 | -1.885 / -1.515 |
| step11_reset1966 | False | +0.784 | 0.091 → 0.469 | 0.00037 | 0.001 | -1.845 / -1.492 |
| step11_reset3 | False | +0.452 | 0.081 → 0.478 | 0.001 | 0.001 | -1.853 / -1.540 |
| step12_lam10 | False | +0.562 | 0.119 → 0.465 | 0.0015 | 0.001 | -1.901 / -1.662 |
| step12_lam10_s42 | False | +0.658 | 0.095 → 0.441 | 0.00055 | 0.000 | -1.844 / -1.509 |
| step12_lam10_s43 | False | +0.662 | 0.094 → 0.442 | 0.00056 | 0.000 | -1.842 / -1.508 |
| step13_obsreg | False | +0.471 | 0.157 → 0.494 | 0.0044 | 0.000 | -1.916 / -1.650 |
| step14b_dispera | False | +0.554 | 0.097 → 0.406 | 0.00053 | 0.000 | -1.842 / -1.485 |
| step15_lowdens | False | +0.244 | 0.126 → 0.523 | 0.00014 | 0.000 | -1.840 / -1.514 |
| step16_sthab | False | +0.643 | 0.106 → 0.434 | 0.00018 | 0.000 | -1.840 / -1.495 |
| step17_r64_free | False | +0.775 | 0.093 → 0.471 | 0.0004 | 0.001 | -1.845 / -1.492 |
| step17_r64_freepen | False | +0.744 | 0.103 → 0.458 | 0.00029 | 0.000 | -1.843 / -1.490 |
| step17_r64_regional | False | +0.741 | 0.104 → 0.458 | 0.00029 | 0.001 | -1.843 / -1.490 |
| step17_r64_regpen | False | +0.746 | 0.104 → 0.458 | 0.00029 | 0.000 | -1.843 / -1.490 |
| step18_warp64pen | False | +0.740 | 0.104 → 0.459 | 0.00037 | 0.000 | -1.843 / -1.491 |
| step19b_anchor64 | False | +0.734 | 0.101 → 0.455 | 0.00018 | 0.000 | -1.845 / -1.490 |
| step19b_anchor64pen | False | +0.769 | 0.100 → 0.458 | 0.00033 | 0.000 | -1.845 / -1.493 |
| step1_relallee | False | +0.591 | 0.055 → 0.313 | 2.6e-10 | 0.000 | -1.829 / -1.194 |
| step20_jboth | False | +0.368 | 0.142 → 0.548 | 0.00071 | 0.000 | -1.841 / -1.503 |
| step20_jcost | False | +0.744 | 0.104 → 0.457 | 0.0003 | 0.000 | -1.843 / -1.492 |
| step20_jdec | False | +0.358 | 0.127 → 0.508 | 0.00022 | 0.001 | -1.846 / -1.516 |
| step21_j1 | False | +0.744 | 0.104 → 0.457 | 0.0003 | 0.001 | -1.843 / -1.492 |
| step22_b44 | False | +0.748 | 0.101 → 0.458 | 0.00055 | 0.000 | -1.845 / -1.494 |
| step22_j2 | False | +0.748 | 0.103 → 0.458 | 0.00029 | 0.001 | -1.843 / -1.490 |
| step24_attr | False | +0.698 | 0.107 → 0.475 | 0.0003 | 0.001 | -1.840 / -1.465 |
| step25_natdep | False | +0.761 | 0.115 → 0.472 | 0.00014 | 0.001 | -1.842 / -1.496 |
| step26_mdd110 | False | +0.689 | 0.255 → 0.550 | 2.2e-10 | 0.000 | -1.854 / -1.479 |
| step26_mdd200 | False | +0.786 | 0.113 → 0.464 | 1e-06 | 0.001 | -1.854 / -1.512 |
| step29_yb_native | True | +1.345 | 0.101 → 0.443 | 2.3e-10 | 0.000 | -1.786 / -1.424 |
| step29_yb_native_s42 | True | +1.341 | 0.101 → 0.442 | 2.1e-10 | 0.001 | -1.786 / -1.424 |
| step2_single | False | +0.763 | 0.094 → 0.419 | 1.9e-09 | 0.001 | -1.866 / -1.206 |
| step3_native | False | +0.853 | 0.136 → 0.498 | 7.4e-08 | 0.000 | -1.790 / -1.410 |
| step3_native_s42 | False | +0.891 | 0.151 → 0.500 | 6.8e-10 | 0.000 | -1.791 / -1.400 |
| step3_native_v2 | False | +0.824 | 0.147 → 0.501 | 2.4e-08 | 0.000 | -1.791 / -1.433 |
| step3_native_v2_s42 | False | +0.851 | 0.164 → 0.496 | 2e-10 | 0.000 | -1.788 / -1.405 |
| step3_obsroute | False | +0.743 | 0.104 → 0.456 | 0.00029 | 0.001 | -1.844 / -1.493 |
| step3_seed42 | False | +0.742 | 0.103 → 0.465 | 0.00025 | 0.000 | -1.843 / -1.490 |
| step3_seed43 | False | +0.766 | 0.101 → 0.467 | 0.00036 | 0.000 | -1.844 / -1.494 |
| step4_amp1 | False | +0.579 | 0.096 → 0.435 | 0.0077 | 0.000 | -1.876 / -1.602 |
| step4_amp2 | False | +0.733 | 0.089 → 0.441 | 0.0029 | 0.001 | -1.876 / -1.581 |
| step4_amp4 | False | +0.810 | 0.089 → 0.449 | 0.0015 | 0.000 | -1.878 / -1.573 |
| step4_amp8 | False | +0.572 | 0.197 → 0.455 | 8.3e-06 | 0.000 | -1.880 / -1.613 |
| step5_inj1960 | False | +0.721 | 0.096 → 0.450 | 0.00021 | 0.000 | -1.845 / -1.501 |
| step6_fdi_inj1960 | False | +0.481 | 0.198 → 0.427 | 3.2e-05 | 0.000 | -1.853 / -1.543 |
| step6_fdi_step3 | False | +0.508 | 0.212 → 0.421 | 8.6e-07 | 0.000 | -1.867 / -1.574 |
| step7_allee005 | False | +0.636 | 0.136 → 0.518 | 0.0035 | 0.005 | -1.892 / -1.636 |
| step7_allee010 | False | +0.752 | 0.156 → 0.518 | 2.5e-05 | 0.018 | -1.889 / -1.624 |
| step7_allee020 | False | +0.551 | 0.436 → 0.615 | 2e-06 | 0.055 | -1.915 / -1.786 |
| step8a_kernel | False | +0.537 | 0.096 → 0.404 | 0.00035 | 0.001 | -1.841 / -1.484 |
| step8a_kernel_s42 | False | +0.550 | 0.097 → 0.408 | 0.0005 | 0.000 | -1.841 / -1.484 |
| step8a_native | False | +0.716 | 0.112 → 0.453 | 6.6e-10 | 0.000 | -1.786 / -1.429 |
| step8a_native_v2 | False | +0.686 | 0.124 → 0.451 | 4.5e-10 | 0.001 | -1.781 / -1.420 |
| step8b2_brake | False | +0.770 | 0.187 → 0.556 | 0.003 | 0.018 | -1.873 / -1.597 |
| step8b_brake | False | +0.677 | 0.195 → 0.504 | 0.00032 | 0.015 | -1.884 / -1.609 |
| step9_fdi_kernel | False | +0.435 | 0.174 → 0.382 | 3.4e-06 | 0.000 | -1.853 / -1.540 |
