# Retrospective pathology checklist — cached diagnostics

Root: /work/07980/bwa386/ls6/houfin/research/grid54/runs. 96 run directories; 70 checkpoints; 69 accepted refinements.

Historical fits use different habitat inputs, life cycles and observation models. This table is an inventory of tradeoffs, not a pooled ranking. Lambda change is not itself a pathology. Tiny K in unoccupied cells also requires map/observation context. Synthetic-observation runs identified in the logs are retained in JSON but excluded from this real-data table.

| Run | Yearling breeding | Lambda change | Suitable share early → modern | Modern K minimum | Allee-dead share of land | Temporal / spatial log score |
|---|---|---:|---|---:|---:|---|
| step10_pn05 | False | +0.467 | 0.116 → 0.485 | 0.0023 | 0.000 | -1.911 / -1.663 |
| step10_pn10 | False | +0.680 | 0.084 → 0.465 | 0.00045 | 0.001 | -1.862 / -1.520 |
| step10_pn20 | False | +0.452 | 0.060 → 0.519 | 0.00033 | 0.000 | -1.885 / -1.515 |
| step12_lam10 | False | +0.562 | 0.119 → 0.465 | 0.0015 | 0.001 | -1.901 / -1.662 |
| step12_lam10_s42 | False | +0.658 | 0.095 → 0.441 | 0.00055 | 0.000 | -1.844 / -1.509 |
| step12_lam10_s43 | False | +0.662 | 0.094 → 0.442 | 0.00056 | 0.000 | -1.842 / -1.508 |
| step13_obsreg | False | +0.471 | 0.157 → 0.494 | 0.0044 | 0.000 | -1.916 / -1.650 |
| step14b_dispera | False | +0.554 | 0.097 → 0.406 | 0.00053 | 0.000 | -1.842 / -1.485 |
| step15_lowdens | False | +0.244 | 0.126 → 0.523 | 0.00014 | 0.000 | -1.840 / -1.514 |
| step16_sthab | False | +0.643 | 0.106 → 0.434 | 0.00018 | 0.000 | -1.840 / -1.495 |
| step17_r64_regpen | False | +0.743 | 0.104 → 0.457 | 0.0003 | 0.000 | -1.843 / -1.490 |
| step17_r64_total | False | +0.738 | 0.103 → 0.464 | 0.00032 | 0.001 | -1.843 / -1.490 |
| step19b_anchor64 | False | +0.734 | 0.101 → 0.455 | 0.00018 | 0.000 | -1.845 / -1.490 |
| step19b_anchor64pen | False | +0.769 | 0.100 → 0.458 | 0.00033 | 0.000 | -1.845 / -1.493 |
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
| step27_dirch | False | +0.656 | 0.116 → 0.507 | 0.0003 | 0.002 | -1.836 / -1.471 |
| step29_yb_native | True | +1.345 | 0.101 → 0.443 | 2.3e-10 | 0.000 | -1.786 / -1.424 |
| step29_yb_native_s42 | True | +1.341 | 0.101 → 0.442 | 2.1e-10 | 0.001 | -1.786 / -1.424 |
| step30_comp | False | +0.745 | 0.103 → 0.459 | 0.00025 | 0.000 | -1.842 / -1.489 |
| step31_nest | True | +1.168 | 0.111 → 0.451 | 2.2e-10 | 0.000 | -1.781 / -1.431 |
| step31_nest_bloose | True | +1.234 | 0.108 → 0.450 | 1e-09 | 0.001 | -1.780 / -1.429 |
| step31_nestdrop | True | +1.457 | 0.107 → 0.447 | 3.7e-10 | 0.001 | -1.484 / -0.940 |
| step31_nestonly | True | +1.178 | 0.110 → 0.454 | 4e-10 | 0.001 | -1.781 / -1.431 |
| step31_nestsub | True | +1.346 | 0.106 → 0.442 | 5e-11 | 0.001 | -1.486 / -0.938 |
| step32_cbc_native | False | +0.863 | 0.131 → 0.507 | 2.5e-08 | 0.000 | -1.794 / -1.438 |
| step32_cbc_yb | True | +1.177 | 0.103 → 0.446 | 2e-08 | 0.000 | -1.777 / -1.408 |
| step32_cbchi_yb | True | +0.955 | 0.098 → 0.430 | 3.4e-07 | 0.001 | -1.800 / -1.453 |
| step32_cbclo_yb | True | +1.152 | 0.103 → 0.446 | 3e-08 | 0.000 | -1.776 / -1.408 |
| step3_native | False | +0.853 | 0.136 → 0.498 | 7.4e-08 | 0.000 | -1.790 / -1.410 |
| step3_native_s42 | False | +0.891 | 0.151 → 0.500 | 6.8e-10 | 0.000 | -1.791 / -1.400 |
| step3_native_v2 | False | +0.824 | 0.147 → 0.501 | 2.4e-08 | 0.000 | -1.791 / -1.433 |
| step3_native_v2_s42 | False | +0.851 | 0.164 → 0.496 | 2e-10 | 0.000 | -1.788 / -1.405 |
| step3_seed43 | False | +0.766 | 0.101 → 0.467 | 0.00036 | 0.000 | -1.844 / -1.494 |
| step4_amp8 | False | +0.572 | 0.197 → 0.455 | 8.3e-06 | 0.000 | -1.880 / -1.613 |
| step8a_kernel_s42 | False | +0.550 | 0.097 → 0.408 | 0.0005 | 0.000 | -1.841 / -1.484 |
| step8a_native | False | +0.716 | 0.112 → 0.453 | 6.6e-10 | 0.000 | -1.786 / -1.429 |
| step8a_native_v2 | False | +0.686 | 0.124 → 0.451 | 4.5e-10 | 0.001 | -1.781 / -1.420 |
| step9_seed42 | False | +0.434 | 0.175 → 0.383 | 6.2e-06 | 0.001 | -1.855 / -1.545 |
