# Sparse calibration replication — 2026-10-25

All six additional scores (thinning seeds3/4 × three representations) completed successfully at13:28 CDT. Combined with seeds1/2, these reuse the existing noisy-observation scorer, target eligibility, spatial holdouts, ABBA truth and paired6×6 block bootstrap. Encoder features are the original four-seed means. These remain sensitivities on previously opened holdouts, with the original CBC completeness leakage.

3% of modern target-calibration rows; early1966–1986, modern2005–2025. Differences below are v2(64summer+64winter) minus v1(64summer), in noise-corrected change correlation.

| Thinning seed | Within-cell difference [95% interval] | Pooled difference [95% interval] |
|---|---|---|
|1|−.004 [−.025,.018]|+.021 [.006,.035]|
|2|−.041 [−.062,−.017]|+.021 [.004,.039]|
|3|−.007 [−.026,.011]|+.029 [.012,.046]|
|4|−.004 [−.032,.025]|+.027 [.010,.044]|

The pooled advantage replicates in all four thinning seeds with positive paired intervals. Full128 has negative within-cell point differences in all four; only seed2's interval excludes zero. This does not support transferring its full-data within-cell advantage to sparse calibration. The repeats reuse the same heldout truth and are not four independent ecological replications; do not multiply their significance.

At equal total width (summer32+winter32), within differences are+.023,+.002,+.015,+.033; only seed4's interval excludes zero. Pooled differences are−.012,−.002,+.003,+.014, all intervals crossing zero. Thus equal width behaves differently from simply adding64winter coordinates.

All paired comparisons check identical target columns and truth-bootstrap components. Numerical estimates and full intervals: `/home/breallis/houfin/work/houfin/research/FINAL_HOLDOUT_20261025/SPARSE_FOLLOWUP_SUMMARY.json`; additional raw results: `sparse_followup/`. This is readout/representation evidence, not a recommendation to change House Finch priors before its numerical audit is resolved.

A workspace copy of the numerical summary is FINAL_SPARSE_REPLICATION_20261025.json.
