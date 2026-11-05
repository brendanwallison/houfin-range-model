# Clean candidate interpretation

Updated 2026-10-25 17:36 CDT. Matched seeds1/2; all observations were previously opened.

Full-width v2 improves full-calibration summer change correlation relative to v1: within +0.026 [0.008,0.043], pooled +0.051 [0.028,0.071]. Equal-total-width32+32 differences are +0.014 [-0.009,0.034] within and -0.002 [-0.020,0.014] pooled; these intervals include zero. Sparse results remain sensitive to calibration realization.

Winter outcomes below compare each candidate's backcast-minus-modern-feature gain to v1's corresponding gain. They measure conditional recorded list composition, not abundance truth. Positive favors v2.

| Winter era | Lists / circles | Full-width v2 minus v1 temporal gain [95%] |
|---|---|---|
|1902_1919|74/35|-0.195 [-0.331,-0.057]|
|1920_1939|168/52|-0.155 [-0.214,-0.104]|
|1940_1952|204/50|-0.050 [-0.096,-0.002]|
|1953_1965|382/68|-0.009 [-0.044,+0.029]|
|1966_1985|1257/109|-0.004 [-0.020,+0.016]|
|1986_1995|876/113|+0.001 [-0.009,+0.010]|
|1996_2004|910/113|+0.006 [-0.001,+0.012]|
|2005_2024|2372/146|-0.001 [-0.002,0.000]|

The early winter deficit remains after clean completeness training. Later absolute backcast advantages can reflect better modern composition rather than stronger temporal information: for1966–1985 full-width v2's absolute advantage is +0.020 [0.003,0.036], while its temporal-gain contrast is -0.004 [-0.020,0.016]. Earliest same-circle subsets are sparse; retain that limitation rather than interpreting pooled early comparisons as reliable population recovery.

Recommendation: retain clean v1 and full-width v2 as separate candidates while completing comparable House Finch refinement and ecological checks. Full-width v2 has the stronger well-calibrated summer result; v1 remains an essential historical winter comparator. A two-seed diagnostic is insufficient for a final four-seed production adoption. The reused holdouts provide exploratory evidence, not fresh confirmation.

Artifacts: FINAL_CLEAN_CANDIDATES.json records paired summer results and primary, same-circle and low-counter winter contrasts. FINAL_CLEAN_RESULTS.md/.json quantify the matched completeness intervention. No readout or training parameter was tuned from these comparisons.
