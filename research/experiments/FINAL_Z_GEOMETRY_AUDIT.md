# Multi-decadal Z geometry audit

Feature-only analysis of2791 BBS training cells, excluding encoder holdout and buffer cells. No heldout bird outcomes or House Finch observations were read to fit a metric. Details/source hashes: FINAL_Z_GEOMETRY_AUDIT.json.

The following is RMS feature change divided by spatial standard deviation, using the trace of centered1996–2025 cell-mean covariance. It measures movement in Z under the isotropic kernel; it is **not** a calibrated downstream temporal coefficient ratio.

| Epoch contrast | v1 summer64 | v2 summer64 | v2 winter64 | Full128 |
|---|---:|---:|---:|---:|
|1996–2005 →2016–2025|.212|.208|.235|.221|
|1966–1975 →2006–2015|.420|.421|.425|.423|
|1902–1911 →1932–1941|.251|.248|.224|.237|
|1902–1911 →2006–2015|.626|.618|.705|.662|

Winter64 does not show universally larger multi-decadal movement: its40-year contrast is almost identical to summer64, and its early30-year contrast is smaller. The century-scale contrast is larger and more concentrated: the continental mean accounts for42% of winter change energy, versus26% in summer64; change effective rank is3.59 versus5.11. This is a possible lead for a direction/epoch mismatch, not evidence that the continental trend is spurious or that movement should be suppressed.

The two seasonal halves' spatial traces are.376 and.349; full128 is their sum,.725. Full128 therefore changes the covariance/prior energy as well as feature count. Existing amplitude priors and any future normalization need explicit review; no normalization was silently added to round1.

Together with equal-width winter list-score differences, these findings favor investigating direction, seasonal information and observation calibration rather than assuming excess overall temporal amplitude. Geometry alone cannot identify which movements are ecological signal. A learned metric requires independent development-species change/reliability targets and a fixed validation plan.
