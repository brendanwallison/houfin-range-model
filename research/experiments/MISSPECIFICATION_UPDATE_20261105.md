# TACC connection restored and new results — November 5, 2026

Live check at 04:02 CST: dev allocation 3552642 is running on c301-002, with
three active scientific tasks and about 27 minutes remaining. Logs are advancing.
The lightweight resource watcher has resumed successful collection, including
GPU queries. The local GPU remains unused by Codex.

Scientific queue before design holds: **8 done, 3 running, 21 pending, 4 failed**.
After preserving the three seed-42 shooting repeats in `held_early_design`, the
36 registered tasks are **8 done, 3 running, 18 pending, 4 failed, 3 held**.
The separate summary-recovery task has completed and is outside that count.

Next dev allocation **3556610** was submitted before the current allocation
ends. It is pending under QOSMaxJobsPerUserLimit, so it cannot run alongside the
current dev allocation. No start estimate is available. Existing regular requests
3547292/3548331 remain pending. The development-wave limit remains nine in total;
this submission is within the previously registered maximum of five additional
waves/30 GPU-hours beginning with 3552642. No new budget extension was made.

## What finished since the last scientific review

1. **Arrival redistribution completed.** The repaired diagnostic now passes the
   nonzero full-data likelihood, gradient and replay check, and the fit, contribution
   audit and follow-up all completed. The fitted model itself was not changed by
   the diagnostic repair.
2. **Smaller correlated abundance corrections completed.** This is the .05 log-scale
   arm under reduced historical growth flexibility, complementing the .15 arm.
3. **The occurrence-focused establishment fit's summary was recovered.** Its
   contribution audit still fails negative-flow QC. Summary recovery does not
   make it a fully audited accepted comparison.
4. Both additional first-seed shooting formulations also stalled. All three
   first-seed versions now fail refinement with recent gradient norms around
   1e8–1e9. Their second-seed repeats are held intact, rather than repeating the
   same conditioning problem. This is not evidence against multiple shooting as
   a method; the present log-state continuity formulation needs repair.

The active tasks are the frozen-growth control, smaller establishment correction,
and independent five-year cell abundance corrections. The last two had passed
their nonzero full-data checks; all three had passed their respective checks.

## New arrival result in context

Matched equal-cell support: 25–49N, East >=95W, Plains 100–95W, West <100W.
Historical growth change is 2010–2025 minus 1902–1915. The same .008027860278758616
internal-density threshold and same 782 eligible eastern observation cells are
used for arrival comparisons, rather than a different relative threshold per fit.

| Quantity | Full control | Establishment correction | Arrival redistribution |
|---|---:|---:|---:|
| Growth change, East / Plains / West | 1.246 / .733 / .817 | .568 / .397 / .394 | .597 / .435 / .359 |
| Median modeled arrival minus first detection | −3 yr | −1 yr | −1 yr |
| Eligible cells never reaching threshold | 6.8% | 9.8% | 9.7% |
| Median adult stayer survival, population weighted | .576 | .545 | .582 |
| Median juvenile/adult stayer survival ratio | .447 | .339 | .274 |
| Mean realized reproduction at nests | 3.68 | 4.18 | 4.43 |
| Plains mean immigrant share | .635 | .365 | .310 |
| Plains actual-state local replacement sink area | 96.8% | 90.5% | 92.9% |
| Plains contribution sink area | 78.9% | 89.2% | 90.8% |
| Plains K/Allee nonviability area | .79% | 0% | 0% |

Both mechanisms reduce historical growth changes without fixing those changes
to a smaller value, improve median timing, and retain predominantly sink-like
Plains structure. They require more realized reproduction at nest locations and
worsen raw held-out count scores in all three regions. They also leave more
observed eastern cells below the common arrival threshold. These tradeoffs are
material; neither is an overall winner. Replicates and dynamic isolation remain
outstanding, and no claim of posterior identification follows from these fits.

The smaller abundance correction does not remove the reduced-growth model's
survival tradeoff: median adult survival .373, juvenile/adult ratio .934, with
45% of modeled population above ratio .95. Its historical changes are
.351/.230/.253, versus .308/.219/.244 in its no-correction reduced-growth control.
It remains about four years early at the common threshold. It provides a gentler
comparison to the .15 arm, rather than a demonstrated solution to growth inflation.

Evidence: `MISSPECIFICATION_MATCHED_SUMMARY_20261104.json` (refreshed November 5),
`MISSPECIFICATION_RESULTS_20261104.json` (refreshed November 5),
`MISSPEC_LIVE_LATEST.json`, and `MISSPEC_RESUMED_20261105.json`.
