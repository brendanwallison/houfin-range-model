# First results and execution audit — November 4, 2026

At 15:28 CST, 36 scientific tasks were accounted for: six completed end to end,
three stopped at different stages, and 27 had not started. Seven fits reached an
accepted refinement; one of those stopped in post-fit contribution diagnostics.
Queued tasks were not lost. Dev allocation 3550613 ran 04:36–06:34 (three GPUs),
and small allocation 3548970 ran 07:10–11:09 (one GPU). Both ended normally.
The two regular allocations remained pending. The earlier automatic development
wave limit was exhausted, so there was no running allocation after 11:09.

Continuation is now concrete: replacement dev job **3552642** was submitted and
released to the scheduler. It is pending with no start estimate at this check.
The arrival smoke has been returned to pending after the diagnostic-only fix;
the scientific queue therefore now contains six done, 28 pending (including that
retry), and two failed tasks. The 27 never-started tasks remain among the pending
ones. A separate accepted-fit summary recovery task is also queued.

The watcher can request at most five new two-hour, three-GPU dev waves including
3552642 (30 GPU-hours maximum) for the existing authorized backlog, only while
at least three pending eligible tasks remain. The two existing regular requests
are retained. This avoids stopping again after a single replacement while
keeping a finite continuation budget. Submission receipt:
`MISSPEC_RECOVERY_QUEUE_20261104.json`.

## Matched results, one optimizer seed

All regional comparisons use identical equal-weight land cells between 25–49N,
with East/Plains/West boundaries at 95W and 100W. Historical change is modern
(2010–2025) minus early (1902–1915) potential annual growth. Front comparisons
below use **one common absolute density threshold**, .008027860278758616 internal
units, and the same 782 eastern cells surveyed before their first positive record.
Negative offsets mean modeled arrival precedes the first recorded detection.

| Fit | Growth change East / Plains / West | Median front offset | Never reaches threshold |
|---|---|---:|---:|
| Full-growth control | 1.246 / .733 / .817 | −3 yr | 6.8% |
| Reduced-growth control | .308 / .219 / .244 | −3 yr | 4.3% |
| Reduced growth + abundance corrections | .353 / .258 / .323 | −4 yr | 2.3% |
| Full growth + establishment corrections | .568 / .397 / .394 | −1 yr | 9.8% |
| Occurrence-focused control | .974 / .718 / .488 | −12 yr | 0% |
| Occurrence-focused + abundance corrections | .202 / .195 / .105 | −16 yr | 0% |

The reduced-growth control imposes less temporal variation, so its reduction
cannot itself count as an empirical improvement. It preserves count fit fairly
well, but compensates with low adult survival and near-equal juvenile survival:
population-weighted stayer medians Sa .356, Sj/Sa .944, with 47% of modeled
population above ratio .95. The full control has Sa .576 and ratio .447.

Adding abundance flexibility to reduced growth improves training counts but
worsens held-out raw-count scores in every region. Adult survival rises to .516,
but the juvenile/adult ratio rises to .959 and 52% of population exceeds .95.
Fitted annual log adjustments have RMS .109 East, .172 Plains and .386 West;
these corrections are substantial, unlike the earlier tiny independent terms.
Historical change is slightly greater than the matched reduced-growth control.

The establishment correction is the strongest new mechanism signal: the full
growth model is free to vary historically, yet its historical change roughly
halves in all three regions and the common-threshold front median improves.
Costs remain: more cells never reach that threshold; held-out raw-count scores
worsen modestly in every region; realized reproduction at nest locations rises
from mean 3.68 to 4.18 offspring per breeder, despite lower maximum reproduction
(population-weighted median 8.64 to 6.14). Adult stayer median .545 and juvenile/
adult ratio .339 avoid the near-equal survival compensation seen above.

On identical Plains cells, establishment corrections reduce mean immigrant share
from .635 to .365 and raise mean local replacement from .616 to .689. The Plains
remains dominated by realized sinks: 90.5% of cells below local replacement one,
and 89.2% below contribution one. Positive-growth K/Allee nonviability is zero on
this band, versus .79% in the control. These are actual-state point comparisons,
not posterior identification or proof of a viable biological mechanism. Source
bridge connectivity and dynamic no-immigration experiments have not been rerun.

The occurrence-focused objective improves its targeted in-sample occurrence
component but produces much earlier spread and substantially worse raw-count
prediction. It is not a simple remedy for the wavefront/count imbalance. This
does not invalidate other occurrence/range likelihoods or historical supervision;
this particular .25 positive-count weighting permits major compensations.

## Three different failures

- `shoot10_weak_s41`: refinement invalid/stalled, recent median gradient about
  2.49 billion. This is an optimization/conditioning failure, not a fitted
  biological verdict on multiple shooting. Its checkpoints remain stored.
- `arrival10_full_s41`: failed the nonzero full-data accounting smoke before
  fitting. Replay computed density-dependent disease after arrival reweighting,
  whereas the forward applies that reweighting afterward. A diagnostic-only
  correction restores the order; the full-data gate must pass before fitting.
- `hurdle_est_s41`: fitting and refinement completed; contribution diagnostics
  rejected a negative immigrant-breeder flow. The magnitude and cause still need
  examination. The QC gate is retained. Independent summary recovery is queued
  so the accepted fit is not discarded because this diagnostic stopped.

The disease-order diagnostic amendment preserves model code and checkpoints and
archives the prior audit implementation. It does not waive the negative-flow QC.
Continuation allocation requests and requeue actions are recorded separately in
`MISSPEC_RECOVERY_QUEUE_20261104.json` when submission succeeds.

Evidence: `MISSPECIFICATION_RESULTS_20261104.json` and
`MISSPECIFICATION_MATCHED_SUMMARY_20261104.json`. No local GPU was used. These
first results do not establish an overall winning model; optimizer replication,
paired predictive uncertainty, effective establishment trajectories, permanent
movement, dynamic isolation and HMC geometry remain outstanding.
