Phase 0 (gate 0 pending) with Phase 1 experiments running on cached arrays. Data landed and verified 2026-10-09.
GATE 0 DUE when E001 base + tempho1995 finish (~08:59 / ~10:15): compare_reports verdicts + timings to the user,
with the morning's (unverified) results as a preview.
Live questions (UNVERIFIED leads, see DIGEST 2026-10-09 morning):
- B7: is the readout's level-fitted beta the main loss of change? E011 x4 queued (1975/1985 pre-registered).
  If yes -> E012: does a temporal beta learned INSIDE the trained era transfer to withheld decades?
- D0: DESK backcast change ~ observed community change for the species readout (half-window comparator is
  noisy -- needs an errors-in-variables-corrected comparator before it can conclude).
- E1/readout ceiling by prevalence: E009 (planted v2) running.
Verification debt (none of the above is a FINDING yet): E004 atlas (b) re-derivation, (c) second split,
(d) skeptic; E011 the same once the jobs land.
Then: E010 DESK reproduction (GPU, after E006) = Phase 0.4 gate.
Cold copy to E: (the nohup'd rsyncs died with their session): pueue group "io", tasks 36/37 (not in the registry).
When done: cd /mnt/e/Datasets/houfin/work_houfin && sha256sum -c --quiet processed_xfer.sha256 (same for data).
TACC baseline (old suite, A1-diluted): out-of-community primary change skill base +0.0002 (401 spp),
tempho1975 -0.0007, 1985 -0.0037, 1995 -0.0100; community mode +0.0086/+0.0108/+0.0017/-0.0036 (~91 spp).
2026-10-09 (user's over/under-movement speculations -> S1-S3): the target becomes an HONEST backcast (accurate where
possible, uncertainty the readout carries), not minimal or maximal movement. Proposed next, cheap, on cached arrays,
before any DESK training: E017 where DESK's backcast errors point (analog direction, false neighbours; S2); E018 does the
readout know when it guesses (leverage vs error, habitat-level coverage, covariate novelty -> z novelty; S1/T9); E019
backcast error vs distance from the training years (tempho 1975/1985/1995; input to an errors-in-features variance).
Shrinkage (T10) no longer counts as a fix by itself. Proposed rung 5 for the user's go: age-model MAP under 2-3
plausible backcasts (DESK, no-change, within-cell surrogate) -- how far do the vital rates move? Awaiting the user's go.
2026-10-09 evening: E006 (corrected suite) failed on a theta_bounds bug (fixed, re-enqueued behind E010/E022). Its
spacetime_sum shape fit also took 51 min and ended ABNORMAL with ls_change at the floor (a degenerate change
component) -- inspect report.json baseline_shapes when the re-run lands; the honest temporal rival may need a
better-posed fit (A5b). GPU order: E010 r1/r2 (DESK reproduction gate) -> E022 x4 seeds (only if E010 r1 passes) ->
E006 base/tempho1995. CPU: E019b after the seeds; E018b/c, E021 replications.
2026-10-09 late (user): the loop had drifted into mitigation mode and treated the age-model sensitivity fit as next.
Re-oriented: the program is iterative exploration of what limits extrapolation in time and what would extend it.
Mitigations (recalibration, static habitat + reach-dependent uncertainty) are fallbacks, not the plan; the rung-5 fit
comes last. Next: E023 span x reach (rung 1: does more training history help at a FIXED reach? the tempho models
confound span and reach), then lags (S3), temporal supervision in DESK (T1), and the scoreable target (coherent,
observer-adjusted change).
2026-10-09 ~14:30: E010 r1 PASSED (Phase 0.4); r2 segfaulted at epoch 362 (bcba64c8, no resume; watch for recurrence).
E024 redesigned (DESK cannot withhold its label year 2025 -> "span + 2025"); GPU order: E022 seed (running) -> E024 x3
(priority) -> E022 x3 -> E006 x2. E023b analyses the span runs (+2025 surrogates for comparison) when they land.
2026-10-09 ~16:00: GPU memory fixed -- earlier DESK runs reserved 25.6 GB (log "VRAM a/b" = max allocated / reserved)
and spilled ~4.3 GB into shared system memory; with expandable_segments (paths.job_env, GPU group) the E024 1976-1995 run
reserves 16.2 GB, no spill, ~6 s/epoch as before. True-span smoke passed (2025 withheld; diagnostic 1976->1995).
Two void span+2025 runs used the GPU before cancel was fixed: E024/desk-span-plus2025-1976-1995 finished (kept, registry
says cancelled), 1986-2005 was killed part-way.
2026-10-09 ~21:45 TENTATIVE PLAN after E024 1996-2015 (agreed with the user; decision rules fixed before results):
Tonight (automatic): E024 1996-2015 resume -> E023b span analysis vs the current-code long-span seeds; backcast re-runs
(E013/E017/E018/E020) on the seed caches; E006 x2.
Tomorrow, branch on:
 1. E024 rule (short span beats same-T0 long-span seeds by > seed spread in >= 2 of 3): yes -> time-weighted DESK (all
    years, earliest decades up-weighted via per-cell-year weights; 2 weightings x 2 seeds, tempho1995 setup); borderline ->
    one more seed per span model first; no -> N2 stays a surrogate-level result, GPU to branch 2.
 2. Why current-code retrains backcast ~50% better than the archived tempho models (e.g. 1966-71 from 1996: 0.12 ->
    0.20/0.18; archived models record no selected epoch): diff their recorded settings vs HEAD and the code history since
    2026-08-21 (A9 selection, moved defaults); per-epoch withheld-year diagnostics in the training logs; if epoch
    selection drives it, test selecting on a trained decade held back from selection only.
 3. Re-establish the DESK-failure picture on current-code models (space-for-time pull, readout corr, over-movement).
Then: lags (multi-timescale covariate history vs the trend placebo), temporal supervision in DESK, observer-continuity
target, and the Gate 1 report once E006 + the re-runs are in.
2026-10-09 ~22:55 (overnight, user asleep; plan followed/revised per their go-ahead):
- E024 decided by its pre-registered rule: NO (1966-71: 1996 short 0.102 vs long seeds 0.122/0.105; 1986 0.143 vs
  0.136/0.112, within the 0.024 spread; 1976 below the archived model). Short spans win only at 1972-77. N2 stays a
  surrogate-level result; time-weighted DESK NOT launched.
- Branch 2 started. Training logs (rung 0): after the selected epoch (65-162), the withheld pair's direction cosine is
  flat-to-rising through epoch 490 while rotation doubles -> E025 grades the epoch-490 states (caches from
  resume_checkpoint.pt.completed). Recorded difference archived vs HEAD: metric_pairs 4,096 -> 65,536 (a3633d2, Aug 25)
  -> E026 tempho1995 at 4,096 and 262,144 (T11: optimization-limited temporal learning).
- GPU order: E006 base (running) -> E025 late caches -> E026 mp4096 -> mp262144. CPU analyses chained.
- Next decision after E025/E026: if late > selected or mp262k > 65k beyond the seed spread -> replicate (2nd seed, and
  tempho1985) and test more optimizer steps per epoch (spatial_tiles: DESK takes ONE step per epoch, 500 in all).
  If null -> lags (S3) and temporal supervision (T1) are next.
