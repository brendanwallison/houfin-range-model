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
