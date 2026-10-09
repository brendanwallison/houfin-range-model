# DIGEST

## CURRENT STATE (rewritten each round; read this first -- the dated log below keeps the history)
Last updated 2026-10-09 evening, after the third skeptic round. "Solid" = independently re-derived, replicated on
a second split or seed, and survived a skeptic review. "Tentative" = one run, or checks still pending.

SOLID
- The local machine reproduces TACC for DESK, the spacetime GP, no-change and the oracle. The covariate-GP baseline
  does NOT reproduce (fragile optimizer; defect A13) and is not relied on.
- DESK's temporal change on held-out cells correlates 0.41 with the true community change in trained years and
  0.21 in withheld decades, and it moves more than that accuracy justifies (a calibrated DESK would be 0.62x /
  0.37x as large). In absolute terms it moves less than the communities did; the problem is direction.
- Read through the age model's readout, DESK's change into withheld decades predicts species change worse than
  assuming no change.
- ESK (64 dims) keeps the regionally shared part of a 40-year change about as well as comparable spatial
  structure; it drops route-level idiosyncrasy, temporal or spatial. Truncation is not selectively discarding time.

TENTATIVE (skeptic-reviewed; replications or seeds still pending)
- The pattern across the 1975 / 1985 / 1995 models (1-10, 11-20, 21-30 years of reach) is one dose-response: as
  DESK reaches further back, its useful temporal signal fades and its movement increasingly follows space-for-time.
  - DESK's backcast moves along the space-for-time direction (your point 2): alignment +0.02 / +0.16 / +0.21 (the
    explicit space-for-time map: ~+0.2 everywhere); real communities stay at ~0 or slightly against it.
  - DESK's change read through the age-model readout: correlation with species change 0.13 / 0.07 / 0.06. The
    observed community's change, read the same way, gives ~0.24, but a third or more of that is observers moving
    community and species together; adjusted, DESK reaches about half of it in withheld decades and all of it in
    trained years.
  - Beyond regional trends, DESK's temporal deviations add +0.02-0.04 of species change skill at 1-20 years of reach
    and nothing at 21-30.
- Space-for-time fails at the covariate stage (robust in all four caches): coefficients learned by comparing places
  at a fixed time predict none of the decadal community change. What does predict it is REGIONAL TREND PERSISTENCE:
  a placebo of smooth position fields x year, with no covariates, matches or beats a within-cell covariate map, and
  the covariates add only +0.03-0.05 beyond it. Raw DESK sits below that placebo everywhere; DESK recalibrated on its
  trained years reaches it, and is then well calibrated.
- DESK's backcast degrades steadily with distance from its training years: for the same target decade its
  correlation falls 0.42 -> 0.12 and calibration 0.58 -> 0.20 between 0 and 28 years of reach. It is at most ~10%
  better than no change even in trained years, and 10-20% worse beyond ~15 years. Two more seeds per model queued.
- The readout cannot tell when it is guessing: its change intervals are ~2x too narrow (sd) beyond what the
  instrument itself produces, in trained years too, and leverage does not flag the errors.
- About 30% of what every change metric counts as "true change" is observer turnover (81% of cells share no observer
  between epochs). No habitat model should predict it; every ceiling and correlation here is diluted by it.
- Rank 64 would add little: DESK carries almost nothing in components 25-64. (Half of the measured temporal change
  sits there, but that share matches the footprint of any diffuse perturbation, so it is not evidence that the
  24-dim cut discards ecological change.)

OVERTURNED (kept so the corrections are visible)
- "A change-fitted readout captures 25-29% of species change from community change": spatial interpolation;
  position alone does as well.
- "The true community's change, read through spatial betas, is worse than no change": errors-in-variables
  artifact; it is about zero.
- "The split kernel works through DESK's temporal information": a placebo with no DESK content does as well.
- "ESK keeps 23% of temporal change because time is small": confounded by a between-route nugget and by
  survey effort; superseded by the route-level measurement above.
- "DESK's backcast errors are not pulled toward space-for-time analogs" (my first analog test): its direction was
  ~80% the cell's own quirks; the corrected test shows they are.
- "Intervals ~3x too narrow": ~2x on a stable floor.
- "Within-cell covariate coefficients transfer to the past, so train DESK on within-cell variation": the within map
  is a regional trend extrapolator (a position x year placebo does as well); its better calibration was the
  recalibration step, which works on DESK too. Withdrawn as a DESK-side lever.

WHAT IT MEANS FOR 1902-1939 (working view)
- The target is an honest backcast, not maximal or minimal movement: DESK's pre-BBS deviations probably cost more
  than they carry unless heavily shrunk, and the readout needs an explicit error term that it does not have today.
- Because DESK's backcast error is partly SYSTEMATIC (toward climate-analog places), it will not average out
  regionally: 1902-1939 habitat would lean toward the habitat of places whose present climate resembles the past
  -- the over-similarity bias you were worried about.
- Little of the decadal change we can score is covariate-driven beyond regional trends (the drivers are mostly not
  in the covariates, or act on lags the covariates cannot carry -- your point 3). Every candidate backcast for
  1902-1939 (DESK, a within-cell map, trend persistence) amounts to extrapolating 1966-2025 trends backward.
- Cheap and supported: recalibrate DESK's temporal deviations on its trained years before the readout sees them.
- Next decisive check (needs your go): fit the age model under two or three plausible backcasts and see how far
  the inferred vital rates move.

---- dated log (newest first) ----

## 2026-10-09 ~09:40: GATE 0 (reproduction and timing)
- Reproduction: DESK, the spacetime GP, no-change and the oracle reproduce TACC's reports (base 76/77 medians
  SAME or within CI; tempho1995 173/198, every miss in the covariate-GP baseline). The covariate GP's ARD
  shape fit is fragile across machines (different optimizer terminations; ~0.04 level skill) -- new defect
  A13. The three concerns are reproducible; DESK-vs-covariate-GP is not, until A13 is fixed.
- Timing: full suite 38 / 58 min locally vs ~3 h on TACC; a research cache ~60 s; readout arms 3-9 s;
  route-level variogram 18 s; DESK training reproduction (E010) queued behind the corrected suite.
- R1 measured directly (E014, replicated): ESK keeps the regionally coherent part of a 40-y change as well
  as structured spatial differences; it drops route-level idiosyncrasy. Strong form not supported.
- E016 (development run): covariate->community coefficients learned ACROSS space predict none of the
  temporal community change; within-cell coefficients predict it as well as DESK, better calibrated.

## 2026-10-09 ~09:00: the skeptic's review (verification step d) overturns three of the morning's readings
- R1 is OPEN again. Adjacent-cell pairs are different routes; temporal pairs mostly the same route. The
  spatial curve carries a between-route nugget (fits: 35% of the lag-1 difference, up to 70%) that ESK
  discards. Against the structured spatial part, temporal retention may be LOWER (your hypothesis's strong
  form), and the 40-y change is ~100 km of structured turnover, not "< 27 km". Next: measure the nugget
  directly from route-level counts (N1). The retention numbers also depend on years per half (fixing).
- "Level readout of community change is worse than no change": an errors-in-variables artifact for the
  TRUE community (it is ~0, not negative). Only DESK's change, read through level betas, harms the
  backcast -- that is DESK backcast error. E013 says the same with noise-free correlations: 0.23 (true
  community) vs 0.06 (DESK, inside the planted null band).
- The split kernel's gain is NOT shrinkage, and a placebo deviation block with no DESK content (regional
  position x linear time) matches it. So DESK's temporal deviations may add nothing; regional trends may
  be the whole effect (B8). Testing with a paired placebo arm.
- DESK's atlas correlations were uncentered; centered: 0.41 (trained), 0.21 (withheld decades).
- Robust and now the clearest result: DESK over-moves for its information (a calibrated DESK would be
  shrunk to 0.62x / 0.37x), and the gap to a perfect readout is ~85-90% DESK's backcast, not the readout.

## 2026-10-09 morning: first real-data results (ALL UNVERIFIED -- they steer, they do not conclude)
- Size, not direction (R1, E004): a cell's 40-year community change (1966-86 -> 2005-25) is smaller than the
  difference between adjacent 27-km cells. ESK r24 keeps 23% of it, and the same 23% of adjacent-cell
  differences; r64 keeps 39% of both. Truncation drops small differences, temporal or spatial alike.
- DESK (held-out cells, r24): its temporal change correlates 0.49 with the truth (slope 0.36) in trained
  years and 0.30 (slope 0.20) into withheld decades (tempho1995). Small spatial differences fare no better
  (adjacent cells: slope 0.18, corr 0.30). Given its correlation, DESK moves MORE than a calibrated
  predictor would, so "under-moves 2-4x" looks like the wrong diagnosis; the limit is information.
- The age-model readout (BLR, raw z r24, iid prior; E005) captures ~0 of resolvable dev-species change, and
  into withheld decades is significantly worse than no change (-0.016 [-0.029, -0.008]).
- The readout FORM looks like the larger loss (E011 development run): a beta fitted on LEVELS captures
  4-10% of available species change (negative into withheld decades); a beta fitted on CHANGE captures
  25-29% from the same features, and DESK's backcast change does as well as the observed community's. New
  hypotheses B7 (space-for-time at the readout) and D0 (DESK's information is not the first bottleneck).
  CORRECTED later the same morning (placebo arms): position alone captures 0.30 and the static community
  0.33, so the change-fitted 25-29% is spatial interpolation of where species changed, not species
  following community change. Survives (verified (b)+(c)): the level readout of community change is worse
  than no change into withheld decades, while for planted species whose truth is linear in z it
  recovers 98% -- the failure is the real species' space-for-time mismatch, not the instrument.
- Split kernel (E012): a separate deviation block with a long reference window turns the common-species
  backcast from worse than no change (-0.02/-0.04) to modestly better (+0.02/+0.04, CIs exclude 0), place-
  specific. The data give the deviation block a tenth of the level block's amplitude: it works mostly by
  NOT forcing spatial betas onto temporal change. A short (2011-2025) reference window loses the gain.
- Route turnover inflates the change budget by ~3% (dev species), so it is not a major factor (O3).
- Planted pilot: even a perfect-form readout shows change attenuation 0.2-0.75 depending on prevalence;
  below 10% prevalence the true change is <3% of the observed change variance (E1). E009 adds power.
- Queued: E009 planted v2, E011 change oracle x4, E010 DESK reproduction x2 (bcba64c8, ~100 min each).

## 2026-10-09 (data landed)
- TACC artifacts transferred and verified: 880 processed files (27 GB on TACC; history_vectors excluded)
  and 93 raw files (BBS 2026 release, AVONET, masks), every one matching its TACC sha256.
- The concerning TACC numbers, for the record (old suite, A1-diluted): out-of-community DESK-vs-no-change
  change skill +0.0002 (base), -0.0007 / -0.0037 / -0.0100 (tempho 1975/1985/1995); even the 96 community
  species score +0.009 / +0.011 / +0.002 / -0.004. Those reports were written by ef7b6a2, which is what
  the reproduction runs.
- 18 jobs queued: caches, ESK projections, reproductions, the first experiments (atlas, readout, planted,
  route ceiling), the corrected suite.

## 2026-10-08 (setup, no data yet)
- Local loop infrastructure is up in WSL2: systemd-managed pueue queue (jobs survive sessions; a 15-min
  canary ran to completion across wsl.exe exits), immutable code snapshots, append-only registry, tick
  journal/LOCK/STOP, generated STATE.md. Full test suite: 938 pass; 2 fail -- one needs data (ref grid),
  one was a real optimizer fragility, now fixed.
- GP-species suite audited and fixed (A1-A8): change skill now only over species whose change is
  resolvable above split-half noise; in-sample rows get exact leave-one-out (local GPs were reproducing
  their own modern noise); noise-only back-transform (local GPs predicted spurious declines); baseline
  scales from every training row; a persistent+changing spacetime kernel as the honest temporal rival;
  persistence baseline; block-safe bootstrap; oracle no longer reads withheld decades.
- Found while testing: the per-species amplitude likelihood is multimodal -- a single-start fit can
  "collapse" a species' amplitude where a far better optimum exists. Every scale fit now multi-starts.
  This also touches DESK's own BLR fits; how much it changes past "rare species collapse" readings is an
  open question for the corrected-suite run (E006).
- DESK training can resume from checkpoints; tempho epoch selection no longer reads withheld years (A9).
- Pre-registered before any data: E004 (atlas) and E005 (downstream-matched readout + kernel variants).
