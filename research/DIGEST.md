# DIGEST

## CURRENT STATE (rewritten each round; read this first -- the dated log below keeps the history)
Last updated 2026-10-09 ~23:40, overnight (you said to follow the plan or revise it on the results). "Solid" =
independently re-derived, replicated on a second split or seed, and survived a skeptic review. "Tentative" = one run,
or checks still pending. Experiment numbers point to research/experiments/EXXX.md.

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

NEW TONIGHT (tentative: one run each unless noted)
- Training span vs reach (your question), E023 / E024:
  - for cheap linear surrogates, at a fixed reach a SHORTER span next to the target backcasts better -- later
    decades' trends do not describe earlier ones (strong: e.g. 0.22 vs 0.08 for 20- vs 40-year spans);
  - for DESK itself, much weaker: retrained on 20-year spans, it gains about +0.03 (pooled over windows) at one start
    year and nothing at another; the third comparison was spoiled by a badly selected run. The pre-registered test
    said "no", but the skeptic showed it rested on one window and one seed, so a second seed is queued before the
    time-weighted DESK is ruled in or out.
- Is there temporal signal in the covariates that DESK misses? (E027, E027b) Mostly no. My first reading of E027
  ("slow history and nonlinear responses keep their skill at long spans") did not survive the skeptic: it came from a
  weak 64-component linear reference, a lucky random-feature draw, a 30-year average that was half its 1940 starting
  value, and the trend placebo fading at long spans. Added to DESK's own features, slow covariate history adds only
  +0.02-0.05. A cleaner test (a continental-clock placebo) is next.
- The corrected validation suite on tempho1995 (E006), for your three concerns:
  - DESK is still slightly worse than no change on species change -- and so is the oracle built from the TRUE
    community; on held-out cells DESK beats the oracle. Most of the loss is downstream of DESK (community -> species
    readout, plus noise), as E013 found from the other side.
  - DESK "loses to the spacetime GP" only on trained cells. There the GP's fit degenerated into "each cell's modern
    quirks were smaller in the past". In cells whose observers changed, about half (~57% after correcting for noise)
    of that fading is the change of observer; the rest is real fading of local deviations of unknown cause -- it does
    not track local covariate change (E028, survived the skeptic; 2005-2025 only, earlier eras queued). DESK's readout
    cannot represent this fading. On held-out cells the GP does no better than DESK.
  - The covariate GP's fit is broken in the corrected suite too (absurd predictions; A13).
- Replications on fresh seeds of the current code: the decline with reach holds (seed spread 0.02-0.03); the pull
  toward climate analogs, the readout bottleneck and the over-narrow intervals all hold. Current-code models track
  the direction of change better than the archived ones at long reach but are no better than no change in squared
  error, because they move more.
- DESK keeps changing after the epoch we select. From the selected epoch (65-162, chosen on a spatial score) to the
  end of training, its direction accuracy on withheld decades is flat to rising while its movement doubles.
  Production ships FINAL-epoch weights (it had no validation set), so the models we have been grading may not behave
  like production. Grading the late states now (E025).

TENTATIVE (earlier, still standing)
- As DESK reaches further back, its useful temporal signal fades and its movement increasingly follows
  space-for-time (alignment with the climate-analog direction +0.02 / +0.16 / +0.21 at 1-10 / 11-20 / 21-30 years of
  reach; real communities ~0 or slightly against it).
- Space-for-time fails at the covariate stage: coefficients learned by comparing places at one time predict none of
  the decadal change. Regional trend persistence predicts some of it; linear covariate maps add only +0.03-0.05.
- The readout cannot tell when it is guessing: change intervals far too narrow, and leverage does not flag the
  errors.
- About 30% of what every change metric counts as "true change" is observer turnover.
- Rank 64 would add little over 24.

OVERTURNED (kept so the corrections are visible)
- "A change-fitted readout captures 25-29% of species change from community change": spatial interpolation.
- "The true community's change, read through spatial betas, is worse than no change": errors-in-variables; ~zero.
- "The split kernel works through DESK's temporal information": a placebo does as well.
- "ESK keeps 23% of temporal change because time is small": confounded by the route nugget and survey effort.
- "DESK's backcast errors are not pulled toward space-for-time analogs": the corrected test shows they are.
- "Intervals ~3x too narrow": ~2x on a stable floor.
- "Train DESK on within-cell covariate variation": the within map was a regional-trend extrapolator.
- "Longer reach fails because the tempho models have fewer training years" (my working assumption before E023/E024):
  true for linear surrogates; for DESK at most a small effect.
- "With all history, slow and nonlinear covariate features keep backcast skill where linear maps lose it" (E027,
  first reading): reference, random-draw and warm-up artifacts.

WHAT THIS DOES AND DOES NOT TELL US
- Everything above characterizes the current DESK, its readout, and linear or simple nonlinear surrogates on 10-50
  training years. It diagnoses where today's system loses temporal signal; it is not a limit on how well
  extrapolation in time can work.
- Fallbacks to keep in view, NOT the plan: recalibrating DESK's temporal deviations on its trained years; holding
  pre-BBS habitat near its BBS-era state with uncertainty that grows with reach.

OPEN QUESTIONS (the plan: what limits extrapolation in time, and what would extend it)
1. Optimization: does DESK under-learn temporal structure? It takes one optimizer step per epoch (500 in all), and its
   only cross-year supervision is a noisy pair-sampled term. E025 (late epochs) and E026 (4,096 / 262,144 metric
   pairs vs 65,536; the archived models used 4,096) are running.
2. Covariate content: DESK already carries most of what simple covariate surrogates extract (E027b). E027c asks
   whether ANY covariate representation carries cell-specific temporal signal beyond a continental clock; if not,
   the inputs, not DESK, bound temporal extrapolation at this grain (T5).
3. The readout: even the true community's change, read through the spatial readout, predicts species change no
   better than no change. A readout that lets local deviations fade with reach (E028: half of that fading is real)
   is a candidate.
4. The scoreable target: observer turnover inflates "true change" and rewards models that predict a return to the
   regional mean; grade on the observer-adjusted part.
The age-model sensitivity fit comes last, once there are candidate backcasts worth comparing.

---- dated log (newest first) ----

## 2026-10-09 night: overnight run (E024 decided; E006, E019b, E027, E028 in; E025-E027b running)
- E024 (DESK on 20-year spans): pre-registered rule NOT met -> no time-weighted DESK. E023b table in E024.md.
- E019b + seed re-runs of E013/E017/E018/E020: everything replicates on the current code (records updated).
- E006 tempho1995 (corrected suite): desk -0.008 / oracle -0.014 (TIME), desk -0.010 / oracle -0.029 (SPACE_TIME) vs
  no change; spacetime_sum +0.119 on TIME is a degenerate damped-persistence fit; covariate GP broken (A13).
- E028: damped persistence by observer continuity: slope -0.182 (observers replaced) vs -0.084 (unchanged).
- E027: slow / nonlinear covariate features hold up at long spans (+0.07 to +0.11 over linear+trend), not at short.
- Training logs: selected epochs 65-162; withheld-decade direction flat-to-rising and movement doubling to epoch 490
  -> E025 (late states); archived models trained with 4,096 metric pairs vs 65,536 now -> E026.
- Archived DESK sweep logs: 16 optimizer steps per epoch (tiles32) and heavier metric weights did not raise held-out
  direction accuracy in trained years (weak evidence against "more steps" for T11).
- Skeptic round (~00:30): E024's "no" rests on one window and one seed (pooled +0.03 at T0 1986) -> second seed of the
  1986-2005 span queued; E027's headline withdrawn (weak 64-PC reference, lucky RFF draw, lag30 warm-up, trend decay)
  -> E027c; E028 survives (observer share ~57% disattenuated) -> earlier-era replication E028b.


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
