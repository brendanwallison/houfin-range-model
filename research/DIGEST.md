# DIGEST

## CURRENT STATE (rewritten each round; read this first -- the dated log below keeps the history)
Last updated 2026-10-10 ~07:50 (overnight run + your 'why' question) (you said to follow the plan or revise it on the results). "Solid" =
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

TERMS (fixed 2026-10-10 after you asked):
- "Change" is always between multi-year AVERAGES (6-21-year windows strip weather and survey noise), never year to
  year. Gaps between window centres: readout/oracle (E013, E030) and ESK retention 1966-86 vs 2005-25, ~40 years;
  corrected suite (E006) 1966-75 vs 2005-25, ~45; span/reach backcasts (E023-E029) a 6-year window vs the first ten
  trained years, ~8-32 (1966-71 vs 1996-2005 = 32); reach curve (E019, E025) vs 2005-25, ~22-46; observer fading
  (E028) 2005-14 vs 2016-25, 11. "Decadal" below means over one to several decades, mostly 20-45 years.
- "Regional" = smooth over a few hundred km: the trend placebo and E030's regional field are Gaussian-smooth maps at a
  300 km scale (places 300 km apart correlate ~0.6, 600 km ~0.14); E028's surface uses 150 km. "Local" = one 27 km
  cell (one or a few BBS routes). "Continental" = the whole grid.

NEW SINCE YESTERDAY EVENING (tentative unless noted; each finding carries WHY: established / likely / unknown)
- Why the readout of change does no better than "no change" -- even for the TRUE community (E030, E011, E013).
  Squared-error skill against no change = (variance of the predicted change / variance of the true change) x (2k - 1),
  where k is the slope of truth on prediction: a prediction must have k > 1/2 to win, whatever its correlation.
  WHY (established, one run): three factors multiply --
  - information: the true community's ~40-year change (1966-86 vs 2005-25) carries only ~5-7% of a species' change
    over the same interval (corr 0.23-0.27). Species change is strongly regional (a 300 km smooth map of the species'
    own change, fitted on training places, explains 23% on held-out places), but species-specific: the rest of the
    community does not share it;
  - scaling: the readout's coefficients come from differences between places, which are large, so applied to change
    over time they predict changes ~2x too big for their accuracy (k ~0.5, the break-even point);
  - noise: the oracle's half-window features carry survey noise the large coefficients amplify (k -> ~0.4).
  Rare species carry almost no information and pull the median down. DESK in withheld decades has almost none
  (corr 0.07, k 0.19); in trained years it reaches corr 0.22, k 0.70 and beats no change.
  Is the low information a defect? Mostly not (E031, your test): about 9 in 10 parts of a species' 40-year change
  are not shared even with the five species it lives with most closely -- its own story, which your population model
  is meant to carry. The ~1 in 10 that is shared is real (random species share ~0.3%; it survives where the same
  observer counted throughout) and the community summary already carries most of it (5-7% vs 8-9%). So ESK's loss to
  no change in the score comes from over-scaling and noise, not from missing habitat. Note: the score asks habitat to
  predict a species' whole change, which your model does not ask of it.
- Training span vs reach (your question), E023 / E024 / E024b:
  - cheap linear surrogates: at a fixed reach a SHORTER span next to the target backcasts better (strong: 0.22 vs
    0.08 for 20- vs 40-year spans);
  - DESK: two seeds of a 20-year span next to the target beat the 40-year model by +0.038 [0.020, 0.057] at the 1986
    start year (pre-registered rule met after a second seed overturned my first "no"); nothing at the 1996 start
    (one seed). A time-weighted DESK (all years, those near the target weighted up) is training now (E029, ~09:00).
  WHY (likely, not shown for DESK): decadal trends drift; a long span averages in later decades whose trends do not
  describe the earlier ones (shown for the surrogates, E023).
- Where the covariates' temporal signal is (E027, E027b-d, two skeptic rounds): their backcast skill is largely matched
  by a placebo using them only as a STATIC position (a place's average environment) times LINEAR YEAR. A place's own
  covariate change adds at most ~0.05 in correlation (two independent splits); the direction the continent's
  covariates moved does not matter; slow 30-year history adds nothing; DESK already holds most of what the surrogates
  extract.
  WHY (unknown). Candidates: local covariate change at 27 km is small or smoothed relative to its noise (interpolated
  products: decadal HYDE, LUH, coarse climate); responses are lagged and heterogeneous (slow averages did not help);
  the drivers are not in the covariates. Whether the linear trend is habitat or observation is also open.
- DESK keeps changing after the epoch we select (E025): by epoch 490 its backcast direction is better (+0.01 to
  +0.08), its pull toward climate analogs halves (+0.14-0.21 -> +0.03-0.09), but it moves even more than its accuracy
  justifies (squared error vs no change 1.27 -> 1.47 at 27.5 years) and fits held-out places worse. Production ships
  FINAL-epoch weights.
  WHY (partly likely): early in training DESK has learned the spatial mapping and moves along it (space-for-time);
  later epochs add movement in other directions. Why those later directions are better, and why movement keeps
  outgrowing accuracy, is unknown.
- The metric term's pair count is not a lever (E026: 4,096 and 262,144 pairs backcast like 65,536).
  WHY (likely): the term's gradient noise was not what limited temporal learning -- the logged withheld-decade
  direction accuracy was the same in every run from epoch ~100 on.
- The corrected validation suite (E006), your three concerns: on HELD-OUT places DESK is the best arm (beats the
  spacetime GPs and the true-community oracle), but still ~no change on species change; the oracle is also ~no change
  (WHY: above). DESK "loses to the spacetime GP" only on trained places, where the GP's degenerate fit predicts that
  each place's modern quirks were smaller in the past.
  WHY (established, E028/E028b, three eras, skeptic-reviewed): where observers changed, 55-60% of that fading is the
  change of observer; the rest (~7-8% per 11 years) is real fading of local deviations of unknown cause -- it does not
  track local covariate change. The covariate GP's fit is broken (A13).
- Replications on fresh seeds of the current code: the decline with reach holds (seed spread 0.02-0.03); the pull
  toward climate analogs, the readout bottleneck and the over-narrow intervals all hold. Current-code retrains track
  the direction of change better than the archived ones at long reach, but are no better than no change in squared
  error. WHY the archived models are worse: unknown -- not the metric-pair count (E026); epoch selection explains at
  most part of it.

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
- "A shared continental covariate trajectory x environment carries the signal" (E027c, first reading): survey
  composition plus linear year; random directions do as well.

WHAT THIS DOES AND DOES NOT TELL US
- Everything above characterizes the current DESK, its readout, and linear or simple nonlinear surrogates on 10-50
  training years. It diagnoses where today's system loses temporal signal; it is not a limit on how well
  extrapolation in time can work.
- Fallbacks to keep in view, NOT the plan: recalibrating DESK's temporal deviations on its trained years; holding
  pre-BBS habitat near its BBS-era state with uncertainty that grows with reach.

OPEN QUESTIONS (the plan: what limits extrapolation in time, and what would extend it)
1. Training state: later epochs give better direction and less space-for-time pull but more over-movement (E025);
   gradient noise in the metric term is not a lever (E026). What sets the over-movement, and can it be calibrated
   out downstream without losing the direction gain?
2. Covariate content (T5): the covariates' decadal temporal signal at this grain is small beyond "kind of place x
   linear year" (E027c/d: at most ~0.05 in correlation). What inputs or grain would carry it, and is the linear trend habitat or observation?
3. The readout: even the true community's change, read through the spatial readout, predicts species change no
   better than no change. A readout that lets local deviations fade with reach (E028: half of that fading is real)
   is a candidate.
4. The scoreable target: observer turnover inflates "true change" and rewards models that predict a return to the
   regional mean; grade on the observer-adjusted part.
The age-model sensitivity fit comes last, once there are candidate backcasts worth comparing.

## GATE 1 REPORT (2026-10-10 ~03:24, draft for your review; all UNVERIFIED in the protocol's sense unless noted)
1. Your three concerns, on the corrected validation suite (E006: audit fixes A1-A8, tempho1995 and the base model):
   - "DESK loses to the spacetime GP": only on TRAINED cells in withheld decades, where the GP's (degenerate) fit
     pulls each cell's modern quirks toward its region -- and about half of that advantage is observer turnover
     (E028, replicated in three eras). On HELD-OUT cells, the setting a backcast needs, DESK beats or ties both
     spacetime GPs: +0.025 to +0.07 in trained years, +0.049 / ~0 in withheld decades.
   - "DESK ~ no change on change": still true: +0.010 (held-out cells, trained years), -0.008 / -0.010 (withheld
     decades).
   - "The oracle is weak too": yes. The oracle built from the TRUE community is also at or below no change (-0.002,
     -0.014, -0.029), and DESK beats it on held-out cells (+0.025 to +0.034). Why (E030): a species' decadal change is
     mostly its own regional dynamics -- the community's change carries ~5-7% of it; the readout's coefficients,
     learned from differences between places, over-scale what little it carries (truth-on-prediction slope ~0.5, the
     break-even); and the oracle's survey noise, amplified by those coefficients, tips it below no change.
   - The covariate-GP baseline is broken in the corrected suite too (absurd predictions; A13) and must not be quoted.
2. Which stages lose temporal signal:
   - ESK: no selective loss; it keeps the regionally coherent part of change like comparable spatial structure.
   - DESK: direction of withheld-decade change correlates 0.2-0.4 with the truth, falling with reach; it moves more
     than its accuracy justifies, leans toward climate analogs, and captures ~5-9% of the noise-free temporal
     variance. Training longer (production ships final-epoch weights) improves direction a little and over-movement
     more (E025).
   - Readout: even the true community's change, read through the spatial readout, correlates only ~0.23 with species
     change and is no better than no change in squared error; DESK's reaches 0.07-0.10.
   - Inputs: a place's own covariate change adds little to a backcast -- at most ~0.05 in correlation beyond "kind of
     place x linear year" (E027c + skeptic + a second split, E027d). This is the sharpest constraint found so far: at
     27 km and decadal scales, the covariates as they are carry little temporal signal beyond a trend surface.
3. How much real change there is: most species have resolvable change between epochs (323 / 401 and 249 / 372),
   noise ceilings 0.62-0.67; ~30% of change energy is observer turnover (E021); local deviations from the regional
   surface also fade for real, by ~7-8% per 11 years, unrelated to local covariate change (E028).
4. What I would look at next (your call at this gate): (a) T5 directly -- which drivers or which grain would carry
   temporal signal the current covariates do not (e.g. finer land-use or habitat products, or a coarser grain where
   local noise averages out), and whether the "kind of place x year" trend is habitat or observation; (b) the
   readout, which loses the most; (c) over-movement and epoch selection, since production ships the most
   over-moving state; (d) the second span seed (E024b, running) and the metric-pair runs (E026, running) settle the
   time-weighting and optimization questions. (An "environment x continental trajectory" backcast model, which I
   floated earlier tonight, is withdrawn: it reduces to extrapolating a linear trend.)

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
- Skeptic round (~00:00): E024's "no" rests on one window and one seed (pooled +0.03 at T0 1986) -> second seed of the
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
