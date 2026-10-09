# DIGEST (newest first)

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
