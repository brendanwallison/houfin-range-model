# SDM libraries locally: kissmig cross-check of B2, and B2-Allee on static suitability (pre-registered 2026-10-30)

Parent: SDM_BASELINE_PLAN_2026-10-30.md; B2 record SDM_BASELINE_B2_2026-10-30.md (NumPy expected-occupancy automaton).
Framing (user, 2026-10-30): B1-dyn's early-East contrast is the equilibrium confound (not-yet-arrived attributed to Z),
not evidence about location quality; any suitability derived from B1-dyn carries it.

## 1. Project-local R library (DONE before this record)
research/r/install_rlib.R -> research/r/lib (untracked; .libPaths = project lib + base R only; user and system
libraries untouched). Windows R 4.5.1 (ucrt). Installed from CRAN binaries 2026-10-30: kissmig 2.0.1, rangr 1.0.10,
terra 1.9.50 (+ Rcpp, assertthat, pbapply, zoo). No failures. Note: the binaries were built under R 4.5.3 (load
warning only). R jobs run at Idle priority with processor affinity cores 0-3 (Windows equivalent of nice/taskset).

## 2. kissmig cross-check of B2 (pre-registered, before any kissmig run)
Why: B2's conclusions (static-suitability colonisation reaches the East decades early with the wrong cohort shape;
the B1-dyn-based arm fits the front) come from my own NumPy automaton. kissmig (Nobis & Normand 2014) is an
independent implementation of the same model class (3x3 neighbourhood, stochastic, suitability = colonisation
probability), so agreement says the B2 findings are not an implementation artefact.
Inputs (same 54 km grid, 67 x 112, ocean NA):
- Suitability matched to B2's fitted establishment: E(t, c) = 1 - exp(-S(t, c) / S0) with S and S0 from the B2 fits
  (B2-static: S0 -> 0, so E = 1 everywhere: kissmig S = NULL; B2-dyn: S0 .263, one layer per year 1902-2025).
  Secondary variant: S01 = min(1, S / q99(S)) from B1-static / B1-dyn (the raw SDM surface).
- Origins: native seed (1,110 cells) from 1902; release (4 cells) from 1940; run separately and combined by taking the
  earlier first occurrence per cell (kissmig takes one origin raster).
- Speed: k iterations per year, k in {1, 2, 3}, pext in {.05, .2} (local extinction between steps; kissmig default 1.0
  is far too high for an annual-step bird range), pcor .2 (default). 20 replicate seeds; per cell, first-occurrence
  year = median over replicates (never -> NA).
- Calibration: for each arm, the (k, pext) whose eastern-cell median first-occurrence lag (vs first BBS positive)
  is closest to B2's own occupancy-front median (p >= .5: static -15, dyn -3). The median match is by construction;
  the tests are the SHAPE and the cells.
Readouts (782 eastern cells, as everywhere): median lag and cohort medians (66-74 / 75-84 / 85-94 / 95-09) vs BBS;
cell-wise Spearman between kissmig and B2 first-occurrence years; median absolute cell difference.
Rule: AGREE if Spearman >= .8 AND every cohort median within +-5 yr of B2's; PARTIAL if Spearman >= .6 or the cohort
TREND has the same sign (late cohorts earlier than early cohorts relative to BBS, for static); otherwise DISAGREE.
Prediction: AGREE for both arms (same model class). LIKELY.

## 3. B2-Allee on STATIC suitability (pre-registered, before fitting)
Why (B2 record): static-suitability colonisation is too early and mis-shaped; B2-dyn fits but its suitability carries
the arrival confound. The remaining candidate that does not need time-varying habitat is density-dependent
establishment (Veit & Lewis 1996: Allee + long-distance dispersal). Test it in the SDM class on STATIC suitability.
Model: B2 (sdm_b2_migsdm.py) with suitability from B1-static, plus an establishment Allee term: colonisation of an
empty cell col = [1 - exp(-pressure)] x E x A, where A = pressure^h / (pressure^h + c50^h) (Hill: lone propagules rarely
establish; h, c50 fitted, h in [1, 6]); and a growth lag: an occupied cell's contribution to pressure ramps up over
its first L years (L in {0, 3, 6}, grid). Fitted exactly as revised B2 (grid over a, r, q, c50, h, L with m and phi
profiled, then Nelder-Mead from the best 3), same training rows.
Readouts: as B2 -- PRIMARY front (10% density, occupancy p >= .5), cohorts, never-reached; held-out paired vs B1-static,
B2-static, step 3; fitted c50/h/L.
Rules:
- ALLEE SUFFICES (in this class): 10% front median within +-5 yr AND cohort spread <= 8.5 (the age model's) on static
  suitability. Then the front's lag-then-acceleration does not require time-varying habitat; an establishment brake
  reproduces it. Ladder implication: the age model's early-East sink is not needed for the front if its own
  establishment structure is strong enough (its relative Allee n50/K .0197 is weak; step 7 tested .05-.20).
- ALLEE INSUFFICIENT: median outside +-10 or cohort spread > 20 with c50/h at bounds or degenerate.
- Otherwise PARTIAL.
- Degeneracy check: if the fit pushes c50 -> 0 (Allee off) and reproduces B2-static, report "Allee not used".
Prediction: PARTIAL to SUFFICES -- an Allee brake delays the early front; whether it also gives the late-cohort
acceleration with static suitability is UNKNOWN.

## RESULTS (sec. 2 and sec. 3 below)
(see the sections appended below)

## ADDENDUM to sec. 2 (after the first kissmig pass, before the extension below was run)
The pre-registered calibration grid cannot reach B2's speed: kissmig moves at most one 54 km cell per iteration, so
with k >= 1 iteration per year and establishment E ~ 1 (B2-static) its slowest front is ~54 km/yr, faster than B2's
fitted spread (first pass: kissmig median -66 at k = 1 vs B2 -15). pext barely matters. Extension (added now, same
rules): k = 1 with the suitability layer multiplied by a uniform factor s in {1, .5, .3, .2, .1, .05} (colonisation
probability per step), pext {.05, .2}; calibrated setting = closest median to B2's, then the same AGREE / PARTIAL /
DISAGREE rule. First-pass numbers are kept in kissmig_crosscheck.csv.

## RESULTS sec. 2 -- kissmig cross-check (E:/Datasets/houfin/grid54/baselines/b0/kissmig/: kissmig_crosscheck.csv, kissmig_crosscheck_ext.csv, logs; scripts research/r/kissmig_crosscheck.R, kissmig_crosscheck_ext.R; export research/exp/sdm_b2_export.py)
kissmig 2.0.1 ran in seconds per setting (20 replicates). Front lags vs first BBS positive, 782 eastern cells; kissmig =
median first-occurrence year over replicates; B2 = own occupancy front (p >= .5).

| arm | kissmig setting | median (B2) | cohorts 66-74 / 75-84 / 85-94 / 95-09 (B2) | cell Spearman vs B2 | median abs cell diff | never |
|---|---|---|---|---:|---:|---:|
| static | k 1, s 1 (pre-registered fastest-slow end) | -66 (-15) | -32 / -42 / -64 / -78 (-9 / -2 / -15 / -35) | .81 | 44 | -- |
| static | k 1, s .2, pext .05 | -34 (-15) | -26 / -24 / -33 / -50 | .97 | 17 | -- |
| static | k 1, s .1, pext .05 | -4.5 (-15) | -22 / -8 / +4 / -17 | .79 | 15 | -- |
| static | rule's "closest median": s .05, pext .2 | -7.5 (-15) | -16 / -11 / +5 / +1 | -.19 | 15 | .88 (DEGENERATE: 88% never reached) |
| dyn | k 1, s .5, pext .2 (calibrated) | -4 (-3) | -16.5 / -7 / 0 / -5 (+2 / 0 / -2 / -7) | .85 | 4.5 | .003 |

Rule outcome:
- STATIC: PARTIAL. The rule's mechanical calibration picked a degenerate setting (stochastic extinction: 88% of cells
  never reached), so it is not used. No non-degenerate kissmig speed matches B2's median exactly (s .2 -> -34, s .1 ->
  -4.5: kissmig's discrete 1-cell steps plus stochastic loss make the speed jump between these). At every
  non-degenerate speed kissmig reproduces B2's qualitative finding: with static suitability the LATE cohorts arrive far
  too early relative to the early cohorts (s .2: -26 / -24 / -33 / -50; B2: -9 / -2 / -15 / -35), cell ranks agree
  (Spearman .97 at s .2). ESTABLISHED: B2-static's "constant-speed spread gets the cohort shape wrong" is not an
  artefact of my NumPy automaton.
- DYN: PARTIAL. Median matches (-4 vs -3), cells agree (Spearman .85, median cell difference 4.5 yr), the three later
  cohorts are within 5 yr, but the first cohort (1966-74) arrives 16.5 yr early in kissmig vs +2 in B2 (max cohort
  difference 18.5 > 5). Likely mechanism: kissmig's minimum colonisation step (one whole 54 km cell per success, no
  density) lets the earliest front run ahead where B2's continuous occupancy builds up gradually (LIKELY).
- Also established by the first pass: kissmig with k >= 1 iteration/yr and full establishment cannot be slower than
  ~54 km/yr; the House Finch's fitted spread is slower, so any kissmig use on this grid needs a sub-unit suitability
  scaling (or k < 1, which the package does not allow).

## RESULTS sec. 3 -- B2-Allee on STATIC suitability (/mnt/e/Datasets/houfin/grid54/baselines/b0/b2allee/; code research/exp/sdm_b2_allee.py)
Fit (grid 360 points, then Nelder-Mead from the best 3; all three NM runs stopped at 300 iterations, best two within
.0005 of each other): spread scale 60 km, maturation lag L = 3 yr, r .095, LDD q ~0, Allee c50 .036 (in units of
propagule pressure), Hill h 1.55, S0 .077, e0 .013. Training mean lpd -1.2768 (B2-static -1.2805): the Allee term is
USED (c50 well above 0; not degenerate) and buys .004.

| model (static suitability unless noted) | 10% front median | cohorts 66-74 / 75-84 / 85-94 / 95-09 | spread | occupancy p >= .5 median (cohorts) | S-real |
|---|---:|---|---:|---|---:|
| age model step 3 | -4 | +2 / -1 / -3 / -6.5 | 8.5 | -- | -1.918 |
| B2-static | -33 | -28 / -22 / -32 / -44 | 22 | -15 (-9 / -2 / -15 / -35) | -2.156 |
| B2-Allee | -23.5 | -26 / -15 / -21 / -39 | 24 | -16 (-17 / -7 / -14 / -33.5) | -2.175 |
| B2-dyn (Z-gated; carries the arrival confound) | -5 | -1 / -4 / -5 / -7 | 6 | -3 | -2.046 |
Held-out (paired): B2-Allee minus B1-static S-real +.026 [.007, .048], T marginal +.011; age model minus B2-Allee
S-real +.257 [.110, .372] (midwest +.43, east +.42), T marginal +.117.

Rule outcome: NOT SUFFICIENT (PARTIAL by the letter: median -23.5 is outside +-10 and the cohort spread 24 > 20, but
c50 / h are not at bounds, so the pre-registered INSUFFICIENT clause, which requires degeneracy, is not met). The
establishment brake delays the median by ~10 yr relative to B2-static but leaves the cohort shape unchanged -- the
1995-2009 cohort still arrives ~39 yr early. ESTABLISHED for this model class: an Allee/maturation brake on static
suitability does not reproduce the lag-then-acceleration either.
Caveat found here (applies to all B2 variants; NOT pre-registered, reported as a finding to check): the native seed
reaches -98W, and in every static-suitability B2 fit the native population spreads east across the Plains on its own
(east-of-95W occupied area reaches 10 cells by 1923 and 100 by 1940, before the release can matter). The late-cohort
"too early" pattern may therefore come largely from the WESTERN front crossing the Plains early, not from the release
front -- the same mechanism the age model's front-speed diagnostic found with frozen habitat ("native seed too: the
West crosses the Plains before 1940"). Suggested next check (not run): B2 / B2-Allee with the native seed confined
west of the Rockies or with release-only dynamics east of 95W, to separate the two fronts.
