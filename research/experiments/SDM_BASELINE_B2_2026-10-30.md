# SDM baseline B2 -- dispersal-limited SDM (MigClim class) on B1-static vs B1-dyn suitability (pre-registered 2026-10-30, before fitting)

Parent: research/experiments/SDM_BASELINE_PLAN_2026-10-30.md sec. 4 B2; B1 results in SDM_BASELINE_B1_2026-10-30.md.
Same design table, scorer and front readout as B0/B1 (GRID54 step 3 rows and masks). Code: research/exp/sdm_b2_migsdm.py.
CPU, 2 threads, taskset 0-3, nice 19.

Question: does a colonisation process on SDM suitability -- with no demography, no Allee, no density dependence --
reproduce the eastern front timing (PRIMARY readout), and does it need Z's temporal gating to do so? GRID54's
front-speed diagnostic (established for step 3): with habitat frozen at its 1966-2025 mean, the fitted demography and
dispersal fill the East 26-30 years early; the fit holds the front back with a deep early-East fundamental sink. B1
found that a Z-only SDM's expected counts in the early East are ~12x below modern (B1-dyn) -- the equilibrium
confound (not-yet-arrived read as a Z effect; user correction in SDM_BASELINE_B1), not evidence about location
quality. Static suitability (B1-static) has no such temporal gating.

## Model (expected-occupancy cellular automaton; MigClim's rules made continuous and deterministic)
- Suitability S(t, c): the B1 SDM's expected count at u = 0 (observer effect removed), from B1-static (lambda 100,
  constant in time) or B1-dyn (lambda 1000, year-matched Z, 1902-2025), refitted here on the same training rows.
- State p(t, c) = probability a cell is occupied. Seeds: native range = cells with initpop_shape > 0 in the GRID54 BBS
  input (the age model's native seed region), p = 1 in 1902; release = the age model's 4 release cells
  (inv_locations), p = 1 from 1940.
- Each year: colonisation of empty cells c: col = [1 - exp(-r sum_j p_j K(d_cj))] x E(t, c), with
  K(d) = (1 - q) exp(-d / a) + q exp(-d / 400 km) (short scale a, long-distance share q), j != c;
  establishment E = 1 - exp(-S(t, c) / S0). Local loss of occupied cells: ext = e0 exp(-S(t, c) / S0).
  p <- p (1 - ext) + (1 - p) col.
- Counts: E[y] = m p(t, c) S(t, c) x exp(u_observer) with the B1 fit's observer x route effects and sigma .97; NB2,
  phi refitted. Parameters r, a, q, S0, e0, m (6) + phi, fitted by maximum NB likelihood on the TRAINING rows (real
  BBS, outside both held-out sets), Nelder-Mead on transformed parameters, 3 starts, best kept.
- Arms: B2-static (S from B1-static), B2-dyn (S from B1-dyn).

## Readouts
PRIMARY: the front (M-F, the same code that reproduced step 3's front_speed.json): first year expected count >= 1% /
10% of the 2010-25 median over the 782 eastern cells, minus first BBS positive; median, by arrival cohort, never
reached. Also the occupancy front (first year p >= .5). Secondary: held-out T marginal / conditional, S-real (paired vs
step 3 and vs the same arm's B1), region strata; fitted parameters (spread scale, LDD share); year the release
population's occupied area (sum p, east of 95W) reaches 10 / 100 / 500 cells.

## Predictions (mine)
- P1: B2-static reaches the median front within +-5 yr at 10% (spread speed is free) BUT with a cohort trend: a
  constant-speed automaton cannot reproduce the observed lag-then-acceleration, so early cohorts arrive late or late
  cohorts early (cohort spread >= 15 yr). LIKELY.
- P2: B2-dyn fits the cohort pattern better than B2-static (Z's gating shapes the timing). LIKELY given B1's ratio.
- P3: both B2 arms beat their B1 counterparts on S-real in the east (dispersal limitation removes predicted birds
  where there are none early). LIKELY; and both stay below the age model. LIKELY.

## Rules (implications reported, not decided)
- R1 If B2-static's 10% front median is within +-5 yr of the BBS AND its cohort spread (max - min of cohort medians)
  is <= the age model's (8.5 yr at 10%: +2 .. -6.5): the eastern front timing does not need time-varying habitat;
  dispersal limitation on static suitability suffices. Ladder implication: the early-east sink is not required by
  the front; the next lever is the age model's spread / establishment structure (dispersal kernel, Allee), not
  habitat amplitude.
- R2 If B2-static fails (|median| > 10 or cohort spread > 20) while B2-dyn passes: within this model class the front
  needs time-varying suitability -- but B2 has no density-dependent establishment, so this cannot separate "habitat
  gated the front" from "an Allee / establishment lag gated it". Report as unresolved between those two.
- R3 If both fail: neither static nor Z-gated suitability with simple colonisation explains the front; the shape
  needs density-dependent establishment (Veit & Lewis 1996: Allee + LDD), which the age model has.
- If B2-static's fitted spread is implausible (a < 5 km with q ~ 0 or r pinned at a bound), report the fit as degenerate.

## REVISION of the fitting procedure (2026-10-30, after the first B2-static fit; rules and readouts unchanged)
The first B2-static fit (one Nelder-Mead start per spread scale, all from r = 1) was DEGENERATE: every cell east of
95W was occupied by ~1908 (the native seed spreads east immediately), m ~ 1, and the training fit equalled B1-static
(-1.2964) -- the likelihood is flat in the spread parameters once occupancy saturates, so the optimiser never left
the saturated plateau. Its front (-85 yr at 10%) is therefore a property of the start, not a fitted result; it is not
used. Revised procedure: coarse grid over a {30, 60, 120 km} x r {.001 ... 1, 7 values} x q {0, .02, .1} with the
count scale m and phi profiled at each point (S0 = median modern suitability / 3, e0 = .01), then Nelder-Mead over all
six parameters from the best three grid points. The first run's B2-dyn arm was stopped before finishing.

## RESULTS (2026-10-30, revised fitting; /mnt/e/Datasets/houfin/grid54/baselines/b0/b2/b2_results.json, b2.log; the degenerate first run kept as b2_degenerate_first.log / b2_results_degenerate_first.json)
Seeds: 1,110 native cells (initpop_shape > 0), 4 release cells (inv_locations) from 1940.

### Fits
| arm | train mean lpd (B1 same arm) | spread scale a | r | LDD share q | S0 | e0 | m | NM status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| B2-static | -1.2805 (B1-static -1.2964 level) | 60 km | .023 | ~0 | ~2e-6 | ~1 | 1.08 | max iterations (2 of 3 starts agree to 1e-7) |
| B2-dyn | -1.2371 | 120 km | .019 | ~0 | .26 | .031 | 1.08 | converged |
B2-static drove S0 -> 0: establishment E = 1 everywhere and local loss ~0, i.e. suitability plays no role in
colonisation -- a pure distance-limited spread on top of the static count surface. Both arms chose no long-distance
dispersal (q -> 0).

### PRIMARY: front timing (782 eastern cells; model first year minus first BBS positive)
| model | 10% density: median (cohorts 66-74 / 75-84 / 85-94 / 95-09) | cohort spread | 1% density median | occupancy p >= .5: median (cohorts) | never reach 10% |
|---|---|---:|---:|---|---:|
| age model step 3 | -4 (+2 / -1 / -3 / -6.5) | 8.5 | -8 | -- | 4% |
| B1-static (no dispersal) | -89 | -- | -90 | -- | 8% |
| B1-dyn (no dispersal) | -16 (-10 / -13 / -16 / -19) | 9 | -67 | -- | 6% |
| B2-static | -33 (-28 / -22 / -32 / -44) | 22 | -43 | -15 (-9 / -2 / -15 / -35) | 8% |
| B2-dyn | -5 (-1 / -4 / -5 / -7) | 6 | -12 | -3 (+2 / 0 / -2 / -7) | 9% |
East of 95W occupied area (sum p) reaches 10 / 100 / 500 cells: B2-static 1923 / 1941 / 1968 (the native seed's
eastward spread arrives before the release matters); B2-dyn 1942 / 1977 / 1988.

### Secondary: held-out (paired, 90% 216 km block bootstrap)
| comparison | T marginal | T conditional | S-real | S-real east | S-real midwest |
|---|---|---|---|---|---|
| B2-static minus B1-static | +.017 [.014, .020] | +.014 | +.045 [.011, .070] | +.093 [.011, .119] | +.035 |
| B2-dyn minus B1-dyn | +.002 [-.002, .005] | -.001 | +.037 [.015, .056] | +.060 [.002, .087] | +.068 |
| age minus B2-static | +.111 [.091, .132] | +.089 | +.238 [.106, .337] | +.382 | +.407 |
| age minus B2-dyn | +.091 [.070, .112] | +.077 | +.128 [.075, .166] | +.173 [.069, .211] | +.184 |
B2-dyn S-real -2.046 (B1-dyn -2.083, S-IDW -2.050, age -1.918).

### Predictions vs outcome
- P1 FAILED in its first half and HELD in its second: B2-static does not reach the front within +-5 yr (10% density
  -33; occupancy -15), and its cohorts trend (-22 to -44 at 10%; -2 to -35 on occupancy): a constant-speed spread is
  far too early for the late cohorts.
- P2 HELD: B2-dyn fits the median and the cohort pattern (-5; spread 6, tighter than the age model's 8.5).
- P3 HELD: dispersal limitation improves both SDMs on S-real (+.045 / +.037), most in the east and midwest; both
  stay well below the age model (+.13 to +.24).

### Rules
- R1 NOT met (B2-static: |median| 33 > 10 at 10%; cohort spread 22 > 20).
- R2 MET: B2-static fails, B2-dyn passes. Within this model class the observed front needs time-varying suitability
  -- with the user's correction applied: B2-dyn's suitability is B1-dyn's, which already absorbed "not yet arrived" as
  a Z effect (equilibrium confound; fitted to the same 1966-2020 counts whose arrival timing the front readout scores).
  So B2-dyn's good front is PARTLY CIRCULAR (LIKELY): the timing is learned into the suitability from the counts, not
  predicted. And B2 has no density-dependent establishment, so even without the circularity it could not separate
  "habitat gated the front" from "an Allee / establishment lag gated it" (pre-registered caveat). Reading: UNRESOLVED
  between the two; what IS established is narrower --
- ESTABLISHED (this model class, these data): simple distance-limited colonisation on a static suitability surface
  (fitted spread ~60 km scale, no LDD, no establishment threshold) reaches the East decades too early and with the
  wrong cohort shape (late cohorts far too early). Constant-speed spread does not produce the observed lag-then-
  acceleration; something must slow the early front -- a time-varying gate (habitat or its proxy) or density-dependent
  establishment (Allee). This matches the age model's front-speed diagnostic (frozen habitat: East filled 26-30 yr
  early) from an entirely different model.
- Implication for the ladder (reported): the dispersal-only answer is ruled out for this class; the comparison that
  separates the two remaining candidates is the age model's own dispersal / establishment levers (Allee profile,
  front-local corrections, GRID54 step 20) with habitat change held to what non-invading residents share (the Z
  attribution found the House Finch dims only partly shared east of 80W). A B2 variant with an Allee-type
  establishment term (colonisation probability rising with local p or neighbour density) on STATIC suitability would
  test the Allee route in the SDM class without the circular suitability -- not run; pre-register before running.
